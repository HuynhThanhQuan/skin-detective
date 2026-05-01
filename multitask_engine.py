"""Joint train / evaluate loops for the multi-task pipeline.

Mirrors the original ``engine.py`` API (``train_one_epoch`` / ``evaluate``) so it
plugs into the existing logging + utils, but additionally tracks the severity
loss on training and severity accuracy on evaluation.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Dict

import torch
import torch.nn.functional as F

import torchvision.models.detection.mask_rcnn  # noqa: F401  (kept for backwards-compat with utils)

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import utils
from multitask_configs import SEVERITY_CLASSES, SEVERITY_IGNORE_INDEX


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    if writer is not None:
        for k, meter in metric_logger.meters.items():
            writer.add_scalar(f"train/{k}", meter.global_avg, epoch)


@torch.no_grad()
def evaluate(model, data_loader, device, writer, epoch: int = 0):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox"])

    sev_correct = 0
    sev_total = 0
    sev_confusion: Dict[int, Dict[int, int]] = {
        i: {j: 0 for j in range(len(SEVERITY_CLASSES))} for i in range(len(SEVERITY_CLASSES))
    }

    for images, targets in metric_logger.log_every(data_loader, 100, "Test:"):
        images = [img.to(device) for img in images]
        targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = model(images)
        model_time = time.time() - t0

        outputs_cpu = [{k: (v.to(cpu_device) if torch.is_tensor(v) else v) for k, v in o.items()} for o in outputs]
        res = {
            t["image_id"].item(): {k: v for k, v in o.items() if k in ("boxes", "labels", "scores")}
            for t, o in zip(targets_dev, outputs_cpu)
        }
        coco_evaluator.update(res)

        # Severity accuracy (skip samples with ignore label).
        for t, o in zip(targets, outputs_cpu):
            gt = int(t["severity"].item())
            if gt == SEVERITY_IGNORE_INDEX:
                continue
            pred = int(o["severity_pred"])
            sev_total += 1
            sev_correct += int(pred == gt)
            sev_confusion[gt][pred] += 1

        metric_logger.update(model_time=model_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    sev_acc = (sev_correct / sev_total) if sev_total else float("nan")
    print(f"Severity accuracy: {sev_acc:.4f}  (n={sev_total})")
    for i, name in enumerate(SEVERITY_CLASSES):
        row = sev_confusion[i]
        row_total = sum(row.values()) or 1
        print(f"  GT={name:8s} -> " + ", ".join(
            f"{SEVERITY_CLASSES[j]}={row[j]/row_total:.2f}" for j in range(len(SEVERITY_CLASSES))
        ))

    if writer is not None:
        writer.add_scalar("eval/severity_acc", sev_acc, epoch)
        bbox = coco_evaluator.coco_eval.get("bbox")
        if bbox is not None and bbox.stats is not None:
            writer.add_scalar("eval/AP", float(bbox.stats[0]), epoch)
            writer.add_scalar("eval/AP50", float(bbox.stats[1]), epoch)
            writer.add_scalar("eval/AP75", float(bbox.stats[2]), epoch)

    torch.set_num_threads(n_threads)
    return {"coco_evaluator": coco_evaluator, "severity_acc": sev_acc}
