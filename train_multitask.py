"""Single-stage multi-task training: detection + severity grading in one model.

Usage examples:

    python train_multitask.py --data ./data/final/f_circle/ds \
        --grade-csv ./data/final/grade1/label.csv ./data/final/grade2/label.csv \
        --model fasterrcnn_r50_fpn --severity-head attention --epochs 100

    python train_multitask.py -d ./data/final/f_circle/ds \
        --grade-csv ./data/final/grade1/label.csv \
        --model retinanet_r50_fpn --severity-head detection_aware
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

from acne_utils import collate_fn  # noqa: E402
from multitask_configs import MODEL_REGISTRY, SEVERITY_HEADS  # noqa: E402
from multitask_dataset import MultiTaskAcneDataset, load_grade_map  # noqa: E402
from multitask_engine import evaluate, train_one_epoch  # noqa: E402
from multitask_models import build_model  # noqa: E402
from multitask_transforms import get_eval_transform, get_train_transform  # noqa: E402


def setup_logging(log_dir: Path, level: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_fn = log_dir / "log"
    level_map = {
        "critical": logging.CRITICAL, "error": logging.ERROR, "warning": logging.WARNING,
        "info": logging.INFO, "debug": logging.DEBUG,
    }
    logging.basicConfig(
        level=level_map.get(level, logging.DEBUG),
        format="%(asctime)s: %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_fn, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    return log_fn


def make_dataloaders(args, grade_map):
    root = Path(args.data)
    splits = {}
    for name, transform in (
        ("train", get_train_transform(args.image_size)),
        ("val", get_eval_transform(args.image_size)),
        ("test", get_eval_transform(args.image_size)),
    ):
        split_dir = root / name
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split: {split_dir}")
        ds = MultiTaskAcneDataset(split_dir, transform, grade_map=grade_map)
        splits[name] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    return splits["train"], splits["val"], splits["test"]


def find_last_checkpoint(model_store: Path, prefix: str):
    if not model_store.exists():
        return None, 0
    candidates = []
    for f in model_store.iterdir():
        m = re.match(rf"{re.escape(prefix)}_epoch(\d+)\.pt$", f.name)
        if m:
            candidates.append((int(m.group(1)), f))
    if not candidates:
        return None, 0
    candidates.sort()
    return candidates[-1][1], candidates[-1][0] + 1


def make_optimizer(args, params):
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-4)
    return torch.optim.Adam(params, lr=args.learning_rate)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-task training for ACNE detection + severity")
    p.add_argument("--data", "-d", default="./data/final/f_circle/ds", type=str)
    p.add_argument("--grade-csv", nargs="*", default=[
        "./data/final/grade1/label.csv",
        "./data/final/grade2/label.csv",
    ], help="One or more grade CSV files (id, classification).")
    p.add_argument("--model", default="fasterrcnn_r50_fpn", choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--severity-head", default="attention", choices=SEVERITY_HEADS)
    p.add_argument("--sev-loss-weight", default=1.0, type=float)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--image-size", default=800, type=int)
    p.add_argument("--epochs", "-e", default=50, type=int)
    p.add_argument("--batch-size", "-b", default=2, type=int)
    p.add_argument("--num-workers", "-w", default=4, type=int)
    p.add_argument("--optimizer", "-o", default="adamw", choices=["sgd", "adam", "adamw"])
    p.add_argument("--learning-rate", "-lr", default=1e-4, type=float)
    p.add_argument("--lr-step", default=10, type=int)
    p.add_argument("--lr-gamma", default=0.5, type=float)
    p.add_argument("--cuda", "-c", default=0, type=int)
    p.add_argument("--save-every", default=5, type=int)
    p.add_argument("--model-store", default="./models", type=str)
    p.add_argument("--print-freq", default=50, type=int)
    p.add_argument("--log-level", default="info")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()

    today = datetime.date.today().strftime("%Y%m%d")
    now = datetime.datetime.now().strftime("%H%M%S")
    log_dir = Path(f"./logs/{today}/{now}")
    log_fn = setup_logging(log_dir, args.log_level)
    logging.info(f"Log file: {log_fn}")
    for k, v in vars(args).items():
        logging.info(f"  {k:18s}: {v}")

    grade_map = load_grade_map(args.grade_csv) if args.grade_csv else {}
    logging.info(f"Loaded {len(grade_map)} severity labels from {len(args.grade_csv)} CSV(s)")

    train_dl, val_dl, test_dl = make_dataloaders(args, grade_map)
    logging.info(f"Splits: train={len(train_dl.dataset)} val={len(val_dl.dataset)} test={len(test_dl.dataset)}")

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    family = MODEL_REGISTRY[args.model]["family"]
    model = build_model(
        family=family,
        severity_head=args.severity_head,
        sev_loss_weight=args.sev_loss_weight,
        pretrained=args.pretrained,
    )

    model_store = Path(args.model_store)
    model_store.mkdir(parents=True, exist_ok=True)
    ckpt_prefix = f"{args.model}__{args.severity_head}"

    start_epoch = 0
    if args.resume:
        last_path, next_epoch = find_last_checkpoint(model_store, ckpt_prefix)
        if last_path is not None:
            logging.info(f"Resuming from {last_path}")
            ckpt = torch.load(last_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            start_epoch = next_epoch

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(args, params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    train_writer = SummaryWriter(log_dir=str(log_dir / "tb" / "train"))
    eval_writer = SummaryWriter(log_dir=str(log_dir / "tb" / "eval"))

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, optimizer, train_dl, device, epoch, args.print_freq, train_writer)
        lr_scheduler.step()
        evaluate(model, val_dl, device=device, writer=eval_writer, epoch=epoch)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = model_store / f"{ckpt_prefix}_epoch{epoch}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                },
                ckpt_path,
            )
            logging.info(f"Saved checkpoint: {ckpt_path}")

    logging.info("Final evaluation on the test split")
    evaluate(model, test_dl, device=device, writer=eval_writer, epoch=args.epochs)


if __name__ == "__main__":
    main()
