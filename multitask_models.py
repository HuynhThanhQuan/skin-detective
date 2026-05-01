"""Single-stage multi-task models: detection + severity grading.

Two detector families wrap torchvision detectors and add a severity head fed by
shared FPN features:

  * ``MultiTaskFasterRCNN``  — two-stage detector (legacy default)
  * ``MultiTaskRetinaNet``   — one-stage detector (faster, more recent)

Severity head variants live in :class:`SeverityHead`. Pick one with
``severity_head=...`` when building the model. Picking the right head is the
main lever for ablations:

  * ``global_pool``      — pool deepest FPN level + MLP. Simple baseline.
  * ``multiscale``       — pool every FPN level, concat, MLP. Sees fine + coarse lesions.
  * ``attention``        — learnable [SEV] query attends FPN tokens. DETR-style.
  * ``detection_aware``  — pool RoI features at GT (train) / predicted (eval) boxes
                            and aggregate. The severity prediction is conditioned
                            on the lesions the detector currently believes are there.

Forward contract:
  * training:   forward(images, targets) -> dict of losses (incl. ``loss_severity``)
  * inference:  forward(images)          -> list of detection dicts; each dict
                                            carries an extra ``severity_logits`` key.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN, RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
)
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_Weights,
    RetinaNetClassificationHead,
)
from torchvision.models import ResNet50_Weights
from torchvision.ops import roi_align

from multitask_configs import (
    NUM_DET_CLASSES,
    NUM_SEVERITY_CLASSES,
    SEVERITY_IGNORE_INDEX,
)


# ---------------------------------------------------------------------------
# Severity heads
# ---------------------------------------------------------------------------


class GlobalPoolSeverityHead(nn.Module):
    """Average-pool the deepest FPN level then MLP."""

    def __init__(self, fpn_channels: int = 256, num_classes: int = NUM_SEVERITY_CLASSES, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fpn_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, features: Dict[str, torch.Tensor], **_) -> torch.Tensor:
        # FPN dict keys are usually "0".."3" (+ "pool"). Use the deepest non-pool level.
        keys = [k for k in features.keys() if k != "pool"]
        x = features[keys[-1]]
        return self.mlp(x)


class MultiScaleSeverityHead(nn.Module):
    """Pool every FPN level, concat, MLP. Captures multi-scale lesion context."""

    def __init__(self, fpn_channels: int = 256, num_classes: int = NUM_SEVERITY_CLASSES, hidden: int = 512):
        super().__init__()
        self.fpn_channels = fpn_channels
        # Lazy: don't know how many FPN levels until first forward. Build MLP on first call.
        self.hidden = hidden
        self.num_classes = num_classes
        self.mlp: Optional[nn.Module] = None

    def _build_mlp(self, num_levels: int, device: torch.device) -> None:
        self.mlp = nn.Sequential(
            nn.Linear(self.fpn_channels * num_levels, self.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.hidden, self.num_classes),
        ).to(device)

    def forward(self, features: Dict[str, torch.Tensor], **_) -> torch.Tensor:
        levels = [features[k] for k in features.keys() if k != "pool"]
        if self.mlp is None:
            self._build_mlp(len(levels), levels[0].device)
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in levels]
        return self.mlp(torch.cat(pooled, dim=1))


class AttentionSeverityHead(nn.Module):
    """Learnable [SEV] query attends to flattened FPN tokens (DETR-style aggregation)."""

    def __init__(
        self,
        fpn_channels: int = 256,
        num_classes: int = NUM_SEVERITY_CLASSES,
        hidden: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.feat_proj = nn.Linear(fpn_channels, hidden)
        self.sev_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, features: Dict[str, torch.Tensor], **_) -> torch.Tensor:
        tokens = []
        for k, f in features.items():
            if k == "pool":
                continue
            B, C, H, W = f.shape
            tokens.append(f.flatten(2).transpose(1, 2))  # (B, H*W, C)
        tokens = torch.cat(tokens, dim=1)  # (B, N, C)
        tokens = self.feat_proj(tokens)
        q = self.sev_query.expand(tokens.size(0), 1, -1)
        out, _ = self.attn(q, tokens, tokens)
        out = self.norm(out.squeeze(1))
        return self.head(out)


class DetectionAwareSeverityHead(nn.Module):
    """RoI-pool at boxes (GT during training, predicted during eval), then attend.

    During training we pool features at GT boxes — the head learns to map
    lesion-conditional features to severity. During inference we pool at the
    detector's top-K predicted boxes; if there are no boxes we fall back to a
    learnable empty embedding (predicts "mild" by inductive bias of the loss).
    """

    def __init__(
        self,
        fpn_channels: int = 256,
        num_classes: int = NUM_SEVERITY_CLASSES,
        hidden: int = 256,
        num_heads: int = 4,
        roi_output_size: int = 7,
        max_boxes_per_image: int = 100,
        score_thresh: float = 0.3,
    ):
        super().__init__()
        self.roi_output_size = roi_output_size
        self.max_boxes = max_boxes_per_image
        self.score_thresh = score_thresh

        roi_dim = fpn_channels * roi_output_size * roi_output_size
        self.roi_proj = nn.Linear(roi_dim, hidden)
        self.empty_token = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.sev_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def _select_boxes(
        self,
        targets: Optional[List[dict]],
        detections: Optional[List[dict]],
        batch_size: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        boxes: List[torch.Tensor] = []
        if targets is not None:
            for t in targets:
                b = t["boxes"]
                if b.numel() == 0:
                    boxes.append(torch.zeros((0, 4), device=device))
                else:
                    boxes.append(b[: self.max_boxes].to(device))
        elif detections is not None:
            for d in detections:
                keep = d["scores"] >= self.score_thresh
                b = d["boxes"][keep][: self.max_boxes]
                boxes.append(b.to(device))
        else:
            boxes = [torch.zeros((0, 4), device=device) for _ in range(batch_size)]
        return boxes

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        image_sizes: Optional[List[tuple]] = None,
        targets: Optional[List[dict]] = None,
        detections: Optional[List[dict]] = None,
        **_,
    ) -> torch.Tensor:
        keys = [k for k in features.keys() if k != "pool"]
        # Use the highest-resolution FPN level for RoI alignment — good detail for small acne lesions.
        feat_map = features[keys[0]]
        B = feat_map.size(0)
        device = feat_map.device

        boxes_per_image = self._select_boxes(targets, detections, B, device)

        # Spatial scale = feat_map.H / image.H. With FPN level "0" stride is typically 4.
        if image_sizes is None:
            spatial_scale = feat_map.shape[-1] / max(feat_map.shape[-1], 1)
        else:
            spatial_scale = feat_map.shape[-1] / float(image_sizes[0][1])

        per_image_logits = []
        for i, boxes in enumerate(boxes_per_image):
            if boxes.numel() == 0:
                tokens = self.empty_token.unsqueeze(0)  # (1, 1, hidden)
            else:
                pooled = roi_align(
                    feat_map[i : i + 1],
                    [boxes],
                    output_size=(self.roi_output_size, self.roi_output_size),
                    spatial_scale=spatial_scale,
                    aligned=True,
                )  # (N, C, k, k)
                tokens = self.roi_proj(pooled.flatten(1)).unsqueeze(0)  # (1, N, hidden)
            q = self.sev_query  # (1, 1, hidden)
            out, _ = self.attn(q, tokens, tokens)
            per_image_logits.append(self.norm(out.squeeze(1)))

        x = torch.cat(per_image_logits, dim=0)
        return self.head(x)


def build_severity_head(name: str, fpn_channels: int = 256) -> nn.Module:
    if name == "global_pool":
        return GlobalPoolSeverityHead(fpn_channels=fpn_channels)
    if name == "multiscale":
        return MultiScaleSeverityHead(fpn_channels=fpn_channels)
    if name == "attention":
        return AttentionSeverityHead(fpn_channels=fpn_channels)
    if name == "detection_aware":
        return DetectionAwareSeverityHead(fpn_channels=fpn_channels)
    raise ValueError(f"Unknown severity head: {name}")


# ---------------------------------------------------------------------------
# Multi-task detector wrappers
# ---------------------------------------------------------------------------


def _severity_loss(logits: torch.Tensor, targets: List[dict]) -> torch.Tensor:
    sev = torch.stack([t["severity"] for t in targets]).to(logits.device)
    return F.cross_entropy(logits, sev, ignore_index=SEVERITY_IGNORE_INDEX)


def _attach_severity(detections: List[dict], logits: torch.Tensor) -> List[dict]:
    probs = F.softmax(logits, dim=-1)
    for det, lg, pr in zip(detections, logits, probs):
        det["severity_logits"] = lg
        det["severity_probs"] = pr
        det["severity_pred"] = int(pr.argmax().item())
    return detections


class MultiTaskFasterRCNN(FasterRCNN):
    """Faster R-CNN + severity head sharing the same backbone/FPN."""

    def __init__(
        self,
        num_det_classes: int = NUM_DET_CLASSES,
        severity_head: str = "global_pool",
        sev_loss_weight: float = 1.0,
        pretrained: bool = True,
        **rcnn_kwargs,
    ):
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        # Reuse torchvision's helper to build the backbone exactly the way fasterrcnn_resnet50_fpn does.
        base = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=backbone_weights)
        super().__init__(base.backbone, num_classes=num_det_classes, **rcnn_kwargs)

        if pretrained:
            # Copy the pretrained RPN + RoI weights, then swap the box predictor for our num_det_classes.
            self.rpn.load_state_dict(base.rpn.state_dict(), strict=False)
            self.roi_heads.load_state_dict(base.roi_heads.state_dict(), strict=False)
            in_features = self.roi_heads.box_predictor.cls_score.in_features
            self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_det_classes)

        fpn_channels = self.backbone.out_channels
        self.severity_head_name = severity_head
        self.severity_head = build_severity_head(severity_head, fpn_channels=fpn_channels)
        self.sev_loss_weight = sev_loss_weight

    def forward(self, images, targets=None):
        # Mirrors GeneralizedRCNN.forward but lets us tap FPN features for the severity head.
        if self.training and targets is None:
            raise ValueError("targets required in training mode")

        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        images_t, targets_t = self.transform(images, targets)
        features = self.backbone(images_t.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, proposal_losses = self.rpn(images_t, features, targets_t)
        detections, detector_losses = self.roi_heads(features, proposals, images_t.image_sizes, targets_t)
        detections = self.transform.postprocess(detections, images_t.image_sizes, original_image_sizes)

        sev_logits = self.severity_head(
            features,
            image_sizes=images_t.image_sizes,
            targets=targets_t if self.training else None,
            detections=None if self.training else detections,
        )

        if self.training:
            losses = {**proposal_losses, **detector_losses}
            losses["loss_severity"] = self.sev_loss_weight * _severity_loss(sev_logits, targets_t)
            return losses

        return _attach_severity(detections, sev_logits)


class MultiTaskRetinaNet(RetinaNet):
    """RetinaNet (one-stage) + severity head."""

    def __init__(
        self,
        num_det_classes: int = NUM_DET_CLASSES,
        severity_head: str = "global_pool",
        sev_loss_weight: float = 1.0,
        pretrained: bool = True,
        **rn_kwargs,
    ):
        weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        base = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights)
        super().__init__(base.backbone, num_classes=num_det_classes, **rn_kwargs)

        if pretrained:
            # Copy pretrained weights then replace the classification head for our class count.
            self.head.load_state_dict(base.head.state_dict(), strict=False)
            num_anchors = self.head.classification_head.num_anchors
            in_channels = base.head.classification_head.conv[0].in_channels
            self.head.classification_head = RetinaNetClassificationHead(
                in_channels, num_anchors, num_det_classes
            )

        fpn_channels = self.backbone.out_channels
        self.severity_head_name = severity_head
        self.severity_head = build_severity_head(severity_head, fpn_channels=fpn_channels)
        self.sev_loss_weight = sev_loss_weight

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("targets required in training mode")

        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        images_t, targets_t = self.transform(images, targets)
        features = self.backbone(images_t.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        feature_list = list(features.values())

        head_outputs = self.head(feature_list)
        anchors = self.anchor_generator(images_t, feature_list)

        sev_logits = self.severity_head(
            features,
            image_sizes=images_t.image_sizes,
            targets=targets_t if self.training else None,
        )

        if self.training:
            losses = self.compute_loss(targets_t, head_outputs, anchors)
            losses["loss_severity"] = self.sev_loss_weight * _severity_loss(sev_logits, targets_t)
            return losses

        # Eval: torchvision's RetinaNet.postprocess_detections expects head outputs
        # and anchors split per FPN level. Mirror that logic here.
        num_anchors_per_level = [x.size(2) * x.size(3) for x in feature_list]
        HW = sum(num_anchors_per_level)
        HWA = head_outputs["cls_logits"].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

        split_head_outputs = {
            k: list(v.split(num_anchors_per_level, dim=1)) for k, v in head_outputs.items()
        }
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        detections = self.postprocess_detections(split_head_outputs, split_anchors, images_t.image_sizes)
        detections = self.transform.postprocess(detections, images_t.image_sizes, original_image_sizes)

        if self.severity_head_name == "detection_aware":
            # Re-run with the actual detections so RoI features come from predicted boxes.
            sev_logits = self.severity_head(
                features,
                image_sizes=images_t.image_sizes,
                targets=None,
                detections=detections,
            )

        return _attach_severity(detections, sev_logits)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_model(
    family: str,
    severity_head: str = "global_pool",
    sev_loss_weight: float = 1.0,
    pretrained: bool = True,
) -> nn.Module:
    if family == "fasterrcnn":
        return MultiTaskFasterRCNN(
            severity_head=severity_head,
            sev_loss_weight=sev_loss_weight,
            pretrained=pretrained,
        )
    if family == "retinanet":
        return MultiTaskRetinaNet(
            severity_head=severity_head,
            sev_loss_weight=sev_loss_weight,
            pretrained=pretrained,
        )
    raise ValueError(f"Unknown model family: {family}")
