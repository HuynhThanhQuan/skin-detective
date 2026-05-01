"""Config for the single-stage multi-task pipeline (detection + severity grade)."""

from __future__ import annotations

import acne_configs

# Detection: same 5 lesion classes as the original two-stage pipeline.
# Background is class 0 inside torchvision detectors -> NUM_DET_CLASSES = 5 + 1.
NUM_LESION_CLASSES = len(acne_configs.ACNE_ID)
NUM_DET_CLASSES = NUM_LESION_CLASSES + 1  # +1 for background

# Severity grade: 3 ordinal classes.
SEVERITY_CLASSES = ["mild", "moderate", "severe"]
NUM_SEVERITY_CLASSES = len(SEVERITY_CLASSES)
SEVERITY_IGNORE_INDEX = -100  # samples without a grade label are skipped in CE loss

# CSV labels in data/final/grade*/label.csv use 1/2/3 -> shift to 0/1/2.
RAW_TO_SEVERITY = {1: 0, 2: 1, 3: 2}

# Backbone / model registry. Each entry is consumed by models.builder.build_model.
MODEL_REGISTRY = {
    "fasterrcnn_r50_fpn": {
        "family": "fasterrcnn",
        "backbone": "resnet50_fpn",
    },
    "retinanet_r50_fpn": {
        "family": "retinanet",
        "backbone": "resnet50_fpn",
    },
}

# Severity head registry. Each name maps to a class in models.heads.
SEVERITY_HEADS = ["global_pool", "multiscale", "attention", "detection_aware"]
