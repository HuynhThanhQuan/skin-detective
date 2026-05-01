"""Dataset for joint detection + severity grading.

Each sample yields ``(image, target)`` where ``target`` carries both:
  * detection fields (``boxes``, ``labels``, ``image_id``, ``area``, ``iscrowd``)
  * a scalar severity field (``severity``) in {0, 1, 2} or SEVERITY_IGNORE_INDEX.

Severity comes from a side CSV (``data/final/grade*/label.csv``) joined on the
labelbox image id, which equals the COCO ``file_name`` minus its extension.
If no grade is available for an image the severity loss is skipped for that
sample (CE ignore_index), so detection-only images can still train.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from multitask_configs import RAW_TO_SEVERITY, SEVERITY_IGNORE_INDEX


def load_grade_map(grade_csvs: Iterable[str | Path]) -> Dict[str, int]:
    """Build {labelbox_id: severity_class_idx} from one or more grade CSVs."""
    mapping: Dict[str, int] = {}
    for csv_path in grade_csvs:
        df = pd.read_csv(csv_path)
        if "id" not in df.columns or "classification" not in df.columns:
            raise ValueError(f"{csv_path} must have columns 'id' and 'classification'")
        for raw_id, raw_cls in zip(df["id"], df["classification"]):
            if pd.isna(raw_cls):
                continue
            cls = RAW_TO_SEVERITY.get(int(raw_cls))
            if cls is None:
                continue
            mapping[str(raw_id)] = cls
    return mapping


class MultiTaskAcneDataset(Dataset):
    """COCO-formatted detection dataset + per-image severity label."""

    def __init__(
        self,
        dataset_loc: str | Path,
        transforms=None,
        grade_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.dataset_loc = Path(dataset_loc)
        self.image_folder = self.dataset_loc / "image"
        with open(self.dataset_loc / "coco_instances.json", "r") as fh:
            coco = json.load(fh)
        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.transforms = transforms
        self.grade_map = grade_map or {}
        self._build_index()

    def _build_index(self) -> None:
        per_image: Dict[int, dict] = {img["id"]: {"classes": [], "boxes": []} for img in self.images}
        for ann in self.annotations:
            per_image[ann["image_id"]]["classes"].append(ann["category_id"])
            per_image[ann["image_id"]]["boxes"].append(ann["bbox"])
        # Stable ordering keyed by image position in the COCO file.
        self.index = [(i, img["id"]) for i, img in enumerate(self.images)]
        self.per_image = per_image

    def _severity_for(self, file_name: str) -> int:
        stem = os.path.splitext(file_name)[0]
        return self.grade_map.get(stem, SEVERITY_IGNORE_INDEX)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        img_pos, img_id = self.index[idx]
        meta = self.images[img_pos]
        anns = self.per_image[img_id]

        image = cv2.imread(str(self.image_folder / meta["file_name"]))
        if image is None:
            raise FileNotFoundError(f"Could not read {meta['file_name']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # COCO bbox = [x, y, w, h] -> Pascal VOC [x0, y0, x1, y1] for albumentations / torchvision.
        boxes = np.array(
            [[x, y, x + w, y + h] for x, y, w, h in anns["boxes"]],
            dtype=np.float32,
        )
        # Detection class ids in this dataset start at 0; torchvision reserves 0 for background.
        # Shift labels by +1 so the head outputs NUM_LESION_CLASSES + 1 logits.
        labels = np.array([c + 1 for c in anns["classes"]], dtype=np.int64)

        if self.transforms is not None:
            sample = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = sample["image"]
            boxes = np.asarray(sample["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.asarray(sample["labels"], dtype=np.int64)
        else:
            # Default: HWC uint8 -> CHW float in [0, 1].
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if boxes.shape[0] == 0:
            # torchvision detectors require non-empty boxes; create a single 1-pixel box on background.
            # The matched proposal will simply be ignored by the RPN loss for that sample.
            boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
            labels = np.array([0], dtype=np.int64)  # background

        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor([img_pos]),
            "area": torch.from_numpy((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            "severity": torch.tensor(self._severity_for(meta["file_name"]), dtype=torch.long),
        }
        return image, target
