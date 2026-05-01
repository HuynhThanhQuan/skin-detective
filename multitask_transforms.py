"""Transforms for the multi-task pipeline.

Standalone from ``acne_utils.get_train_transform`` because that pipeline does
``Resize(224)`` followed by ``RandomCrop(450)`` which is wider than the resized
image and crashes albumentations.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# ImageNet stats — matches what the torchvision detector backbones were pretrained on.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _bbox_params():
    return {"format": "pascal_voc", "label_fields": ["labels"], "min_visibility": 0.1}


def get_train_transform(image_size: int = 800):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size, border_mode=0, value=0
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4, border_mode=0
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )


def get_eval_transform(image_size: int = 800):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size, border_mode=0, value=0
            ),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )
