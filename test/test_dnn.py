"""Smoke tests — keep these dependency-light so CI on CPU runners passes fast."""

import importlib

import pytest


def test_torch_imports():
    torch = importlib.import_module("torch")
    assert torch.__version__.startswith("2.")


def test_torchvision_detection_model_loads():
    """Pretrained Faster R-CNN should construct (no weights download in CI — use no-weight init)."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=5)
    assert model is not None


def test_acne_configs_constants():
    import acne_configs

    assert len(acne_configs.ACNE_ID) == 5
    assert set(acne_configs.ID_SHORTDESC.keys()) == set(range(5))


@pytest.mark.parametrize("module", ["acne_dataset", "acne_utils", "engine", "utils"])
def test_modules_importable(module):
    importlib.import_module(module)
