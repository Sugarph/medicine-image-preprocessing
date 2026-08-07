from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from medicine_preprocess.config import CropConfig
from medicine_preprocess.crop import _load_yolo_model, _passes_geometry_check, _select_yolo_crop_box, apply_crop
from medicine_preprocess.geometry import TransformState
from medicine_preprocess.result import OperationStatus


def test_none_mode_skips_without_fallback() -> None:
    image = np.full((8, 9, 3), 37, dtype=np.uint8)
    outcome = apply_crop(image, CropConfig(mode="none"), TransformState(np.eye(3)))
    assert outcome.record.status is OperationStatus.SKIPPED
    assert outcome.record.reason == "crop_disabled"
    assert outcome.fallback_used is False
    assert outcome.metadata == type(outcome.metadata)()
    assert np.array_equal(outcome.image, image)


def test_grabcut_foreground_skips_without_experimental_flag() -> None:
    image = np.full((8, 9, 3), 37, dtype=np.uint8)
    outcome = apply_crop(image, CropConfig(mode="grabcut_foreground"), TransformState(np.eye(3)))
    assert outcome.record.status is OperationStatus.SKIPPED
    assert outcome.record.reason == "experimental_crop_not_enabled_in_v1"
    assert outcome.fallback_used is True
    assert outcome.metadata.crop_box_working is None
    assert np.array_equal(outcome.image, image)


def test_yolo_label_skips_without_experimental_flag() -> None:
    image = np.full((8, 9, 3), 37, dtype=np.uint8)
    config = CropConfig(mode="yolo_label", yolo_weights_path=Path("best.pt"))
    outcome = apply_crop(image, config, TransformState(np.eye(3)))
    assert outcome.record.status is OperationStatus.SKIPPED
    assert outcome.record.reason == "experimental_crop_not_enabled_in_v1"
    assert outcome.fallback_used is True
    assert np.array_equal(outcome.image, image)


# -- _select_yolo_crop_box / _passes_geometry_check: pure functions, no
# ultralytics import, so these run without the optional 'yolo' extra.

def test_select_yolo_crop_box_accepts_single_high_confidence_box() -> None:
    config = CropConfig()
    box, reason = _select_yolo_crop_box([(10.0, 10.0, 90.0, 90.0)], [0.8], 100, 100, config)
    assert box == (10.0, 10.0, 90.0, 90.0)
    assert reason == "yolo_single_box_detected"


def test_select_yolo_crop_box_rejects_low_confidence_degenerate_box() -> None:
    config = CropConfig()
    # confidence in the 0.25-0.5 tier only survives a geometry check; a
    # near-zero-area box should fail it and be discarded entirely.
    box, reason = _select_yolo_crop_box([(10.0, 10.0, 10.5, 10.5)], [0.3], 100, 100, config)
    assert box is None
    assert reason == "no_confident_detection"


def test_select_yolo_crop_box_accepts_low_confidence_box_passing_geometry() -> None:
    config = CropConfig()
    box, reason = _select_yolo_crop_box([(10.0, 10.0, 90.0, 90.0)], [0.3], 100, 100, config)
    assert box == (10.0, 10.0, 90.0, 90.0)
    assert reason == "yolo_single_box_detected"


def test_select_yolo_crop_box_unions_multiple_accepted_boxes() -> None:
    config = CropConfig()
    boxes = [(10.0, 10.0, 40.0, 40.0), (60.0, 60.0, 80.0, 80.0)]
    box, reason = _select_yolo_crop_box(boxes, [0.9, 0.9], 100, 100, config)
    assert box == (10.0, 10.0, 80.0, 80.0)
    assert reason == "yolo_2_boxes_united"


def test_select_yolo_crop_box_falls_back_when_union_covers_near_full_frame() -> None:
    config = CropConfig()
    boxes = [(0.0, 0.0, 50.0, 100.0), (50.0, 0.0, 100.0, 100.0)]
    box, reason = _select_yolo_crop_box(boxes, [0.9, 0.9], 100, 100, config)
    assert box is None
    assert reason == "union_near_full_frame"


def test_select_yolo_crop_box_falls_back_with_no_detections() -> None:
    config = CropConfig()
    box, reason = _select_yolo_crop_box([], [], 100, 100, config)
    assert box is None
    assert reason == "no_confident_detection"


def test_passes_geometry_check_rejects_extreme_aspect_ratio() -> None:
    config = CropConfig()
    assert _passes_geometry_check((0.0, 0.0, 99.0, 1.0), 100, 100, config) is False


# -- end-to-end, with the model itself mocked out (no ultralytics needed).

def test_apply_yolo_label_crops_on_confident_detection() -> None:
    _load_yolo_model.cache_clear()
    fake_boxes = MagicMock()
    fake_boxes.__len__ = lambda self: 1
    fake_boxes.xyxy.cpu.return_value.numpy.return_value.tolist.return_value = [[20.0, 20.0, 60.0, 60.0]]
    fake_boxes.conf.cpu.return_value.numpy.return_value.tolist.return_value = [0.9]
    fake_result = MagicMock()
    fake_result.boxes = fake_boxes
    fake_model = MagicMock()
    fake_model.predict.return_value = [fake_result]

    import medicine_preprocess.crop as crop_module
    original_loader = crop_module._load_yolo_model
    crop_module._load_yolo_model = lambda weights_path: fake_model
    try:
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        config = CropConfig(mode="yolo_label", yolo_weights_path=Path("fake.pt"))
        outcome = apply_crop(image, config, TransformState(np.eye(3)), experimental=True)
    finally:
        crop_module._load_yolo_model = original_loader

    assert outcome.record.status is OperationStatus.APPLIED
    assert outcome.record.reason == "yolo_single_box_detected"
    assert outcome.record.details["method"] == "yolo_label"
    assert outcome.record.details["num_detections"] == 1
    assert outcome.fallback_used is False
    assert outcome.image.shape[0] < image.shape[0]
    assert outcome.image.shape[1] < image.shape[1]


def test_apply_yolo_label_falls_back_when_no_boxes_detected() -> None:
    _load_yolo_model.cache_clear()
    fake_result = MagicMock()
    fake_result.boxes = None
    fake_model = MagicMock()
    fake_model.predict.return_value = [fake_result]

    import medicine_preprocess.crop as crop_module
    original_loader = crop_module._load_yolo_model
    crop_module._load_yolo_model = lambda weights_path: fake_model
    try:
        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        config = CropConfig(mode="yolo_label", yolo_weights_path=Path("fake.pt"))
        outcome = apply_crop(image, config, TransformState(np.eye(3)), experimental=True)
    finally:
        crop_module._load_yolo_model = original_loader

    assert outcome.record.status is OperationStatus.SKIPPED
    assert outcome.record.reason == "no_confident_detection"
    assert outcome.fallback_used is True
    assert np.array_equal(outcome.image, image)


def test_load_yolo_model_raises_clear_error_without_ultralytics_installed() -> None:
    _load_yolo_model.cache_clear()
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("ultralytics is installed in this environment")
    with pytest.raises(ImportError, match=r"medicine_preprocess\[yolo\]"):
        _load_yolo_model("does-not-matter.pt")
