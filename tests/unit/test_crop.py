from __future__ import annotations

import numpy as np

from medicine_preprocess.config import CropConfig
from medicine_preprocess.crop import apply_crop
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
