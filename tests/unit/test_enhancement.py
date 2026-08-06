from __future__ import annotations

import cv2
import numpy as np
import pytest

from medicine_preprocess.enhancement import apply_clahe_luminance, apply_gamma


def test_gamma_above_one_brightens_without_changing_dtype() -> None:
    image = np.full((16, 16, 3), 64, dtype=np.uint8)
    result = apply_gamma(image, 2.0)
    assert result.dtype == np.uint8
    assert float(result.mean()) > 64


def test_gamma_below_one_darkens() -> None:
    image = np.full((16, 16, 3), 144, dtype=np.uint8)
    result = apply_gamma(image, 0.5)
    assert float(result.mean()) < 144


def test_gamma_does_not_mutate_input_and_returns_contiguous_pixels() -> None:
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)[:, ::2]
    before = image.copy()
    result = apply_gamma(image, 1.4)
    assert np.array_equal(image, before)
    assert result.flags["C_CONTIGUOUS"]
    assert result.dtype == np.uint8


@pytest.mark.parametrize("gamma", [0.0, -1.0, float("nan"), float("inf")])
def test_gamma_rejects_nonpositive_or_nonfinite_values(gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma must be finite and > 0"):
        apply_gamma(np.zeros((2, 2, 3), dtype=np.uint8), gamma)


def test_clahe_changes_luminance_without_independent_rgb_equalization() -> None:
    ramp = np.tile(np.arange(80, 112, dtype=np.uint8), (32, 1))
    image = np.dstack((ramp, ramp // 2, ramp // 4))
    before_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    result = apply_clahe_luminance(image, clip_limit=2.0, grid_size=(8, 8))
    after_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    assert result.shape == image.shape
    assert not np.array_equal(result, image)
    assert np.max(np.abs(after_lab[:, :, 1:].astype(int) - before_lab[:, :, 1:].astype(int))) <= 2


def test_clahe_does_not_mutate_input_and_returns_contiguous_uint8() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(32, dtype=np.uint8)[:, None]
    before = image.copy()
    result = apply_clahe_luminance(image, clip_limit=2.0, grid_size=(8, 8))
    assert np.array_equal(image, before)
    assert result.flags["C_CONTIGUOUS"]
    assert result.dtype == np.uint8


def test_clahe_rejects_invalid_parameters() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="clip_limit"):
        apply_clahe_luminance(image, clip_limit=0.0, grid_size=(8, 8))
    with pytest.raises(ValueError, match="grid_size"):
        apply_clahe_luminance(image, clip_limit=2.0, grid_size=(0, 8))
