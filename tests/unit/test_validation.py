from __future__ import annotations

import numpy as np

from medicine_preprocess.validation import validate_clahe_change, validate_exposure_change


def test_exposure_validator_rejects_new_clipping() -> None:
    before = np.full((20, 20, 3), 40, dtype=np.uint8)
    after = np.full((20, 20, 3), 255, dtype=np.uint8)
    verdict = validate_exposure_change(
        before,
        after,
        target_luminance=128.0,
        maximum_new_clipped_fraction=0.02,
    )
    assert not verdict.accepted
    assert verdict.reason == "new_clipping_exceeds_limit"
    assert verdict.details["new_clipping"] > 0.02


def test_exposure_validator_accepts_reduced_median_error() -> None:
    before = np.full((20, 20, 3), 64, dtype=np.uint8)
    after = np.full((20, 20, 3), 128, dtype=np.uint8)
    verdict = validate_exposure_change(
        before,
        after,
        target_luminance=128.0,
        maximum_new_clipped_fraction=0.02,
    )
    assert verdict.accepted
    assert verdict.reason == "exposure_error_reduced"
    assert verdict.details["after_error"] < verdict.details["before_error"]


def test_exposure_validator_rejects_increased_median_error() -> None:
    before = np.full((20, 20, 3), 128, dtype=np.uint8)
    after = np.full((20, 20, 3), 64, dtype=np.uint8)
    verdict = validate_exposure_change(
        before,
        after,
        target_luminance=128.0,
        maximum_new_clipped_fraction=0.02,
    )
    assert not verdict.accepted
    assert verdict.reason == "exposure_not_improved"


def test_clahe_validator_accepts_local_contrast_gain() -> None:
    before = np.full((64, 64, 3), 100, dtype=np.uint8)
    before[:, ::2] = 90
    before[:, 1::2] = 110
    after = np.full((64, 64, 3), 100, dtype=np.uint8)
    after[:, ::2] = 70
    after[:, 1::2] = 130
    verdict = validate_clahe_change(before, after, maximum_new_clipped_fraction=0.02)
    assert verdict.accepted
    assert verdict.reason == "local_contrast_improved"
    assert verdict.details["after_local"] >= verdict.details["before_local"] * 1.05


def test_clahe_validator_rejects_local_contrast_loss() -> None:
    before = np.full((64, 64, 3), 100, dtype=np.uint8)
    before[:, ::2] = 70
    before[:, 1::2] = 130
    after = np.full((64, 64, 3), 100, dtype=np.uint8)
    after[:, ::2] = 98
    after[:, 1::2] = 102
    verdict = validate_clahe_change(before, after, maximum_new_clipped_fraction=0.02)
    assert not verdict.accepted
    assert verdict.reason == "local_contrast_not_improved"


def test_clahe_validator_rejects_flat_noop_even_when_baseline_is_zero() -> None:
    before = np.full((64, 64, 3), 100, dtype=np.uint8)
    after = before.copy()
    verdict = validate_clahe_change(before, after, maximum_new_clipped_fraction=0.02)
    assert not verdict.accepted
    assert verdict.reason == "local_contrast_not_improved"


def test_clahe_validator_accepts_zero_baseline_with_positive_local_contrast() -> None:
    before = np.full((64, 64, 3), 100, dtype=np.uint8)
    after = np.full((64, 64, 3), 100, dtype=np.uint8)
    after[:, ::2] = 80
    after[:, 1::2] = 120
    verdict = validate_clahe_change(before, after, maximum_new_clipped_fraction=0.02)
    assert verdict.accepted
    assert verdict.reason == "local_contrast_improved"
    assert verdict.details["before_local"] == 0.0
    assert verdict.details["after_local"] > 0.0


def test_clahe_validator_rejects_new_clipping_before_contrast_decision() -> None:
    before = np.full((64, 64, 3), 100, dtype=np.uint8)
    after = np.full((64, 64, 3), 255, dtype=np.uint8)
    verdict = validate_clahe_change(before, after, maximum_new_clipped_fraction=0.02)
    assert not verdict.accepted
    assert verdict.reason == "new_clipping_exceeds_limit"
