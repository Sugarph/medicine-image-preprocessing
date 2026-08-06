from __future__ import annotations

import numpy as np
import pytest

from medicine_preprocess.enhancement import apply_denoise, apply_unsharp_mask
from medicine_preprocess.validation import validate_denoise_change, validate_sharpen_change


def test_median_denoise_removes_impulse_without_mutating_source() -> None:
    source = np.full((31, 31, 3), 128, dtype=np.uint8)
    source[15, 15] = 255
    before = source.copy()
    result = apply_denoise(source, "median", 3.0)
    assert np.array_equal(source, before)
    assert result[15, 15, 0] == 128
    assert result.dtype == np.uint8
    assert result.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("mode", ["off", "median", "bilateral", "nlmeans"])
def test_all_denoise_modes_return_contiguous_copy(mode: str) -> None:
    source = np.full((16, 16, 3), 100, dtype=np.uint8)
    before = source.copy()
    result = apply_denoise(source, mode, 3.0)
    assert np.array_equal(source, before)
    assert result.dtype == np.uint8
    assert result.flags["C_CONTIGUOUS"]
    result[...] = 0
    assert np.array_equal(source, before)


def test_bilateral_and_nlmeans_reduce_deterministic_noise() -> None:
    rng = np.random.default_rng(7)
    source = np.clip(128 + rng.integers(-30, 31, (32, 32, 3)), 0, 255).astype(np.uint8)
    bilateral = apply_denoise(source, "bilateral", 3.0)
    nlmeans = apply_denoise(source, "nlmeans", 5.0)
    assert float(bilateral.std()) < float(source.std())
    assert float(nlmeans.std()) < float(source.std())


def test_denoise_rejects_invalid_mode_or_strength() -> None:
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="unsupported denoise mode"):
        apply_denoise(source, "unknown", 3.0)
    with pytest.raises(ValueError, match="strength"):
        apply_denoise(source, "median", 0.0)
    with pytest.raises(ValueError, match="strength"):
        apply_denoise(source, "median", float("nan"))


def test_unsharp_threshold_preserves_flat_regions() -> None:
    source = np.full((32, 32, 3), 100, dtype=np.uint8)
    result = apply_unsharp_mask(source, sigma=1.0, amount=0.35, threshold=3)
    assert np.array_equal(result, source)
    assert result.flags["C_CONTIGUOUS"]


def test_unsharp_changes_real_edge_but_respects_threshold() -> None:
    source = np.full((32, 32, 3), 30, dtype=np.uint8)
    source[:, 16:] = 220
    sharpened = apply_unsharp_mask(source, sigma=1.0, amount=1.0, threshold=1)
    skipped = apply_unsharp_mask(source, sigma=1.0, amount=1.0, threshold=255)
    assert not np.array_equal(sharpened, source)
    assert np.array_equal(skipped, source)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": 0.0, "amount": 0.35, "threshold": 3},
        {"sigma": 1.0, "amount": -0.1, "threshold": 3},
        {"sigma": 1.0, "amount": 1.1, "threshold": 3},
        {"sigma": 1.0, "amount": 0.35, "threshold": -1},
        {"sigma": 1.0, "amount": 0.35, "threshold": 256},
    ],
)
def test_unsharp_rejects_invalid_parameters(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        apply_unsharp_mask(np.zeros((8, 8, 3), dtype=np.uint8), **kwargs)


def test_denoise_validator_accepts_noise_reduction_with_retained_edge() -> None:
    rng = np.random.default_rng(11)
    before = np.full((64, 64, 3), 40, dtype=np.uint8)
    before[:, 32:] = 210
    noise = rng.integers(-12, 13, before.shape, dtype=np.int16)
    before = np.clip(before.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    after = apply_denoise(before, "median", 3.0)
    verdict = validate_denoise_change(before, after, maximum_edge_loss_fraction=0.30)
    assert verdict.accepted
    assert verdict.reason == "noise_reduced_with_edges_retained"
    assert verdict.details["after_noise"] < verdict.details["before_noise"]
    assert verdict.details["edge_retention"] >= 0.70


def test_denoise_validator_rejects_lost_edges() -> None:
    before = np.full((64, 64, 3), 40, dtype=np.uint8)
    before[:, 32:] = 210
    after = np.full_like(before, 125)
    verdict = validate_denoise_change(before, after, maximum_edge_loss_fraction=0.30)
    assert not verdict.accepted
    assert verdict.reason == "denoise_not_safe"
    assert verdict.details["edge_retention"] < 0.70


def test_denoise_validator_rejects_invalid_edge_loss_limit() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="maximum_edge_loss_fraction"):
        validate_denoise_change(image, image, maximum_edge_loss_fraction=1.1)


def test_sharpen_validator_rejects_clipped_halos() -> None:
    before = np.full((32, 32, 3), 120, dtype=np.uint8)
    after = before.copy()
    after[:, 15:17] = 255
    verdict = validate_sharpen_change(
        before,
        after,
        maximum_new_clipped_fraction=0.01,
        maximum_noise_increase_fraction=0.2,
    )
    assert not verdict.accepted
    assert verdict.reason == "sharpening_artifact_limit"
    assert verdict.details["new_clipping"] > 0.01


def test_sharpen_validator_rejects_noise_increase() -> None:
    rng = np.random.default_rng(3)
    before = np.full((64, 64, 3), 100, dtype=np.uint8)
    after = np.clip(
        before.astype(np.int16) + rng.integers(-40, 41, before.shape), 0, 255
    ).astype(np.uint8)
    verdict = validate_sharpen_change(
        before,
        after,
        maximum_new_clipped_fraction=0.50,
        maximum_noise_increase_fraction=0.20,
    )
    assert not verdict.accepted
    assert verdict.reason == "sharpening_artifact_limit"
    assert verdict.details["noise_ratio"] > 0.20
