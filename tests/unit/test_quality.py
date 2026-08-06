from __future__ import annotations

from dataclasses import asdict
import json

import cv2
import numpy as np
import pytest

from medicine_preprocess.quality import (
    QualityThresholds,
    analyze_image_quality,
)
from medicine_preprocess.config import PreprocessConfig
from medicine_preprocess.result import (
    CropMetadata,
    OperationRecord,
    OperationStatus,
    PreprocessResult,
    RuntimeEnvironment,
    canonical_config_json,
)


def _gray_image(value: int, *, height: int = 64, width: int = 64) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


_THRESHOLD_FIELDS = (
    "dark_median_max",
    "bright_median_min",
    "low_contrast_max",
    "high_noise_min",
    "unusable_blur_max",
    "slightly_soft_max",
)


def _threshold_kwargs() -> dict[str, float]:
    return {
        "dark_median_max": 40.0,
        "bright_median_min": 210.0,
        "low_contrast_max": 12.0,
        "high_noise_min": 8.0,
        "unusable_blur_max": 5.0,
        "slightly_soft_max": 25.0,
    }


@pytest.mark.parametrize("field", _THRESHOLD_FIELDS)
@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), float("-inf")])
def test_quality_thresholds_reject_bool_and_nonfinite_values(field: str, invalid: object) -> None:
    kwargs = _threshold_kwargs()
    kwargs[field] = invalid  # type: ignore[assignment]

    with pytest.raises(TypeError, match=f"{field} must be a finite number"):
        QualityThresholds(**kwargs)


def test_quality_thresholds_accept_finite_integer_and_float_values() -> None:
    integer_values = {field: 1 for field in _THRESHOLD_FIELDS}
    float_values = {field: 1.5 for field in _THRESHOLD_FIELDS}

    integer_thresholds = QualityThresholds(**integer_values)
    float_thresholds = QualityThresholds(**float_values)

    assert integer_thresholds.dark_median_max == 1
    assert float_thresholds.slightly_soft_max == 1.5


def test_quality_analysis_reports_raw_measurements_without_modifying_image() -> None:
    image = np.tile(np.arange(0, 256, dtype=np.uint8), (256, 1))
    bgr = np.dstack((image, image, image))
    before = bgr.copy()

    report = analyze_image_quality(bgr)

    assert np.array_equal(bgr, before)
    assert report.classifications is None
    assert report.measurements.luminance_p01 <= report.measurements.luminance_p50
    assert report.measurements.luminance_p50 <= report.measurements.luminance_p99
    assert report.measurements.width == 256
    assert report.measurements.height == 256
    assert report.measurements.total_pixels == 256 * 256


def test_thresholds_produce_only_locked_classification_values() -> None:
    image = _gray_image(10)
    thresholds = QualityThresholds(
        dark_median_max=41,
        bright_median_min=215,
        low_contrast_max=12,
        high_noise_min=8,
        unusable_blur_max=5,
        slightly_soft_max=25,
    )

    report = analyze_image_quality(image, thresholds)

    assert report.classifications is not None
    assert report.classifications.exposure == "dark"
    assert report.classifications.contrast in {"normal", "low"}
    assert report.classifications.noise in {"normal", "high"}
    assert report.classifications.sharpness in {"sharp", "slightly_soft", "unusably_blurred"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(40, "dark"), (120, "normal"), (210, "bright")],
)
def test_exposure_classification_uses_inclusive_boundaries(value: int, expected: str) -> None:
    thresholds = QualityThresholds(
        dark_median_max=41,
        bright_median_min=215,
        low_contrast_max=12,
        high_noise_min=8,
        unusable_blur_max=5,
        slightly_soft_max=25,
    )

    report = analyze_image_quality(_gray_image(value), thresholds)

    assert report.classifications is not None
    assert report.classifications.exposure == expected


def test_other_classification_boundaries_are_inclusive() -> None:
    thresholds = QualityThresholds(
        dark_median_max=-1,
        bright_median_min=255,
        low_contrast_max=0,
        high_noise_min=0,
        unusable_blur_max=0,
        slightly_soft_max=0,
    )

    report = analyze_image_quality(_gray_image(100), thresholds)

    assert report.classifications is not None
    assert report.classifications.exposure == "normal"
    assert report.classifications.contrast == "low"
    assert report.classifications.noise == "high"
    assert report.classifications.sharpness == "unusably_blurred"


def test_local_contrast_can_be_low_when_global_contrast_is_high() -> None:
    image = np.full((64, 64, 3), 20, dtype=np.uint8)
    image[32:, :, :] = 220

    report = analyze_image_quality(image)

    assert report.measurements.global_contrast_std > 50
    assert report.measurements.median_local_contrast == 0.0


def test_small_images_keep_tile_measurements_finite() -> None:
    report = analyze_image_quality(_gray_image(80, height=1, width=2))

    assert report.measurements.median_local_contrast == 0.0
    assert report.measurements.total_pixels == 2
    assert np.isfinite(report.measurements.laplacian_variance)


def test_sharp_edge_has_more_laplacian_energy_than_blurred_edge() -> None:
    sharp = np.zeros((64, 64, 3), dtype=np.uint8)
    sharp[:, 32:] = 255
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=3.0)

    sharp_report = analyze_image_quality(sharp)
    blurred_report = analyze_image_quality(blurred)

    assert sharp_report.measurements.laplacian_variance > blurred_report.measurements.laplacian_variance


def test_gaussian_noise_has_higher_residual_than_flat_image() -> None:
    rng = np.random.default_rng(13)
    noisy = np.clip(128 + rng.normal(0, 18, (64, 64, 3)), 0, 255).astype(np.uint8)

    flat_report = analyze_image_quality(_gray_image(128))
    noisy_report = analyze_image_quality(noisy)

    assert noisy_report.measurements.noise_residual_std > flat_report.measurements.noise_residual_std


def test_salt_and_pepper_noise_is_measured_as_impulses() -> None:
    image = _gray_image(128)
    image[::8, ::8] = 0
    image[4::8, 4::8] = 255

    report = analyze_image_quality(image)

    assert report.measurements.impulse_noise_fraction > 0


def test_glare_metrics_ignore_components_below_connected_component_floor() -> None:
    tiny = _gray_image(20, height=100, width=100)
    tiny[10:12, 10:12] = 255
    large = _gray_image(20, height=100, width=100)
    large[10:20, 10:20] = 255

    tiny_report = analyze_image_quality(tiny)
    large_report = analyze_image_quality(large)

    assert tiny_report.measurements.glare_fraction == 0.0
    assert tiny_report.measurements.largest_glare_fraction == 0.0
    assert large_report.measurements.glare_fraction == pytest.approx(0.01)
    assert large_report.measurements.largest_glare_fraction == pytest.approx(0.01)


def test_color_cast_and_clipping_fractions_are_reported() -> None:
    cast = np.full((16, 16, 3), (20, 20, 80), dtype=np.uint8)
    clipped = np.zeros((16, 16, 3), dtype=np.uint8)
    clipped[8:] = 255

    cast_report = analyze_image_quality(cast)
    clipped_report = analyze_image_quality(clipped)

    assert cast_report.measurements.channel_mean_spread > 0
    assert clipped_report.measurements.clipped_shadow_fraction == pytest.approx(0.5)
    assert clipped_report.measurements.clipped_highlight_fraction == pytest.approx(0.5)


def test_quality_report_is_finite_and_json_safe() -> None:
    report = analyze_image_quality(_gray_image(100))

    serialized = asdict(report)
    json.dumps(serialized, allow_nan=False)
    for value in vars(report.measurements).values():
        if isinstance(value, float):
            assert np.isfinite(value)


def test_result_serializer_emits_quality_report_as_json_safe_metadata() -> None:
    report = analyze_image_quality(_gray_image(100))
    config = PreprocessConfig(preset_name="baseline", preset_version="1")
    config_json = canonical_config_json(config)
    transform = np.eye(3, dtype=np.float64)
    result = PreprocessResult(
        image=_gray_image(100, height=1, width=1),
        operations=(OperationRecord("quality", OperationStatus.APPLIED, "measured"),),
        original_to_final_transform=transform,
        final_to_original_transform=transform,
        crop=CropMetadata(),
        resize_scale_factor=1.0,
        resize_scale_x=1.0,
        resize_scale_y=1.0,
        warnings=(),
        fallback_used=False,
        quality_before=report,
        quality_after=None,
        pipeline_version="1.0.0",
        result_schema_version="1.0.0",
        preset_name=config.preset_name,
        preset_version=config.preset_version,
        config_json=config_json,
        config_hash="config-hash",
        input_hash="input-hash",
        canonical_image_hash="canonical-hash",
        output_image_hash="output-hash",
        runtime_environment=RuntimeEnvironment.current(),
        source_id="quality-fixture",
    )

    serialized = result.to_dict()

    assert serialized["quality_before"]["measurements"]["width"] == 64
    assert serialized["quality_before"]["classifications"] is None
    assert serialized["quality_after"] is None
    json.dumps(serialized, allow_nan=False)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((0, 4, 3), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.float32),
    ],
)
def test_quality_rejects_non_bgr_uint8_images(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="non-empty.*uint8 BGR"):
        analyze_image_quality(image)
