from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    dark_median_max: float
    bright_median_min: float
    low_contrast_max: float
    high_noise_min: float
    unusable_blur_max: float
    slightly_soft_max: float

    def __post_init__(self) -> None:
        for name in (
            "dark_median_max",
            "bright_median_min",
            "low_contrast_max",
            "high_noise_min",
            "unusable_blur_max",
            "slightly_soft_max",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TypeError(f"{name} must be a finite number")


@dataclass(frozen=True)
class QualityMeasurements:
    luminance_p01: float
    luminance_p05: float
    luminance_p10: float
    luminance_p50: float
    luminance_p90: float
    luminance_p95: float
    luminance_p99: float
    clipped_shadow_fraction: float
    clipped_highlight_fraction: float
    global_contrast_std: float
    median_local_contrast: float
    laplacian_variance: float
    noise_residual_std: float
    impulse_noise_fraction: float
    glare_fraction: float
    largest_glare_fraction: float
    channel_mean_spread: float
    width: int
    height: int
    total_pixels: int


@dataclass(frozen=True)
class QualityClassifications:
    exposure: Literal["normal", "dark", "bright"]
    contrast: Literal["normal", "low"]
    noise: Literal["normal", "high"]
    sharpness: Literal["sharp", "slightly_soft", "unusably_blurred"]


@dataclass(frozen=True)
class QualityReport:
    measurements: QualityMeasurements
    classifications: QualityClassifications | None


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a non-empty uint8 BGR array")
    if image.dtype != np.uint8:
        raise ValueError("image must be a non-empty uint8 BGR array")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be a non-empty uint8 BGR array")


def _median_tiled_contrast(luminance: np.ndarray) -> float:
    tiles = [
        tile
        for row in np.array_split(luminance, 8, axis=0)
        for tile in np.array_split(row, 8, axis=1)
        if tile.size
    ]
    return float(np.median([np.std(tile) for tile in tiles]))


def _impulse_noise_fraction(gray: np.ndarray) -> float:
    median = cv2.medianBlur(gray, 3)
    impulse = ((gray == 0) | (gray == 255)) & (
        np.abs(gray.astype(np.int16) - median.astype(np.int16)) >= 48
    )
    return float(np.mean(impulse))


def _glare_metrics(image: np.ndarray, luminance: np.ndarray | None = None) -> tuple[float, float]:
    # Reuse the caller's LAB luminance plane instead of recomputing it.
    lab_l = (
        cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        if luminance is None
        else luminance
    )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((lab_l >= 245) & (hsv[:, :, 1] <= 25)).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    minimum = max(1, int(np.ceil(image.shape[0] * image.shape[1] * 0.001)))
    areas = [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum
    ]
    total = image.shape[0] * image.shape[1]
    return (
        sum(areas) / total,
        (max(areas) / total) if areas else 0.0,
    )


def _classify(
    values: QualityMeasurements, thresholds: QualityThresholds
) -> QualityClassifications:
    exposure = "dark" if values.luminance_p50 <= thresholds.dark_median_max else (
        "bright" if values.luminance_p50 >= thresholds.bright_median_min else "normal"
    )
    contrast = "low" if values.median_local_contrast <= thresholds.low_contrast_max else "normal"
    noise = "high" if values.noise_residual_std >= thresholds.high_noise_min else "normal"
    sharpness = "unusably_blurred" if values.laplacian_variance <= thresholds.unusable_blur_max else (
        "slightly_soft"
        if values.laplacian_variance <= thresholds.slightly_soft_max
        else "sharp"
    )
    return QualityClassifications(exposure, contrast, noise, sharpness)


def analyze_image_quality(
    image: np.ndarray, thresholds: QualityThresholds | None = None
) -> QualityReport:
    _validate_image(image)
    lab_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    percentiles = np.percentile(lab_l, [1, 5, 10, 50, 90, 95, 99])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    residual = gray.astype(np.float32) - cv2.GaussianBlur(
        gray, (0, 0), 1.0
    ).astype(np.float32)
    measurements = QualityMeasurements(
        *map(float, percentiles),
        float(np.mean(lab_l <= 1)),
        float(np.mean(lab_l >= 254)),
        float(np.std(lab_l)),
        _median_tiled_contrast(lab_l),
        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        float(np.std(residual)),
        _impulse_noise_fraction(gray),
        *_glare_metrics(image, lab_l),
        float(np.ptp(image.reshape(-1, 3).mean(axis=0))),
        image.shape[1],
        image.shape[0],
        image.shape[0] * image.shape[1],
    )
    classifications = _classify(measurements, thresholds) if thresholds is not None else None
    return QualityReport(measurements, classifications)
