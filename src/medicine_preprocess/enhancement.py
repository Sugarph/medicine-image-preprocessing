from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .quality import QualityReport, QualityThresholds
from .result import OperationRecord


@dataclass(frozen=True)
class EnhancementOutcome:
    """A validated enhancement result and its operation-level fallback record."""

    image: np.ndarray
    record: OperationRecord
    fallback_used: bool = False


# Minimum confirmed impulse-noise fraction to select the median filter.
_IMPULSE_NOISE_MIN = 0.01


@dataclass(frozen=True)
class AutomaticDecision:
    white_balance: bool
    gamma: bool
    percentile_exposure: bool
    clahe: bool
    denoise: bool
    sharpen: bool
    resize: bool
    denoise_mode: str = "off"  # "off" | "median" | "bilateral"


def choose_operations(
    report: QualityReport,
    thresholds: QualityThresholds,
    *,
    glare_warning: bool,
    color_cast_strong: bool,
    short_side: int,
    minimum_short_side: int,
) -> AutomaticDecision:
    classes = report.classifications
    if classes is None:
        raise ValueError("automatic decisions require classifications")
    clipped = (
        report.measurements.clipped_shadow_fraction > 0.20
        or report.measurements.clipped_highlight_fraction > 0.20
    )
    high_noise = classes.noise == "high"
    strong_dark = (
        classes.exposure == "dark"
        and report.measurements.luminance_p50 <= thresholds.dark_median_max / 2.0
    )

    # Fine text can raise noise_residual_std like real noise does; treat
    # "high noise" + "sharp" as unconfirmed unless impulse noise is confirmed.
    confirmed_impulse = report.measurements.impulse_noise_fraction >= _IMPULSE_NOISE_MIN
    text_like_residual = high_noise and classes.sharpness == "sharp" and not confirmed_impulse
    if confirmed_impulse:
        denoise_mode = "median"
    elif high_noise and not text_like_residual:
        denoise_mode = "bilateral"
    else:
        denoise_mode = "off"

    return AutomaticDecision(
        white_balance=color_cast_strong and not glare_warning,
        # Only ever brightens (dark case); never auto-darkens "bright".
        gamma=classes.exposure == "dark" and not clipped and not strong_dark,
        percentile_exposure=strong_dark and not clipped and not high_noise,
        clahe=classes.contrast == "low" and not glare_warning,
        denoise=denoise_mode != "off",
        sharpen=classes.sharpness == "slightly_soft" and not high_noise,
        resize=short_side < minimum_short_side,
        denoise_mode=denoise_mode,
    )


def _validate_bgr_uint8(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be non-empty")


def apply_gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    _validate_bgr_uint8(image)
    means = image.reshape(-1, 3).mean(axis=0, dtype=np.float64)
    target = float(np.mean(means))
    gains = np.divide(target, means, out=np.ones(3), where=means > 0)
    gains = np.clip(gains, 0.8, 1.25)
    return np.ascontiguousarray(
        np.clip(image.astype(np.float32) * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    )


def apply_percentile_exposure(image: np.ndarray) -> np.ndarray:
    _validate_bgr_uint8(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32)
    low, high = np.percentile(luminance, [5, 95])
    if high - low < 1.0:
        raise ValueError("luminance percentile span is too small")
    lab[:, :, 0] = np.clip(
        16.0 + (luminance - low) * (223.0 / (high - low)), 0, 255
    ).astype(np.uint8)
    return np.ascontiguousarray(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))


def automatic_gamma(median_luminance: float) -> float:
    # Floor of 1.0: only ever brightens, never darkens.
    ratio = max(float(median_luminance) / 255.0, 1.0 / 255.0)
    return float(np.clip(np.log(ratio) / np.log(0.5), 1.0, 1.55))


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    _validate_bgr_uint8(image)
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise ValueError("gamma must be finite and > 0")
    value = float(gamma)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("gamma must be finite and > 0")
    table = np.rint(
        255.0
        * np.power(np.arange(256, dtype=np.float64) / 255.0, 1.0 / value)
    ).astype(np.uint8)
    return np.ascontiguousarray(cv2.LUT(image, table))


def apply_clahe_luminance(
    image: np.ndarray,
    *,
    clip_limit: float,
    grid_size: tuple[int, int],
) -> np.ndarray:
    _validate_bgr_uint8(image)
    if isinstance(clip_limit, bool) or not isinstance(clip_limit, (int, float)):
        raise ValueError("clip_limit must be finite and > 0")
    clip = float(clip_limit)
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("clip_limit must be finite and > 0")
    if (
        not isinstance(grid_size, tuple)
        or len(grid_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid_size)
    ):
        raise ValueError("grid_size must contain two positive integers")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return np.ascontiguousarray(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))


def apply_denoise(image: np.ndarray, mode: str, strength: float) -> np.ndarray:
    _validate_bgr_uint8(image)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise ValueError("strength must be finite and > 0")
    value = float(strength)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("strength must be finite and > 0")
    if mode == "median":
        kernel = 3 if value <= 3 else 5
        return np.ascontiguousarray(cv2.medianBlur(image, kernel))
    if mode == "bilateral":
        return np.ascontiguousarray(
            cv2.bilateralFilter(
                image,
                d=5,
                sigmaColor=min(50.0, 8.0 * value),
                sigmaSpace=7.0,
            )
        )
    if mode == "nlmeans":
        h = min(7.0, max(1.0, value))
        return np.ascontiguousarray(
            cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
        )
    if mode == "off":
        return np.ascontiguousarray(image.copy())
    raise ValueError(f"unsupported denoise mode: {mode}")


def apply_unsharp_mask(
    image: np.ndarray, *, sigma: float, amount: float, threshold: int
) -> np.ndarray:
    _validate_bgr_uint8(image)
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise ValueError("sigma must be finite and > 0")
    sigma_value = float(sigma)
    if not math.isfinite(sigma_value) or sigma_value <= 0:
        raise ValueError("sigma must be finite and > 0")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise ValueError("amount must be finite and in [0, 1]")
    amount_value = float(amount)
    if not math.isfinite(amount_value) or not 0 <= amount_value <= 1:
        raise ValueError("amount must be finite and in [0, 1]")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 255:
        raise ValueError("threshold must be an integer in [0, 255]")
    blurred = cv2.GaussianBlur(
        image, (0, 0), sigmaX=sigma_value, sigmaY=sigma_value
    )
    difference = image.astype(np.int16) - blurred.astype(np.int16)
    mask = np.max(np.abs(difference), axis=2) >= threshold
    sharpened = np.clip(
        image.astype(np.float32) + amount_value * difference, 0, 255
    ).astype(np.uint8)
    output = image.copy()
    output[mask] = sharpened[mask]
    return np.ascontiguousarray(output)
