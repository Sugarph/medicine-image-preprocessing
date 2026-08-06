from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ValidationVerdict:
    accepted: bool
    reason: str
    details: dict[str, float]


def validate_final_structure(
    image: np.ndarray,
    forward: np.ndarray,
    inverse: np.ndarray,
) -> ValidationVerdict:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
    ):
        return ValidationVerdict(False, "invalid_output_shape", {})
    if image.dtype != np.uint8:
        return ValidationVerdict(False, "invalid_output_dtype", {})
    if not image.flags.c_contiguous:
        return ValidationVerdict(False, "output_not_contiguous", {})
    if (
        not isinstance(forward, np.ndarray)
        or not isinstance(inverse, np.ndarray)
        or forward.shape != (3, 3)
        or inverse.shape != (3, 3)
        or not np.isfinite(forward).all()
        or not np.isfinite(inverse).all()
    ):
        return ValidationVerdict(False, "invalid_transform_shape_or_values", {})
    round_trip_error = float(np.max(np.abs(inverse @ forward - np.eye(3))))
    if round_trip_error > 1e-8:
        return ValidationVerdict(
            False,
            "transform_inverse_mismatch",
            {"round_trip_error": round_trip_error},
        )
    return ValidationVerdict(True, "valid_output", {"round_trip_error": round_trip_error})


def validate_deskew(
    before: np.ndarray,
    after_or_outcome,
    forward: np.ndarray | object | None = None,
    inverse: np.ndarray | None = None,
    *,
    operation_matrix: np.ndarray | None = None,
) -> ValidationVerdict:
    """Validate a deskew checkpoint without mutating the previous image."""
    outcome = after_or_outcome if hasattr(after_or_outcome, "image") else None
    after = outcome.image if outcome is not None else after_or_outcome
    if outcome is not None:
        state = getattr(outcome, "transform", None)
        if forward is None:
            forward = getattr(state, "forward", None)
        if inverse is None:
            inverse = getattr(state, "inverse", None)
        if operation_matrix is None:
            operation_matrix = getattr(outcome, "operation_matrix", None)
    if hasattr(forward, "forward"):
        state = forward
        forward = getattr(state, "forward", None)
        if inverse is None:
            inverse = getattr(state, "inverse", None)
    try:
        _validate_image(before, "before")
        _validate_image(after, "after")
    except ValueError as exc:
        return ValidationVerdict(False, "invalid_output", {"error": str(exc)})
    if not after.flags.c_contiguous:
        return ValidationVerdict(False, "output_not_contiguous", {})
    if (
        not isinstance(forward, np.ndarray)
        or not isinstance(inverse, np.ndarray)
    ):
        return ValidationVerdict(False, "invalid_transform", {})
    if (
        forward.shape != (3, 3)
        or inverse.shape != (3, 3)
        or not np.isfinite(forward).all()
        or not np.isfinite(inverse).all()
    ):
        return ValidationVerdict(False, "invalid_transform", {})
    if operation_matrix is not None and (
        not isinstance(operation_matrix, np.ndarray)
        or operation_matrix.shape != (3, 3)
        or not np.isfinite(operation_matrix).all()
        or abs(float(np.linalg.det(operation_matrix))) < 1e-12
    ):
        return ValidationVerdict(False, "invalid_transform", {})
    try:
        round_trip_error = float(np.max(np.abs(inverse @ forward - np.eye(3))))
        if (
            not math.isfinite(round_trip_error)
            or abs(float(np.linalg.det(forward))) < 1e-12
            or abs(float(np.linalg.det(inverse))) < 1e-12
            or round_trip_error > 1e-8
        ):
            return ValidationVerdict(
                False,
                "invalid_transform",
                {"round_trip_error": round_trip_error},
            )
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return ValidationVerdict(False, "invalid_transform", {"error": str(exc)})

    height, width = before.shape[:2]
    output_height, output_width = after.shape[:2]
    clipping_matrix = operation_matrix if operation_matrix is not None else forward
    corners = np.asarray(
        [
            [-0.5, -0.5, 1.0],
            [width - 0.5, -0.5, 1.0],
            [-0.5, height - 0.5, 1.0],
            [width - 0.5, height - 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    mapped = (np.asarray(clipping_matrix, dtype=np.float64) @ corners.T).T
    denominators = mapped[:, 2]
    if np.any(np.isclose(denominators, 0.0)) or not np.isfinite(mapped).all():
        return ValidationVerdict(False, "deskew_clipping", {})
    mapped = mapped[:, :2] / denominators[:, None]
    spans = np.max(mapped, axis=0) - np.min(mapped, axis=0)
    if (
        spans[0] > output_width + 1e-7
        or spans[1] > output_height + 1e-7
    ):
        return ValidationVerdict(False, "deskew_clipping", {})
    # Without an explicit operation matrix, only span is checked here.
    if operation_matrix is None:
        return ValidationVerdict(True, "deskew_within_bounds", {"round_trip_error": round_trip_error})
    if (
        np.any(mapped[:, 0] < -0.5 - 1e-7)
        or np.any(mapped[:, 0] > output_width - 0.5 + 1e-7)
        or np.any(mapped[:, 1] < -0.5 - 1e-7)
        or np.any(mapped[:, 1] > output_height - 0.5 + 1e-7)
    ):
        return ValidationVerdict(False, "deskew_clipping", {})
    return ValidationVerdict(True, "deskew_within_bounds", {"round_trip_error": round_trip_error})


def _validate_image(image: np.ndarray, name: str) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be a BGR image with shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")


def _validate_pair(before: np.ndarray, after: np.ndarray) -> None:
    _validate_image(before, "before")
    _validate_image(after, "after")
    if before.shape != after.shape:
        raise ValueError("before and after must have the same shape")


def _validate_clipping_limit(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("maximum_new_clipped_fraction must be finite and in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("maximum_new_clipped_fraction must be finite and in [0, 1]")
    return result


def _luminance(image: np.ndarray) -> np.ndarray:
    _validate_image(image, "image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]


def _new_clipping(before_l: np.ndarray, after_l: np.ndarray) -> float:
    before_fraction = float(np.mean((before_l <= 1) | (before_l >= 254)))
    after_fraction = float(np.mean((after_l <= 1) | (after_l >= 254)))
    return after_fraction - before_fraction


def validate_exposure_change(
    before: np.ndarray,
    after: np.ndarray,
    *,
    target_luminance: float,
    maximum_new_clipped_fraction: float,
) -> ValidationVerdict:
    _validate_pair(before, after)
    if isinstance(target_luminance, bool) or not isinstance(target_luminance, (int, float)):
        raise ValueError("target_luminance must be finite")
    target = float(target_luminance)
    if not math.isfinite(target):
        raise ValueError("target_luminance must be finite")
    clipping_limit = _validate_clipping_limit(maximum_new_clipped_fraction)

    before_l = _luminance(before)
    after_l = _luminance(after)
    before_error = abs(float(np.median(before_l)) - target)
    after_error = abs(float(np.median(after_l)) - target)
    new_clipping = _new_clipping(before_l, after_l)
    if new_clipping > clipping_limit:
        return ValidationVerdict(
            False,
            "new_clipping_exceeds_limit",
            {"new_clipping": new_clipping},
        )
    accepted = after_error < before_error
    return ValidationVerdict(
        accepted,
        "exposure_error_reduced" if accepted else "exposure_not_improved",
        {"before_error": before_error, "after_error": after_error},
    )


def validate_white_balance_change(
    before: np.ndarray,
    after: np.ndarray,
    *,
    maximum_new_clipped_fraction: float,
) -> ValidationVerdict:
    _validate_pair(before, after)
    clipping_limit = _validate_clipping_limit(maximum_new_clipped_fraction)
    before_l, after_l = _luminance(before), _luminance(after)
    new_clipping = _new_clipping(before_l, after_l)
    if new_clipping > clipping_limit:
        return ValidationVerdict(
            False,
            "new_clipping_exceeds_limit",
            {"new_clipping": new_clipping},
        )
    before_spread = float(np.ptp(before.reshape(-1, 3).mean(axis=0)))
    after_spread = float(np.ptp(after.reshape(-1, 3).mean(axis=0)))
    accepted = after_spread < before_spread
    return ValidationVerdict(
        accepted,
        "channel_spread_reduced" if accepted else "white_balance_not_improved",
        {"before_channel_spread": before_spread, "after_channel_spread": after_spread},
    )


def _median_tiled_std(luminance: np.ndarray) -> float:
    rows = np.array_split(luminance, 8, axis=0)
    tiles = [
        tile
        for row in rows
        for tile in np.array_split(row, 8, axis=1)
        if tile.size
    ]
    return float(np.median([float(np.std(tile)) for tile in tiles])) if tiles else 0.0


def validate_clahe_change(
    before: np.ndarray,
    after: np.ndarray,
    *,
    maximum_new_clipped_fraction: float,
) -> ValidationVerdict:
    _validate_pair(before, after)
    clipping_limit = _validate_clipping_limit(maximum_new_clipped_fraction)
    before_l, after_l = _luminance(before), _luminance(after)
    before_local, after_local = _median_tiled_std(before_l), _median_tiled_std(after_l)
    new_clipping = _new_clipping(before_l, after_l)
    if new_clipping > clipping_limit:
        return ValidationVerdict(
            False,
            "new_clipping_exceeds_limit",
            {"new_clipping": new_clipping},
        )
    required = before_local * 1.05
    accepted = after_local >= required and after_local > before_local
    return ValidationVerdict(
        accepted,
        "local_contrast_improved" if accepted else "local_contrast_not_improved",
        {"before_local": before_local, "after_local": after_local},
    )


def _noise_std(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return _noise_std_from_gray(gray)


def _noise_std_from_gray(gray: np.ndarray) -> float:
    gray = np.asarray(gray).astype(np.float32)
    return float(np.std(gray - cv2.GaussianBlur(gray, (0, 0), 1.0)))


def _validate_edge_loss_limit(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("maximum_edge_loss_fraction must be finite and in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("maximum_edge_loss_fraction must be finite and in [0, 1]")
    return result


def validate_denoise_change(
    before: np.ndarray,
    after: np.ndarray,
    *,
    maximum_edge_loss_fraction: float,
) -> ValidationVerdict:
    _validate_pair(before, after)
    edge_loss_limit = _validate_edge_loss_limit(maximum_edge_loss_fraction)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    before_noise = _noise_std_from_gray(before_gray)
    after_noise = _noise_std_from_gray(after_gray)
    before_edges = cv2.Canny(before_gray, 50, 150) > 0
    after_edges = cv2.Canny(after_gray, 50, 150) > 0
    retained = float(
        np.sum(
            before_edges
            & cv2.dilate(
                after_edges.astype(np.uint8), np.ones((3, 3), np.uint8)
            ).astype(bool)
        )
    ) / max(1, int(before_edges.sum()))
    accepted = after_noise < before_noise and retained >= 1.0 - edge_loss_limit
    return ValidationVerdict(
        accepted,
        "noise_reduced_with_edges_retained" if accepted else "denoise_not_safe",
        {
            "before_noise": before_noise,
            "after_noise": after_noise,
            "edge_retention": retained,
        },
    )


def _validate_noise_increase_limit(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("maximum_noise_increase_fraction must be finite and in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("maximum_noise_increase_fraction must be finite and in [0, 1]")
    return result


def validate_sharpen_change(
    before: np.ndarray,
    after: np.ndarray,
    *,
    maximum_new_clipped_fraction: float,
    maximum_noise_increase_fraction: float,
) -> ValidationVerdict:
    _validate_pair(before, after)
    clipping_limit = _validate_clipping_limit(maximum_new_clipped_fraction)
    noise_limit = _validate_noise_increase_limit(maximum_noise_increase_fraction)
    before_l, after_l = _luminance(before), _luminance(after)
    new_clipping = _new_clipping(before_l, after_l)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    before_noise = _noise_std_from_gray(before_gray)
    after_noise = _noise_std_from_gray(after_gray)
    noise_ratio = (after_noise - before_noise) / max(before_noise, 1e-6)
    edges = cv2.Canny(before_gray, 50, 150)
    halo_band = cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0
    overshoot = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2)
    halo_fraction = float(np.mean((overshoot >= 24) & halo_band))
    accepted = (
        new_clipping <= clipping_limit
        and noise_ratio <= noise_limit
        and halo_fraction <= 0.01
    )
    return ValidationVerdict(
        accepted,
        "sharpening_within_artifact_limits" if accepted else "sharpening_artifact_limit",
        {
            "new_clipping": new_clipping,
            "noise_ratio": noise_ratio,
            "halo_fraction": halo_fraction,
        },
    )
