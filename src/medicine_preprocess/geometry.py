from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .result import OperationRecord, OperationStatus


def _validated_matrix(value: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError(f"{label} must be a finite invertible 3x3 matrix")
    return matrix.copy()


@dataclass(frozen=True)
class DeskewEstimate:
    accepted: bool
    correction_degrees: float
    measured_degrees: float
    reason: str
    line_count: int = 0
    support_length: float = 0.0
    weighted_mad_degrees: float = 0.0

    @property
    def angle_degrees(self) -> float:
        return self.measured_degrees

    @property
    def mad_degrees(self) -> float:
        return self.weighted_mad_degrees


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    midpoint = float(np.sum(ordered_weights)) / 2.0
    index = int(np.searchsorted(np.cumsum(ordered_weights), midpoint, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def _deskew_estimate(
    accepted: bool,
    reason: str,
    *,
    correction_degrees: float = 0.0,
    measured_degrees: float = 0.0,
    line_count: int = 0,
    support_length: float = 0.0,
    weighted_mad_degrees: float = 0.0,
) -> DeskewEstimate:
    return DeskewEstimate(
        accepted=bool(accepted),
        correction_degrees=float(correction_degrees),
        measured_degrees=float(measured_degrees),
        reason=reason,
        line_count=int(line_count),
        support_length=float(support_length),
        weighted_mad_degrees=float(weighted_mad_degrees),
    )


def estimate_small_skew(image: np.ndarray, config) -> DeskewEstimate:
    """Estimate a high-confidence horizontal skew using bounded Hough support."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with shape (height, width, 3)")
    if image.dtype != np.uint8 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be a non-empty uint8 BGR image")
    try:
        enabled = bool(config.deskew_enabled)
        maximum = float(config.maximum_deskew_degrees)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("config must provide deskew_enabled and maximum_deskew_degrees") from exc
    if not enabled:
        return _deskew_estimate(False, "disabled")
    if not math.isfinite(maximum) or maximum <= 0.0 or maximum > 10.0:
        raise ValueError("maximum_deskew_degrees must be finite and in (0, 10]")

    height, width = image.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        estimation_width = max(1, int(round(width * scale)))
        estimation_height = max(1, int(round(height * scale)))
        gray = cv2.resize(
            gray,
            (estimation_width, estimation_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        estimation_height, estimation_width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 180.0,
        threshold=40,
        minLineLength=max(1, int(round(estimation_width * 0.15))),
        maxLineGap=max(1, int(round(estimation_width * 0.02))),
    )
    if lines is None or len(lines) == 0:
        return _deskew_estimate(False, "no_lines")

    angles: list[float] = []
    lengths: list[float] = []
    for raw_line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = (float(value) for value in raw_line)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= 0.0:
            continue
        raw_angle = math.degrees(math.atan2(dy, dx))
        angle = -raw_angle
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        if abs(angle) > 10.0:
            continue
        angles.append(angle)
        lengths.append(length)

    if not angles:
        return _deskew_estimate(False, "no_lines")
    angle_values = np.asarray(angles, dtype=np.float64)
    length_values = np.asarray(lengths, dtype=np.float64)
    line_count = int(angle_values.size)
    support_length = float(np.sum(length_values))
    if line_count < 5:
        return _deskew_estimate(
            False,
            "insufficient_lines",
            line_count=line_count,
            support_length=support_length,
        )
    if support_length < float(estimation_width):
        return _deskew_estimate(
            False,
            "insufficient_support",
            line_count=line_count,
            support_length=support_length,
        )

    measured = _weighted_median(angle_values, length_values)
    deviations = np.abs(angle_values - measured)
    weighted_mad = _weighted_median(deviations, length_values)
    if weighted_mad > 1.5:
        return _deskew_estimate(
            False,
            "angle_disagreement",
            measured_degrees=measured,
            line_count=line_count,
            support_length=support_length,
            weighted_mad_degrees=weighted_mad,
        )
    angular_span = float(np.ptp(angle_values))
    inlier_support = float(np.sum(length_values[np.abs(deviations) <= 1.5]))
    if angular_span > 3.0 and inlier_support < 0.8 * support_length:
        return _deskew_estimate(
            False,
            "angle_disagreement",
            measured_degrees=measured,
            line_count=line_count,
            support_length=support_length,
            weighted_mad_degrees=weighted_mad,
        )
    correction = -measured
    if abs(correction) > maximum:
        return _deskew_estimate(
            False,
            "angle_out_of_range",
            correction_degrees=correction,
            measured_degrees=measured,
            line_count=line_count,
            support_length=support_length,
            weighted_mad_degrees=weighted_mad,
        )
    return _deskew_estimate(
        True,
        "high_confidence_horizontal_lines",
        correction_degrees=correction,
        measured_degrees=measured,
        line_count=line_count,
        support_length=support_length,
        weighted_mad_degrees=weighted_mad,
    )


@dataclass(frozen=True)
class TransformState:
    forward: np.ndarray
    canonical_forward: np.ndarray | None = None

    def __post_init__(self) -> None:
        matrix = _validated_matrix(self.forward, "transform")
        object.__setattr__(self, "forward", matrix)
        canonical = matrix if self.canonical_forward is None else _validated_matrix(
            self.canonical_forward, "canonical transform"
        )
        object.__setattr__(self, "canonical_forward", canonical.copy())

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.forward)

    @property
    def canonical_inverse(self) -> np.ndarray:
        return np.linalg.inv(self.canonical_forward)

    def then(self, current_to_next: np.ndarray) -> "TransformState":
        operation = _validated_matrix(current_to_next, "operation")
        return TransformState(operation @ self.forward, operation @ self.canonical_forward)


def map_points(points: tuple[tuple[float, float], ...], matrix: np.ndarray) -> tuple[tuple[float, float], ...]:
    if not points:
        return ()
    values = np.asarray([[x, y, 1.0] for x, y in points], dtype=np.float64).T
    mapped = _validated_matrix(matrix, "transform") @ values
    denominators = mapped[2, :]
    if np.any(np.isclose(denominators, 0.0)):
        raise ValueError("transform maps a point to infinity")
    mapped /= denominators[None, :]
    if not np.isfinite(mapped).all():
        raise ValueError("transform produced non-finite point coordinates")
    return tuple((float(mapped[0, i]), float(mapped[1, i])) for i in range(mapped.shape[1]))


@dataclass(frozen=True)
class GeometryOutcome:
    image: np.ndarray
    transform: TransformState
    record: OperationRecord
    fallback_used: bool
    operation_matrix: np.ndarray | None = None


def apply_explicit_rotation(image: np.ndarray, degrees: int, transform: TransformState) -> GeometryOutcome:
    matrices = {
        90: np.array([[0, -1, image.shape[0] - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
        180: np.array([[-1, 0, image.shape[1] - 1], [0, -1, image.shape[0] - 1], [0, 0, 1]], dtype=np.float64),
        270: np.array([[0, 1, 0], [-1, 0, image.shape[1] - 1], [0, 0, 1]], dtype=np.float64),
    }
    if isinstance(degrees, bool) or not isinstance(degrees, int) or degrees not in (0, 90, 180, 270):
        raise ValueError("explicit rotation must be 0, 90, 180, or 270")
    if degrees == 0:
        return GeometryOutcome(
            np.ascontiguousarray(image.copy()),
            transform,
            OperationRecord("explicit_rotation", OperationStatus.SKIPPED, "zero_degrees"),
            False,
        )
    rotated = np.rot90(image, k={90: 3, 180: 2, 270: 1}[degrees])
    return GeometryOutcome(
        np.ascontiguousarray(rotated),
        transform.then(matrices[degrees]),
        OperationRecord(
            "explicit_rotation",
            OperationStatus.APPLIED,
            f"clockwise_{degrees}",
            details={"degrees": degrees},
        ),
        False,
    )


def rotate_expand(image: np.ndarray, angle: float, transform: TransformState) -> GeometryOutcome:
    """Rotate a BGR image on an expanded reflected canvas and compose transforms."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with shape (height, width, 3)")
    if image.dtype != np.uint8 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be a non-empty uint8 BGR image")
    if isinstance(angle, bool):
        raise ValueError("deskew angle must be finite and between -10 and 10 degrees")
    try:
        degrees = float(angle)
    except (TypeError, ValueError) as exc:
        raise ValueError("deskew angle must be finite and between -10 and 10 degrees") from exc
    if not math.isfinite(degrees) or abs(degrees) > 10.0:
        raise ValueError("deskew angle must be finite and between -10 and 10 degrees")
    if not isinstance(transform, TransformState):
        raise TypeError("transform must be a TransformState")
    if degrees == 0.0:
        operation = np.eye(3, dtype=np.float64)
        return GeometryOutcome(
            np.ascontiguousarray(image.copy(), dtype=np.uint8),
            transform,
            OperationRecord("deskew", OperationStatus.SKIPPED, "zero_degrees"),
            False,
            operation,
        )

    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    rotation = cv2.getRotationMatrix2D(center, degrees, 1.0)
    boundaries = np.asarray(
        [
            [-0.5, -0.5, 1.0],
            [width - 0.5, -0.5, 1.0],
            [-0.5, height - 0.5, 1.0],
            [width - 0.5, height - 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    rotated_boundaries = (np.asarray(rotation, dtype=np.float64) @ boundaries.T).T
    minimum = np.min(rotated_boundaries, axis=0)
    maximum = np.max(rotated_boundaries, axis=0)
    translation = np.array(
        [-0.5 - minimum[0], -0.5 - minimum[1]],
        dtype=np.float64,
    )
    affine = np.asarray(rotation, dtype=np.float64).copy()
    affine[:, 2] += translation
    output_width = max(1, int(math.ceil(float(maximum[0] - minimum[0]))))
    output_height = max(1, int(math.ceil(float(maximum[1] - minimum[1]))))
    warped = cv2.warpAffine(
        np.ascontiguousarray(image),
        affine,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    operation = np.eye(3, dtype=np.float64)
    operation[:2, :] = affine
    return GeometryOutcome(
        np.ascontiguousarray(warped, dtype=np.uint8),
        transform.then(operation),
        OperationRecord(
            "deskew",
            OperationStatus.APPLIED,
            "expanded_canvas",
            details={
                "degrees": degrees,
                "output_width": output_width,
                "output_height": output_height,
            },
        ),
        False,
        operation,
    )


def _ordered_perspective_corners(corners) -> np.ndarray:
    values = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if values.shape != (4, 2) or not np.isfinite(values).all():
        raise ValueError("perspective requires four finite corners")
    center = values.mean(axis=0)
    angles = np.arctan2(values[:, 1] - center[1], values[:, 0] - center[0])
    ordered = values[np.argsort(angles, kind="mergesort")]
    ordered = np.roll(ordered, -int(np.argmin(np.sum(ordered, axis=1))), axis=0)
    if cv2.contourArea(ordered.astype(np.float32)) < 0:
        ordered = ordered[[0, 3, 2, 1]]
    if not cv2.isContourConvex(ordered.astype(np.float32)):
        raise ValueError("perspective corners must be convex")
    return ordered


def apply_perspective_correction(
    image: np.ndarray,
    corners,
    transform: TransformState,
) -> GeometryOutcome:
    """Rectify an accepted quadrilateral without square stretching."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with shape (height, width, 3)")
    if image.dtype != np.uint8 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be a non-empty uint8 BGR image")
    if not isinstance(transform, TransformState):
        raise TypeError("transform must be a TransformState")
    ordered = _ordered_perspective_corners(corners)
    lengths = [
        float(np.linalg.norm(ordered[(index + 1) % 4] - ordered[index]))
        for index in range(4)
    ]
    output_width = int(round(max(lengths[0], lengths[2])))
    output_height = int(round(max(lengths[1], lengths[3])))
    if output_width < 64 or output_height < 64:
        raise ValueError("perspective output dimensions must be at least 64 pixels")
    aspect = output_width / output_height
    if aspect < 0.2 or aspect > 5.0:
        raise ValueError("perspective output aspect is outside [0.2, 5.0]")
    destination = np.asarray(
        [
            [0.0, 0.0],
            [output_width - 1.0, 0.0],
            [output_width - 1.0, output_height - 1.0],
            [0.0, output_height - 1.0],
        ],
        dtype=np.float32,
    )
    operation = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
    if operation.shape != (3, 3) or not np.isfinite(operation).all():
        raise ValueError("perspective transform is invalid")
    try:
        inverse = np.linalg.inv(operation)
    except np.linalg.LinAlgError as exc:
        raise ValueError("perspective transform is invalid") from exc
    mapped = (operation @ np.c_[ordered, np.ones(4)].T).T
    mapped = mapped[:, :2] / mapped[:, 2, None]
    round_trip = (inverse @ np.c_[mapped, np.ones(4)].T).T
    round_trip = round_trip[:, :2] / round_trip[:, 2, None]
    if not np.isfinite(inverse).all() or float(np.max(np.abs(round_trip - ordered))) > 1e-5:
        raise ValueError("perspective corner round-trip exceeds tolerance")
    corrected = cv2.warpPerspective(
        np.ascontiguousarray(image),
        operation,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return GeometryOutcome(
        np.ascontiguousarray(corrected, dtype=np.uint8),
        transform.then(operation),
        OperationRecord(
            "perspective",
            OperationStatus.APPLIED,
            "quadrilateral_rectified",
            details={"output_width": output_width, "output_height": output_height},
        ),
        False,
        operation,
    )
