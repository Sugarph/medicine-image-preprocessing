from __future__ import annotations

import cv2
import numpy as np
import pytest

from medicine_preprocess.config import GeometryConfig
from medicine_preprocess.geometry import (
    TransformState,
    estimate_small_skew,
    map_points,
    rotate_expand,
)
from medicine_preprocess.validation import validate_deskew


def _slanted_lines(angle: float) -> np.ndarray:
    canvas = np.full((300, 500, 3), 255, dtype=np.uint8)
    for y in range(60, 250, 40):
        cv2.line(canvas, (80, y), (420, y), (0, 0, 0), 3)
    matrix = cv2.getRotationMatrix2D((250, 150), angle, 1.0)
    return cv2.warpAffine(canvas, matrix, (500, 300), borderValue=(255, 255, 255))


def _hough_lines(*angles: float, length: int = 120) -> np.ndarray:
    lines = []
    origin_x, origin_y = 20, 100
    for index, angle in enumerate(angles):
        radians = np.deg2rad(angle)
        x1 = origin_x
        y1 = origin_y + index * 8
        x2 = x1 + int(round(length * np.cos(radians)))
        y2 = y1 + int(round(length * np.sin(radians)))
        lines.append([[x1, y1, x2, y2]])
    return np.asarray(lines, dtype=np.int32)


def test_agreeing_lines_produce_inverse_correction_angle() -> None:
    estimate = estimate_small_skew(_slanted_lines(5.0), GeometryConfig(deskew_enabled=True))

    assert estimate.accepted
    assert abs(estimate.correction_degrees + 5.0) <= 0.75


@pytest.mark.parametrize(
    ("angle", "accepted"),
    [(0.0, True), (7.0, True), (-7.0, True), (12.0, False)],
)
def test_deskew_acceptance_range(angle: float, accepted: bool) -> None:
    estimate = estimate_small_skew(_slanted_lines(angle), GeometryConfig(deskew_enabled=True))

    assert estimate.accepted is accepted
    if accepted and angle:
        assert abs(estimate.correction_degrees + angle) <= 0.75


def test_conflicting_lines_skip_deskew() -> None:
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.line(image, (50, 100), (450, 130), (0, 0, 0), 3)
    cv2.line(image, (50, 200), (450, 160), (0, 0, 0), 3)

    estimate = estimate_small_skew(image, GeometryConfig(deskew_enabled=True))

    assert not estimate.accepted


def test_no_lines_skip_deskew() -> None:
    image = np.full((300, 500, 3), 255, dtype=np.uint8)

    estimate = estimate_small_skew(image, GeometryConfig(deskew_enabled=True))

    assert not estimate.accepted
    assert estimate.reason == "no_lines"


def test_short_retained_support_skips_deskew(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(
        cv2,
        "HoughLinesP",
        lambda *args, **kwargs: _hough_lines(0, 0, 0, 0, 0, length=80),
    )

    estimate = estimate_small_skew(image, GeometryConfig(deskew_enabled=True))

    assert not estimate.accepted
    assert estimate.reason == "insufficient_support"


def test_weighted_mad_above_limit_skips_deskew(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(
        cv2,
        "HoughLinesP",
        lambda *args, **kwargs: _hough_lines(-4, -4, 0, 4, 4),
    )

    estimate = estimate_small_skew(image, GeometryConfig(deskew_enabled=True))

    assert not estimate.accepted
    assert estimate.reason == "angle_disagreement"
    assert estimate.weighted_mad_degrees > 1.5


def test_estimate_is_deterministic_and_does_not_mutate_input() -> None:
    image = _slanted_lines(5.0)
    before = image.copy()
    config = GeometryConfig(deskew_enabled=True)

    first = estimate_small_skew(image, config)
    second = estimate_small_skew(image, config)

    assert first == second
    assert np.array_equal(image, before)


def test_disabled_deskew_returns_skipped_estimate() -> None:
    estimate = estimate_small_skew(_slanted_lines(5.0), GeometryConfig())

    assert not estimate.accepted
    assert estimate.reason == "disabled"
    assert estimate.correction_degrees == 0.0


def test_rotate_expand_does_not_clip_source_corners() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    outcome = rotate_expand(image, 7.0, TransformState(np.eye(3)))

    assert outcome.image.shape[0] > 100
    assert outcome.image.shape[1] > 200
    assert outcome.image.dtype == np.uint8
    assert outcome.image.flags.c_contiguous
    corners = ((-0.5, -0.5), (199.5, -0.5), (-0.5, 99.5), (199.5, 99.5))
    mapped = map_points(corners, outcome.transform.forward)
    assert all(-0.5 - 1e-7 <= x <= outcome.image.shape[1] - 0.5 + 1e-7 for x, _ in mapped)
    assert all(-0.5 - 1e-7 <= y <= outcome.image.shape[0] - 0.5 + 1e-7 for _, y in mapped)


def test_rotate_expand_uses_reflected_border_without_black_wedges() -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    outcome = rotate_expand(image, 7.0, TransformState(np.eye(3)))

    assert np.all(outcome.image == 255)


def test_rotate_expand_composes_incoming_original_and_canonical_transforms() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    base = rotate_expand(image, -6.0, TransformState(np.eye(3)))
    incoming = TransformState(
        np.array([[1, 0, 5], [0, 1, 7], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 0, 11], [0, 1, 13], [0, 0, 1]], dtype=np.float64),
    )

    outcome = rotate_expand(image, -6.0, incoming)

    assert np.allclose(outcome.transform.forward, base.transform.forward @ incoming.forward)
    assert np.allclose(
        outcome.transform.canonical_forward,
        base.transform.forward @ incoming.canonical_forward,
    )
    assert np.allclose(
        outcome.transform.inverse @ outcome.transform.forward,
        np.eye(3),
        atol=1e-8,
    )
    assert np.allclose(
        outcome.transform.canonical_inverse @ outcome.transform.canonical_forward,
        np.eye(3),
        atol=1e-8,
    )


def test_rotate_expand_zero_angle_returns_copy_and_skips() -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    outcome = rotate_expand(image, 0.0, TransformState(np.eye(3)))

    assert outcome.record.status.value == "skipped"
    assert outcome.image is not image
    assert np.array_equal(outcome.image, image)


def test_rotate_expand_does_not_mutate_source_or_transform() -> None:
    image = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    before = image.copy()
    incoming = np.eye(3, dtype=np.float64)
    outcome = rotate_expand(image, 6.0, TransformState(incoming))

    outcome.image[0, 0, 0] = 255
    outcome.transform.forward[0, 0] = 99.0

    assert np.array_equal(image, before)
    assert np.array_equal(incoming, np.eye(3))


def test_rotate_expand_rejects_out_of_range_angle() -> None:
    with pytest.raises(ValueError, match="between -10 and 10"):
        rotate_expand(np.zeros((20, 20, 3), dtype=np.uint8), 10.1, TransformState(np.eye(3)))


def test_validate_deskew_accepts_expanded_result_and_round_trip() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    outcome = rotate_expand(image, 7.0, TransformState(np.eye(3)))

    verdict = validate_deskew(image, outcome)

    assert verdict.accepted
    assert verdict.details["round_trip_error"] < 1e-8


def test_validate_deskew_uses_operation_bounds_with_incoming_transform() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    incoming = TransformState(
        np.array([[1, 0, 50], [0, 1, 70], [0, 0, 1]], dtype=np.float64),
    )
    outcome = rotate_expand(image, 7.0, incoming)

    verdict = validate_deskew(image, outcome)

    assert verdict.accepted


def test_validate_deskew_accepts_zero_angle_with_incoming_translation() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    incoming = TransformState(
        np.array([[1, 0, 50], [0, 1, 70], [0, 0, 1]], dtype=np.float64),
    )
    outcome = rotate_expand(image, 0.0, incoming)

    verdict = validate_deskew(image, outcome)

    assert verdict.accepted


def test_validate_deskew_accepts_rotated_result_with_incoming_translation() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    incoming = TransformState(
        np.array([[1, 0, 50], [0, 1, 70], [0, 0, 1]], dtype=np.float64),
    )
    outcome = rotate_expand(image, 7.0, incoming)

    verdict = validate_deskew(image, outcome)

    assert verdict.accepted


def test_validate_deskew_explicit_accumulated_matrices_skip_unknown_clipping() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    incoming = TransformState(
        np.array([[1, 0, 50], [0, 1, 70], [0, 0, 1]], dtype=np.float64),
    )
    outcome = rotate_expand(image, 7.0, incoming)

    verdict = validate_deskew(
        image,
        outcome.image,
        outcome.transform.forward,
        outcome.transform.inverse,
    )

    assert verdict.accepted


def test_validate_deskew_rejects_explicitly_clipped_operation() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    outcome = rotate_expand(image, 7.0, TransformState(np.eye(3)))
    clipped = outcome.image[:-1, :-1].copy()

    verdict = validate_deskew(
        image,
        clipped,
        outcome.transform.forward,
        outcome.transform.inverse,
        operation_matrix=outcome.operation_matrix,
    )

    assert not verdict.accepted
    assert verdict.reason == "deskew_clipping"


def test_validate_deskew_rejects_clipped_candidate_and_preserves_checkpoint() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    before = image.copy()
    outcome = rotate_expand(image, 7.0, TransformState(np.eye(3)))
    clipped = outcome.image[:-1, :-1].copy()

    verdict = validate_deskew(
        image,
        clipped,
        outcome.transform.forward,
        outcome.transform.inverse,
    )

    assert not verdict.accepted
    assert verdict.reason == "deskew_clipping"
    assert np.array_equal(image, before)


def test_validate_deskew_rejects_invalid_transform() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    outcome = rotate_expand(image, 7.0, TransformState(np.eye(3)))
    invalid_inverse = np.zeros((3, 3), dtype=np.float64)

    verdict = validate_deskew(
        image,
        outcome.image,
        outcome.transform.forward,
        invalid_inverse,
    )

    assert not verdict.accepted
    assert verdict.reason == "invalid_transform"
