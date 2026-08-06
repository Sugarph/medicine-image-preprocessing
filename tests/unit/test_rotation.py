import json

import numpy as np
import pytest

from medicine_preprocess.geometry import TransformState, apply_explicit_rotation, map_points


def test_rotate_90_clockwise_preserves_pixels_and_maps_centers() -> None:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    outcome = apply_explicit_rotation(image, 90, TransformState(np.eye(3)))
    assert np.array_equal(outcome.image, np.ascontiguousarray(np.rot90(image, k=3)))
    assert np.allclose(map_points(((0, 0),), outcome.transform.forward), ((1, 0),))
    assert np.allclose(map_points(((2, 1),), outcome.transform.forward), ((0, 2),))


@pytest.mark.parametrize(
    ("degrees", "k"),
    [(0, 0), (180, 2), (270, 1)],
)
def test_right_angle_rotation_matches_exact_pixel_order(degrees: int, k: int) -> None:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    outcome = apply_explicit_rotation(image, degrees, TransformState(np.eye(3)))
    assert np.array_equal(outcome.image, np.ascontiguousarray(np.rot90(image, k=k)))
    assert outcome.image.dtype == np.uint8
    assert outcome.image.flags.c_contiguous


def test_rotation_left_composes_incoming_transform_and_has_inverse() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    incoming = np.array([[1, 0, 5], [0, 1, 7], [0, 0, 1]], dtype=np.float64)
    outcome = apply_explicit_rotation(image, 90, TransformState(incoming))
    expected_rotation = np.array([[0, -1, 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    assert np.allclose(outcome.transform.forward, expected_rotation @ incoming)
    assert np.allclose(outcome.transform.inverse @ outcome.transform.forward, np.eye(3), atol=1e-9)
    # (0, 0) translated to (5, 7), then rotated clockwise to (-6, 5).
    assert np.allclose(map_points(((-6, 5),), outcome.transform.inverse), ((0, 0),))


def test_rotation_does_not_mutate_source_or_incoming_transform() -> None:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    before = image.copy()
    incoming = np.eye(3, dtype=np.float64)
    outcome = apply_explicit_rotation(image, 180, TransformState(incoming))
    outcome.image[0, 0, 0] = 255
    outcome.transform.forward[0, 0] = 99
    assert np.array_equal(image, before)
    assert np.array_equal(incoming, np.eye(3))


def test_zero_rotation_is_skipped_and_returns_copy() -> None:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    outcome = apply_explicit_rotation(image, 0, TransformState(np.eye(3)))
    assert np.array_equal(outcome.image, image)
    assert outcome.image is not image
    assert outcome.record.to_dict() == {
        "name": "explicit_rotation",
        "status": "skipped",
        "reason": "zero_degrees",
        "duration_ms": 0.0,
        "details": {},
    }
    json.dumps(outcome.record.to_dict())


@pytest.mark.parametrize("degrees", [45, 90.0, True])
def test_invalid_rotation_degrees_are_rejected(degrees: object) -> None:
    with pytest.raises(ValueError, match="0, 90, 180, or 270"):
        apply_explicit_rotation(np.zeros((2, 3, 3), dtype=np.uint8), degrees, TransformState(np.eye(3)))
