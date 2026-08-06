import json

import cv2
import numpy as np
import pytest

from medicine_preprocess.config import ResizeConfig
from medicine_preprocess.geometry import TransformState, map_points
from medicine_preprocess.resize import calculate_target_size, resize_preserving_aspect_ratio


def test_upscale_if_small_preserves_aspect_and_caps_factor() -> None:
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    config = ResizeConfig(mode="upscale_if_small", minimum_short_side=1000,
                          maximum_upscale_factor=2.0)
    outcome = resize_preserving_aspect_ratio(image, config, TransformState(np.eye(3)))
    assert outcome.image.shape[:2] == (800, 1600)
    assert outcome.resize_scale_factor == 2.0
    assert np.allclose(map_points(((0, 0),), outcome.transform.forward), ((0.5, 0.5),))


def test_fit_inside_box_downscales_with_area_interpolation(monkeypatch) -> None:
    image = np.zeros((1000, 2000, 3), dtype=np.uint8)
    seen = {}
    real_resize = cv2.resize

    def spy(source, size, interpolation):
        seen["interpolation"] = interpolation
        return real_resize(source, size, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", spy)
    config = ResizeConfig(mode="fit_inside_box", target_box=(500, 500))
    outcome = resize_preserving_aspect_ratio(image, config, TransformState(np.eye(3)))
    assert outcome.image.shape[:2] == (250, 500)
    assert seen["interpolation"] == cv2.INTER_AREA


@pytest.mark.parametrize(
    ("size", "config", "expected"),
    [
        ((800, 400), ResizeConfig(mode="none"), (800, 400)),
        ((800, 400), ResizeConfig(mode="fit_short_side", minimum_short_side=600), (1200, 600)),
        ((800, 400), ResizeConfig(mode="fit_long_side", minimum_short_side=600,
                                  maximum_long_side=600), (600, 300)),
        ((2000, 1000), ResizeConfig(mode="fit_inside_box", target_box=(500, 500)), (500, 250)),
        ((800, 400), ResizeConfig(mode="upscale_if_small", minimum_short_side=1000,
                                  maximum_upscale_factor=2), (1600, 800)),
    ],
)
def test_calculate_target_size_table(size, config, expected) -> None:
    assert calculate_target_size(*size, config) == expected


@pytest.mark.parametrize(
    ("size", "config", "expected"),
    [
        ((8000, 4000), ResizeConfig(mode="fit_short_side", minimum_short_side=960,
                                    maximum_long_side=4096), (4096, 2048)),
        ((500, 250), ResizeConfig(mode="upscale_if_small", minimum_short_side=1000,
                                  maximum_upscale_factor=1.5), (750, 375)),
        ((100, 50), ResizeConfig(mode="fit_inside_box", target_box=(500, 500),
                                 maximum_upscale_factor=2.0), (200, 100)),
        ((5000, 2500), ResizeConfig(mode="fit_long_side", maximum_long_side=3000), (3000, 1500)),
    ],
)
def test_calculate_target_size_honors_caps_and_downscale(size, config, expected) -> None:
    assert calculate_target_size(*size, config) == expected


def test_resize_is_noop_copy_with_skipped_record() -> None:
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    outcome = resize_preserving_aspect_ratio(image, ResizeConfig(mode="none"), TransformState(np.eye(3)))
    assert np.array_equal(outcome.image, image)
    assert outcome.image is not image
    assert outcome.image.flags.c_contiguous
    assert outcome.resize_scale_factor == outcome.resize_scale_x == outcome.resize_scale_y == 1.0
    assert outcome.record.to_dict() == {
        "name": "resize",
        "status": "skipped",
        "reason": "size_already_valid",
        "duration_ms": 0.0,
        "details": {},
    }
    json.dumps(outcome.record.to_dict())


def test_resize_uses_lanczos_for_upscale_and_reports_exact_details(monkeypatch) -> None:
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    seen = {}
    real_resize = cv2.resize

    def spy(source, size, interpolation):
        seen["interpolation"] = interpolation
        return real_resize(source, size, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", spy)
    config = ResizeConfig(mode="upscale_if_small", minimum_short_side=1000)
    outcome = resize_preserving_aspect_ratio(image, config, TransformState(np.eye(3)))
    assert seen["interpolation"] == cv2.INTER_LANCZOS4
    assert outcome.record.to_dict() == {
        "name": "resize",
        "status": "applied",
        "reason": "upscale_if_small",
        "duration_ms": 0.0,
        "details": {"from": [800, 400], "to": [1600, 800]},
    }


def test_resize_composes_incoming_transform_and_preserves_inverse() -> None:
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    incoming = np.array([[1, 0, 5], [0, 1, 7], [0, 0, 1]], dtype=np.float64)
    config = ResizeConfig(mode="upscale_if_small", minimum_short_side=1000)
    outcome = resize_preserving_aspect_ratio(image, config, TransformState(incoming))
    scale = np.array([[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]], dtype=np.float64)
    assert np.allclose(outcome.transform.forward, scale @ incoming)
    assert np.allclose(outcome.transform.inverse @ outcome.transform.forward, np.eye(3), atol=1e-9)
    assert np.allclose(map_points(((0, 0),), outcome.transform.inverse), ((-5.25, -7.25),))


def test_resize_does_not_mutate_input_and_keeps_bgr_uint8_contract() -> None:
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    before = image.copy()
    outcome = resize_preserving_aspect_ratio(
        image,
        ResizeConfig(mode="fit_inside_box", target_box=(2, 2)),
        TransformState(np.eye(3)),
    )
    assert np.array_equal(image, before)
    assert outcome.image.dtype == np.uint8
    assert outcome.image.ndim == 3 and outcome.image.shape[2] == 3
    assert outcome.image.flags.c_contiguous


def test_resize_scale_rounding_keeps_aspect_error_bounded() -> None:
    image = np.zeros((333, 777, 3), dtype=np.uint8)
    config = ResizeConfig(mode="fit_inside_box", target_box=(500, 500))
    outcome = resize_preserving_aspect_ratio(image, config, TransformState(np.eye(3)))
    assert abs(outcome.resize_scale_x - outcome.resize_scale_y) <= 1 / min(777, 333)


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 3, 3), dtype=np.uint8),
        np.empty((3, 0, 3), dtype=np.uint8),
        np.zeros((3, 3), dtype=np.uint8),
        np.zeros((3, 3, 4), dtype=np.uint8),
        np.zeros((3, 3, 3), dtype=np.float32),
    ],
)
def test_resize_rejects_degenerate_or_non_bgr_input(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="non-empty contiguous uint8 BGR"):
        resize_preserving_aspect_ratio(image, ResizeConfig(mode="none"), TransformState(np.eye(3)))


@pytest.mark.parametrize("size", [(0, 10), (10, 0), (-1, 10)])
def test_calculate_target_size_rejects_degenerate_dimensions(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        calculate_target_size(*size, ResizeConfig(mode="none"))
