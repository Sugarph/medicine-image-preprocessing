from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageFile

from medicine_preprocess.config import InputConfig
from medicine_preprocess.image_io import PreprocessError, decode_and_canonicalize
from medicine_preprocess.result import hash_array_input, hash_image_pixels, sha256_bytes


def test_rgb_array_becomes_bgr_without_mutating_input() -> None:
    source = np.array([[[10, 20, 30]]], dtype=np.uint8)
    before = source.copy()
    result = decode_and_canonicalize(source, InputConfig(array_color_order="RGB"))
    assert result.image.tolist() == [[[30, 20, 10]]]
    assert np.array_equal(source, before)
    assert result.image is not source


def test_exif_orientation_6_rotates_pixels_and_maps_centers(tmp_path: Path) -> None:
    image = Image.fromarray(
        np.array(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            ],
            dtype=np.uint8,
        ),
        "RGB",
    )
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "oriented.png"
    image.save(path, exif=exif)
    result = decode_and_canonicalize(path, InputConfig())
    assert result.image.shape[:2] == (3, 2)
    assert result.image.tolist() == [
        [[3, 2, 1], [0, 0, 255]],
        [[6, 5, 4], [0, 255, 0]],
        [[9, 8, 7], [255, 0, 0]],
    ]
    mapped = result.original_to_canonical @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(mapped[:2], [1.0, 0.0])


def test_unsupported_extension_is_typed_rejection(tmp_path: Path) -> None:
    path = tmp_path / "image.heic"
    path.write_bytes(b"not-heic")
    with pytest.raises(PreprocessError, match="Unsupported image format"):
        decode_and_canonicalize(path, InputConfig())


@pytest.mark.parametrize(
    ("orientation", "canonical_size", "expected_transform"),
    [
        (1, (3, 2), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        (2, (3, 2), ((-1, 0, 2), (0, 1, 0), (0, 0, 1))),
        (3, (3, 2), ((-1, 0, 2), (0, -1, 1), (0, 0, 1))),
        (4, (3, 2), ((1, 0, 0), (0, -1, 1), (0, 0, 1))),
        (5, (2, 3), ((0, 1, 0), (1, 0, 0), (0, 0, 1))),
        (6, (2, 3), ((0, -1, 1), (1, 0, 0), (0, 0, 1))),
        (7, (2, 3), ((0, -1, 1), (-1, 0, 2), (0, 0, 1))),
        (8, (2, 3), ((0, 1, 0), (-1, 0, 2), (0, 0, 1))),
    ],
)
def test_all_exif_orientations_have_expected_size_and_transform(
    oriented_png,
    orientation: int,
    canonical_size: tuple[int, int],
    expected_transform: tuple[tuple[int, int, int], ...],
) -> None:
    result = decode_and_canonicalize(oriented_png(orientation), InputConfig())
    assert result.exif_orientation == orientation
    assert result.canonical_size == canonical_size
    assert np.array_equal(result.original_to_canonical, np.asarray(expected_transform, dtype=np.float64))


def test_gray_array_expands_to_three_channel_bgr() -> None:
    source = np.array([[0, 127], [255, 64]], dtype=np.uint8)
    result = decode_and_canonicalize(source, InputConfig(array_color_order="GRAY"))
    assert result.image.tolist() == [
        [[0, 0, 0], [127, 127, 127]],
        [[255, 255, 255], [64, 64, 64]],
    ]
    assert result.original_size == (2, 2)


def test_grayscale_file_decodes_to_three_channel_bgr(tmp_path: Path) -> None:
    source = np.array([[0, 127], [255, 64]], dtype=np.uint8)
    path = tmp_path / "gray.png"
    Image.fromarray(source, "L").save(path)
    result = decode_and_canonicalize(path, InputConfig())
    assert result.image.tolist() == [
        [[0, 0, 0], [127, 127, 127]],
        [[255, 255, 255], [64, 64, 64]],
    ]


def test_rgba_array_composites_against_bgr_background() -> None:
    source = np.array([[[200, 100, 50, 128]]], dtype=np.uint8)
    result = decode_and_canonicalize(
        source,
        InputConfig(array_color_order="RGBA", alpha_background_bgr=(100, 110, 120)),
    )
    assert result.image.tolist() == [[[75, 105, 160]]]


def test_bgra_array_composites_and_does_not_alias_input() -> None:
    source = np.array([[[50, 100, 200, 64]]], dtype=np.uint8)
    before = source.copy()
    result = decode_and_canonicalize(
        source,
        InputConfig(array_color_order="BGRA", alpha_background_bgr=(100, 110, 120)),
    )
    assert result.image.tolist() == [[[87, 107, 140]]]
    assert np.array_equal(source, before)


def test_uint16_array_scales_to_uint8_before_channel_reordering() -> None:
    source = np.array([[[65535, 32768, 0]]], dtype=np.uint16)
    result = decode_and_canonicalize(source, InputConfig(array_color_order="RGB"))
    assert result.image.tolist() == [[[0, 128, 255]]]
    assert result.image.dtype == np.uint8


@pytest.mark.parametrize(
    ("order", "source"),
    [
        ("RGB", np.zeros((2, 2, 2), dtype=np.uint8)),
        ("BGR", np.zeros((2, 2), dtype=np.uint8)),
        ("BGRA", np.zeros((2, 2, 3), dtype=np.uint8)),
        ("RGBA", np.zeros((2, 2, 3), dtype=np.uint8)),
        ("GRAY", np.zeros((2, 2, 3), dtype=np.uint8)),
    ],
)
def test_array_shape_must_match_declared_color_order(order: str, source: np.ndarray) -> None:
    with pytest.raises(PreprocessError, match="does not match declared color order"):
        decode_and_canonicalize(source, InputConfig(array_color_order=order))


def test_empty_array_is_rejected() -> None:
    with pytest.raises(PreprocessError, match="non-empty"):
        decode_and_canonicalize(np.empty((0, 2, 3), dtype=np.uint8), InputConfig())


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.uint32])
def test_unsupported_array_dtype_is_rejected(dtype: np.dtype) -> None:
    source = np.zeros((1, 1, 3), dtype=dtype)
    with pytest.raises(PreprocessError, match="Unsupported array dtype"):
        decode_and_canonicalize(source, InputConfig())


def test_excessive_dimensions_and_pixels_are_rejected() -> None:
    source = np.zeros((3, 2, 3), dtype=np.uint8)
    with pytest.raises(PreprocessError, match="maximum dimension"):
        decode_and_canonicalize(source, InputConfig(max_dimension=2))
    with pytest.raises(PreprocessError, match="maximum pixel"):
        decode_and_canonicalize(source, InputConfig(max_pixels=5))


def test_malformed_supported_file_is_typed_rejection(tmp_path: Path) -> None:
    path = tmp_path / "malformed.png"
    path.write_bytes(b"not-a-png")
    with pytest.raises(PreprocessError, match="Unable to decode image"):
        decode_and_canonicalize(path, InputConfig())


def test_path_hashes_and_sizes_are_reported(tmp_path: Path) -> None:
    source = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    path = tmp_path / "source.png"
    Image.fromarray(source, "RGB").save(path)
    result = decode_and_canonicalize(path, InputConfig())
    assert result.input_hash == sha256_bytes(path.read_bytes())
    assert result.canonical_image_hash == hash_image_pixels(result.image)
    assert result.original_size == (2, 1)
    assert result.canonical_size == (2, 1)
    assert result.exif_orientation == 1


def test_array_input_hash_uses_declared_order() -> None:
    source = np.array([[[10, 20, 30]]], dtype=np.uint8)
    result = decode_and_canonicalize(source, InputConfig(array_color_order="RGB"))
    assert result.input_hash == hash_array_input(source, "RGB")


def test_rgba_file_composites_against_configured_background(rgba_png: Path) -> None:
    config = InputConfig(alpha_background_bgr=(100, 110, 120))
    result = decode_and_canonicalize(rgba_png, config)
    assert result.image[0, 0].tolist() == [75, 105, 160]
    assert result.image[0, 1].tolist() == [100, 110, 120]


def test_cmyk_jpeg_decodes_to_bgr(cmyk_jpeg: Path) -> None:
    result = decode_and_canonicalize(cmyk_jpeg, InputConfig())
    with Image.open(cmyk_jpeg) as opened:
        expected_rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    expected_bgr = cv2.cvtColor(expected_rgb, cv2.COLOR_RGB2BGR)
    assert np.array_equal(result.image, expected_bgr)


def test_missing_supported_path_is_typed_rejection(tmp_path: Path) -> None:
    with pytest.raises(PreprocessError, match="Unable to decode image"):
        decode_and_canonicalize(tmp_path / "missing.jpg", InputConfig())


def test_unsupported_source_type_is_typed_rejection() -> None:
    with pytest.raises(PreprocessError, match="Unsupported image input type"):
        decode_and_canonicalize(42, InputConfig())  # type: ignore[arg-type]


def test_decompression_bomb_is_wrapped_at_decode_boundary(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "bomb.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(PreprocessError, match="Unable to decode image"):
        decode_and_canonicalize(path, InputConfig())


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (InputConfig(max_dimension=2), "maximum dimension"),
        (InputConfig(max_pixels=5), "maximum pixel"),
    ],
)
def test_path_limits_reject_from_header_before_loading_pixels(
    tmp_path: Path,
    monkeypatch,
    config: InputConfig,
    message: str,
) -> None:
    path = tmp_path / "too-large.png"
    Image.new("RGB", (3, 2), (1, 2, 3)).save(path)
    load_calls: list[object] = []

    def fail_if_loaded(self, *args, **kwargs):
        load_calls.append(self)
        raise AssertionError("pixel loading must not happen before configured limits")

    monkeypatch.setattr(ImageFile.ImageFile, "load", fail_if_loaded)
    with pytest.raises(PreprocessError, match=message):
        decode_and_canonicalize(path, config)
    assert load_calls == []
