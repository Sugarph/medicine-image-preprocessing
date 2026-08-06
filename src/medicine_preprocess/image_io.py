from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import InputConfig
from .result import hash_array_input, hash_image_pixels, sha256_bytes


ImageSource: TypeAlias = str | Path | np.ndarray
_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class PreprocessError(RuntimeError):
    """A source cannot be safely decoded or canonicalized."""


@dataclass(frozen=True)
class CanonicalImage:
    image: np.ndarray
    input_hash: str
    canonical_image_hash: str
    original_size: tuple[int, int]
    canonical_size: tuple[int, int]
    original_to_canonical: np.ndarray
    exif_orientation: int


def _exif_transform(orientation: int, width: int, height: int) -> np.ndarray:
    matrices = {
        1: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        2: ((-1, 0, width - 1), (0, 1, 0), (0, 0, 1)),
        3: ((-1, 0, width - 1), (0, -1, height - 1), (0, 0, 1)),
        4: ((1, 0, 0), (0, -1, height - 1), (0, 0, 1)),
        5: ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
        6: ((0, -1, height - 1), (1, 0, 0), (0, 0, 1)),
        7: ((0, -1, height - 1), (-1, 0, width - 1), (0, 0, 1)),
        8: ((0, 1, 0), (-1, 0, width - 1), (0, 0, 1)),
    }
    return np.asarray(matrices.get(orientation, matrices[1]), dtype=np.float64)


def decode_and_canonicalize(source: ImageSource, config: InputConfig) -> CanonicalImage:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise PreprocessError(f"Unsupported image format: {path.suffix}")
        try:
            raw = path.read_bytes()
            with Image.open(path) as opened:
                width, height = opened.size
                _validate_size(width, height, config)
                opened.load()
                orientation = int(opened.getexif().get(274, 1))
                oriented = ImageOps.exif_transpose(opened)
                rgba = oriented.convert("RGBA")
        except Image.DecompressionBombError as exc:
            raise PreprocessError(f"Unable to decode image: {path}") from exc
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise PreprocessError(f"Unable to decode image: {path}") from exc
        background_rgb = tuple(reversed(config.alpha_background_bgr))
        background = Image.new("RGBA", rgba.size, background_rgb + (255,))
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
        bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        input_hash = sha256_bytes(raw)
        transform = _exif_transform(orientation, width, height)
        original_size = (width, height)
    elif isinstance(source, np.ndarray):
        bgr, original_size = _canonicalize_array(source, config)
        orientation = 1
        transform = np.eye(3, dtype=np.float64)
        input_hash = hash_array_input(source, config.array_color_order)
    else:
        raise PreprocessError(f"Unsupported image input type: {type(source).__name__}")
    _validate_dimensions(bgr, config)
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    return CanonicalImage(
        image=bgr,
        input_hash=input_hash,
        canonical_image_hash=hash_image_pixels(bgr),
        original_size=original_size,
        canonical_size=(bgr.shape[1], bgr.shape[0]),
        original_to_canonical=transform,
        exif_orientation=orientation,
    )


def _to_uint8(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.uint8:
        return values.copy()
    if values.dtype == np.uint16:
        return np.rint(values.astype(np.float64) * (255.0 / 65535.0)).astype(np.uint8)
    raise PreprocessError(f"Unsupported array dtype: {values.dtype}")


def _canonicalize_array(source: np.ndarray, config: InputConfig) -> tuple[np.ndarray, tuple[int, int]]:
    if source.size == 0 or source.ndim not in (2, 3):
        raise PreprocessError("Image array must be non-empty with 2 or 3 dimensions")
    values = _to_uint8(np.asarray(source))
    order = config.array_color_order
    if order == "GRAY" and values.ndim == 2:
        bgr = cv2.cvtColor(values, cv2.COLOR_GRAY2BGR)
    elif order == "BGR" and values.ndim == 3 and values.shape[2] == 3:
        bgr = values.copy()
    elif order == "RGB" and values.ndim == 3 and values.shape[2] == 3:
        bgr = cv2.cvtColor(values, cv2.COLOR_RGB2BGR)
    elif order in {"BGRA", "RGBA"} and values.ndim == 3 and values.shape[2] == 4:
        rgba = values if order == "RGBA" else cv2.cvtColor(values, cv2.COLOR_BGRA2RGBA)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        background_rgb = np.asarray(tuple(reversed(config.alpha_background_bgr)), dtype=np.float32)
        rgb = np.rint(
            rgba[:, :, :3].astype(np.float32) * alpha
            + background_rgb * (1.0 - alpha)
        ).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        raise PreprocessError(f"Array shape {values.shape} does not match declared color order {order}")
    return bgr, (int(source.shape[1]), int(source.shape[0]))


def _validate_dimensions(image: np.ndarray, config: InputConfig) -> None:
    height, width = image.shape[:2]
    _validate_size(width, height, config)


def _validate_size(width: int, height: int, config: InputConfig) -> None:
    if width <= 0 or height <= 0:
        raise PreprocessError("Image dimensions must be positive")
    if width > config.max_dimension or height > config.max_dimension:
        raise PreprocessError("Image exceeds maximum dimension")
    if width * height > config.max_pixels:
        raise PreprocessError("Image exceeds maximum pixel count")
