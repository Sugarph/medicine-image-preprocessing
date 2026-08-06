from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def oriented_png(tmp_path: Path):
    """Create a lossless RGB fixture with a caller-selected EXIF orientation."""

    def make(orientation: int) -> Path:
        pixels = np.array(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            ],
            dtype=np.uint8,
        )
        image = Image.fromarray(pixels, "RGB")
        exif = Image.Exif()
        exif[274] = orientation
        path = tmp_path / f"oriented-{orientation}.png"
        image.save(path, exif=exif)
        return path

    return make


@pytest.fixture
def rgba_png(tmp_path: Path) -> Path:
    pixels = np.array(
        [
            [[200, 100, 50, 128], [10, 20, 30, 0]],
            [[40, 50, 60, 255], [70, 80, 90, 64]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "rgba.png"
    Image.fromarray(pixels, "RGBA").save(path)
    return path


@pytest.fixture
def cmyk_jpeg(tmp_path: Path) -> Path:
    pixels = np.array(
        [
            [[0, 255, 255, 0], [255, 0, 255, 0]],
            [[255, 255, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "cmyk.jpg"
    Image.fromarray(pixels, "CMYK").save(path, quality=100, subsampling=0)
    return path
