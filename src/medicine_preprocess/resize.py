from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import ResizeConfig
from .geometry import TransformState
from .result import OperationRecord, OperationStatus


@dataclass(frozen=True)
class ResizeOutcome:
    image: np.ndarray
    transform: TransformState
    resize_scale_factor: float
    resize_scale_x: float
    resize_scale_y: float
    record: OperationRecord


def _positive_dimension(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive")


def calculate_target_size(width: int, height: int, config: ResizeConfig) -> tuple[int, int]:
    _positive_dimension(width, "width")
    _positive_dimension(height, "height")
    if not isinstance(config, ResizeConfig):
        raise TypeError("config must be a ResizeConfig")
    if config.mode == "none":
        return (width, height)

    short_side = min(width, height)
    long_side = max(width, height)
    if config.mode == "fit_short_side":
        requested = config.minimum_short_side / short_side
        scale = max(1.0, requested)
    elif config.mode == "fit_long_side":
        requested = config.maximum_long_side / long_side
        scale = requested
    elif config.mode == "fit_inside_box":
        requested = min(config.target_box[0] / width, config.target_box[1] / height)
        scale = requested
    elif config.mode == "upscale_if_small":
        requested = 1.0 if short_side >= config.minimum_short_side else config.minimum_short_side / short_side
        scale = max(1.0, requested)
    else:
        raise ValueError(f"unsupported resize mode: {config.mode}")

    scale = min(scale, config.maximum_upscale_factor, config.maximum_long_side / long_side)
    target_width = max(1, int(np.floor(width * scale + 0.5)))
    target_height = max(1, int(np.floor(height * scale + 0.5)))
    if max(target_width, target_height) > config.maximum_long_side:
        cap = config.maximum_long_side / max(target_width, target_height)
        target_width = max(1, int(np.floor(target_width * cap + 0.5)))
        target_height = max(1, int(np.floor(target_height * cap + 0.5)))
    return target_width, target_height


def _validate_image(image: np.ndarray) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
        or image.dtype != np.uint8
    ):
        raise ValueError("image must be a non-empty contiguous uint8 BGR array")


def resize_preserving_aspect_ratio(
    image: np.ndarray,
    config: ResizeConfig,
    transform: TransformState,
) -> ResizeOutcome:
    _validate_image(image)
    if not isinstance(config, ResizeConfig):
        raise TypeError("config must be a ResizeConfig")
    if not isinstance(transform, TransformState):
        raise TypeError("transform must be a TransformState")

    height, width = image.shape[:2]
    target_width, target_height = calculate_target_size(width, height, config)
    if (target_width, target_height) == (width, height):
        return ResizeOutcome(
            np.ascontiguousarray(image.copy(), dtype=np.uint8),
            transform,
            1.0,
            1.0,
            1.0,
            OperationRecord("resize", OperationStatus.SKIPPED, "size_already_valid"),
        )

    scale_x = target_width / width
    scale_y = target_height / height
    interpolation = cv2.INTER_AREA if max(scale_x, scale_y) < 1.0 else cv2.INTER_LANCZOS4
    output = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    matrix = np.array(
        [[scale_x, 0, 0.5 * scale_x - 0.5],
         [0, scale_y, 0.5 * scale_y - 0.5],
         [0, 0, 1]],
        dtype=np.float64,
    )
    return ResizeOutcome(
        np.ascontiguousarray(output, dtype=np.uint8),
        transform.then(matrix),
        (scale_x + scale_y) / 2.0,
        scale_x,
        scale_y,
        OperationRecord(
            "resize",
            OperationStatus.APPLIED,
            config.mode,
            details={"from": [width, height], "to": [target_width, target_height]},
        ),
    )
