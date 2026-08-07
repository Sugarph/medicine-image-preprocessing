from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal


ColorOrder = Literal["BGR", "RGB", "GRAY", "BGRA", "RGBA"]
CropMode = Literal["none", "grabcut_foreground", "yolo_label"]
GammaMode = Literal["off", "manual", "automatic"]
ContrastMode = Literal["off", "clahe", "automatic"]
DenoiseMode = Literal["off", "median", "bilateral", "nlmeans", "automatic"]
SharpenMode = Literal["off", "unsharp", "automatic"]
ResizeMode = Literal[
    "none",
    "fit_short_side",
    "fit_long_side",
    "fit_inside_box",
    "upscale_if_small",
]


def _finite(
    name: str,
    value: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _non_empty_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _boolean(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _choice(name: str, value: object, choices: set[str]) -> None:
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {expected}")


def _fraction(name: str, value: float) -> None:
    _finite(name, value, minimum=0.0, maximum=1.0)


def _positive_number(name: str, value: float) -> None:
    _finite(name, value, minimum=0.0)
    if float(value) <= 0:
        raise ValueError(f"{name} must be > 0")


def _tuple_of_positive_ints(name: str, value: object, length: int) -> None:
    if not isinstance(value, tuple) or len(value) != length:
        raise ValueError(f"{name} must be a tuple of {length} positive integers")
    for index, item in enumerate(value):
        _positive_int(f"{name}[{index}]", item)


def _bounded_int(name: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")


@dataclass(frozen=True)
class InputConfig:
    array_color_order: ColorOrder = "BGR"
    max_pixels: int = 80_000_000
    max_dimension: int = 20_000
    alpha_background_bgr: tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self) -> None:
        _choice("array_color_order", self.array_color_order, {"BGR", "RGB", "GRAY", "BGRA", "RGBA"})
        _positive_int("max_pixels", self.max_pixels)
        _positive_int("max_dimension", self.max_dimension)
        if (
            not isinstance(self.alpha_background_bgr, tuple)
            or len(self.alpha_background_bgr) != 3
        ):
            raise ValueError("alpha_background_bgr must contain exactly three integers")
        for index, channel in enumerate(self.alpha_background_bgr):
            _bounded_int(f"alpha_background_bgr[{index}]", channel, 0, 255)


@dataclass(frozen=True)
class QualityConfig:
    profile_name: str = "manual-v1"

    # Caps the quality_before preview resolution (observability only, never
    # feeds a decision). See pipeline.py.
    pre_crop_analysis_max_long_side: int | None = None

    def __post_init__(self) -> None:
        _non_empty_text("profile_name", self.profile_name)
        if self.pre_crop_analysis_max_long_side is not None:
            _positive_int(
                "pre_crop_analysis_max_long_side", self.pre_crop_analysis_max_long_side
            )


@dataclass(frozen=True)
class CropConfig:
    mode: CropMode = "none"
    padding_x_fraction: float = 0.08
    padding_y_fraction: float = 0.10
    minimum_padding_pixels: int = 8
    grabcut_min_area_ratio: float = 0.12
    grabcut_max_area_ratio: float = 0.92
    grabcut_work_max_dim: int = 512
    grabcut_max_iterations: int = 2
    grabcut_soft_budget_ms: float = 1500.0
    grabcut_time_budget_ms: float = 2500.0
    grabcut_quad_min_extent: float = 0.75
    yolo_weights_path: Path | None = None
    yolo_confidence_high: float = 0.5
    yolo_confidence_low: float = 0.25
    yolo_union_max_area_ratio: float = 0.90
    yolo_min_box_area_ratio: float = 0.001
    yolo_max_box_area_ratio: float = 0.98
    yolo_min_aspect_ratio: float = 0.02
    yolo_max_aspect_ratio: float = 50.0

    def __post_init__(self) -> None:
        _choice("mode", self.mode, {"none", "grabcut_foreground", "yolo_label"})
        _fraction("padding_x_fraction", self.padding_x_fraction)
        _fraction("padding_y_fraction", self.padding_y_fraction)
        if (
            isinstance(self.minimum_padding_pixels, bool)
            or not isinstance(self.minimum_padding_pixels, int)
            or self.minimum_padding_pixels < 0
        ):
            raise ValueError("minimum_padding_pixels must be a non-negative integer")

        _fraction("grabcut_min_area_ratio", self.grabcut_min_area_ratio)
        _fraction("grabcut_max_area_ratio", self.grabcut_max_area_ratio)
        if self.grabcut_min_area_ratio > self.grabcut_max_area_ratio:
            raise ValueError("grabcut_min_area_ratio must be <= grabcut_max_area_ratio")
        _positive_int("grabcut_work_max_dim", self.grabcut_work_max_dim)
        _positive_int("grabcut_max_iterations", self.grabcut_max_iterations)
        _positive_number("grabcut_soft_budget_ms", self.grabcut_soft_budget_ms)
        _positive_number("grabcut_time_budget_ms", self.grabcut_time_budget_ms)
        if self.grabcut_soft_budget_ms > self.grabcut_time_budget_ms:
            raise ValueError("grabcut_soft_budget_ms must be <= grabcut_time_budget_ms")
        _fraction("grabcut_quad_min_extent", self.grabcut_quad_min_extent)

        if self.yolo_weights_path is not None and not isinstance(self.yolo_weights_path, Path):
            raise TypeError("yolo_weights_path must be a pathlib.Path or None")
        if self.mode == "yolo_label" and self.yolo_weights_path is None:
            raise ValueError("yolo_weights_path is required when mode is 'yolo_label'")
        _fraction("yolo_confidence_high", self.yolo_confidence_high)
        _fraction("yolo_confidence_low", self.yolo_confidence_low)
        if self.yolo_confidence_low > self.yolo_confidence_high:
            raise ValueError("yolo_confidence_low must be <= yolo_confidence_high")
        _fraction("yolo_union_max_area_ratio", self.yolo_union_max_area_ratio)
        _fraction("yolo_min_box_area_ratio", self.yolo_min_box_area_ratio)
        _fraction("yolo_max_box_area_ratio", self.yolo_max_box_area_ratio)
        if self.yolo_min_box_area_ratio > self.yolo_max_box_area_ratio:
            raise ValueError("yolo_min_box_area_ratio must be <= yolo_max_box_area_ratio")
        _positive_number("yolo_min_aspect_ratio", self.yolo_min_aspect_ratio)
        _positive_number("yolo_max_aspect_ratio", self.yolo_max_aspect_ratio)
        if self.yolo_min_aspect_ratio > self.yolo_max_aspect_ratio:
            raise ValueError("yolo_min_aspect_ratio must be <= yolo_max_aspect_ratio")


@dataclass(frozen=True)
class GeometryConfig:
    right_angle_degrees: Literal[0, 90, 180, 270] = 0
    deskew_enabled: bool = False
    maximum_deskew_degrees: float = 10.0
    perspective_enabled: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.right_angle_degrees, bool)
            or not isinstance(self.right_angle_degrees, int)
            or self.right_angle_degrees not in {0, 90, 180, 270}
        ):
            raise ValueError("right_angle_degrees must be one of 0, 90, 180, or 270")
        _boolean("deskew_enabled", self.deskew_enabled)
        _positive_number("maximum_deskew_degrees", self.maximum_deskew_degrees)
        if self.maximum_deskew_degrees > 10:
            raise ValueError("maximum_deskew_degrees must be <= 10")
        _boolean("perspective_enabled", self.perspective_enabled)


@dataclass(frozen=True)
class EnhancementConfig:
    gamma_mode: GammaMode = "off"
    gamma: float = 1.0
    contrast_mode: ContrastMode = "off"
    clahe_clip_limit: float = 1.4
    clahe_grid_size: tuple[int, int] = (8, 8)
    denoise_mode: DenoiseMode = "off"
    denoise_strength: float = 3.0
    sharpen_mode: SharpenMode = "off"
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 0.35
    unsharp_threshold: int = 3
    white_balance_automatic: bool = False

    def __post_init__(self) -> None:
        _choice("gamma_mode", self.gamma_mode, {"off", "manual", "automatic"})
        _positive_number("gamma", self.gamma)
        _choice("contrast_mode", self.contrast_mode, {"off", "clahe", "automatic"})
        _positive_number("clahe_clip_limit", self.clahe_clip_limit)
        _tuple_of_positive_ints("clahe_grid_size", self.clahe_grid_size, 2)
        _choice("denoise_mode", self.denoise_mode, {"off", "median", "bilateral", "nlmeans", "automatic"})
        _positive_number("denoise_strength", self.denoise_strength)
        _choice("sharpen_mode", self.sharpen_mode, {"off", "unsharp", "automatic"})
        _positive_number("unsharp_sigma", self.unsharp_sigma)
        _fraction("unsharp_amount", self.unsharp_amount)
        _bounded_int("unsharp_threshold", self.unsharp_threshold, 0, 255)
        _boolean("white_balance_automatic", self.white_balance_automatic)


@dataclass(frozen=True)
class ResizeConfig:
    mode: ResizeMode = "none"
    minimum_short_side: int = 960
    maximum_long_side: int = 4096
    target_box: tuple[int, int] = (2048, 2048)
    maximum_upscale_factor: float = 2.0
    # Unconditional downscale cap applied right after geometry, before
    # quality measurement and enhancement. Separate from `mode` above.
    pre_enhancement_max_long_side: int | None = None

    def __post_init__(self) -> None:
        _choice(
            "mode",
            self.mode,
            {"none", "fit_short_side", "fit_long_side", "fit_inside_box", "upscale_if_small"},
        )
        _positive_int("minimum_short_side", self.minimum_short_side)
        _positive_int("maximum_long_side", self.maximum_long_side)
        if self.minimum_short_side > self.maximum_long_side:
            raise ValueError("minimum_short_side must be <= maximum_long_side")
        _tuple_of_positive_ints("target_box", self.target_box, 2)
        _finite("maximum_upscale_factor", self.maximum_upscale_factor, minimum=1.0, maximum=2.0)
        if self.pre_enhancement_max_long_side is not None:
            _positive_int("pre_enhancement_max_long_side", self.pre_enhancement_max_long_side)


@dataclass(frozen=True)
class ValidationConfig:
    maximum_new_clipped_fraction: float = 0.02
    maximum_edge_loss_fraction: float = 0.08
    maximum_noise_increase_fraction: float = 0.20

    def __post_init__(self) -> None:
        _fraction("maximum_new_clipped_fraction", self.maximum_new_clipped_fraction)
        _fraction("maximum_edge_loss_fraction", self.maximum_edge_loss_fraction)
        _fraction("maximum_noise_increase_fraction", self.maximum_noise_increase_fraction)


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool = False
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        _boolean("enabled", self.enabled)
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path or None")
        if self.enabled and self.output_dir is None:
            raise ValueError("debug output_dir is required when enabled")
        if not self.enabled and self.output_dir is not None:
            raise ValueError("debug output_dir requires enabled=True")


@dataclass(frozen=True)
class PreprocessConfig:
    preset_name: str
    preset_version: str
    input: InputConfig = field(default_factory=InputConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    resize: ResizeConfig = field(default_factory=ResizeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def __post_init__(self) -> None:
        _non_empty_text("preset_name", self.preset_name)
        _non_empty_text("preset_version", self.preset_version)
        expected_types = (
            ("input", InputConfig),
            ("quality", QualityConfig),
            ("crop", CropConfig),
            ("geometry", GeometryConfig),
            ("enhancement", EnhancementConfig),
            ("resize", ResizeConfig),
            ("validation", ValidationConfig),
            ("debug", DebugConfig),
        )
        for name, expected_type in expected_types:
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__}")

    @classmethod
    def grabcut(cls) -> "PreprocessConfig":
        from .presets import build_grabcut_v1

        return build_grabcut_v1()

    @classmethod
    def yolo_label_crop_experimental(cls, weights_path: str | Path | None = None) -> "PreprocessConfig":
        from .presets import build_yolo_label_crop_experimental_v1

        return build_yolo_label_crop_experimental_v1(weights_path)
