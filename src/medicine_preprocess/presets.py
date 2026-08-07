from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from .config import CropConfig, EnhancementConfig, GeometryConfig, PreprocessConfig, QualityConfig, ResizeConfig
from .quality import QualityThresholds


_THRESHOLD_FIELDS = (
    "dark_median_max",
    "bright_median_min",
    "low_contrast_max",
    "high_noise_min",
    "unusable_blur_max",
    "slightly_soft_max",
)


def load_quality_thresholds_v2() -> QualityThresholds:
    path = resources.files("medicine_preprocess.data").joinpath("quality_thresholds_v2.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != set(_THRESHOLD_FIELDS):
        raise ValueError("quality threshold JSON has unexpected fields")
    return QualityThresholds(**{field: payload[field] for field in _THRESHOLD_FIELDS})


def build_grabcut_v1() -> PreprocessConfig:
    """GrabCut foreground-crop preset, time-boxed with a no-crop fallback.
    Perspective correction is off by default; opt in via `replace`."""
    return PreprocessConfig(
        preset_name="grabcut",
        preset_version="1",
        quality=QualityConfig(profile_name="frozen-v2", pre_crop_analysis_max_long_side=1024),
        crop=CropConfig(mode="grabcut_foreground", grabcut_work_max_dim=448),
        geometry=GeometryConfig(deskew_enabled=True, perspective_enabled=False),
        enhancement=EnhancementConfig(
            gamma_mode="automatic",
            contrast_mode="automatic",
            denoise_mode="automatic",
            sharpen_mode="automatic",
            white_balance_automatic=True,
        ),
        resize=ResizeConfig(mode="upscale_if_small", pre_enhancement_max_long_side=2048),
    )


def default_yolo_weights_path() -> Path:
    """Path to the bundled trained weights, resolved the same way as
    quality_thresholds_v2.json so it works after a real pip install, not
    just from a repo checkout."""
    return Path(str(resources.files("medicine_preprocess.data").joinpath("best.pt")))


def build_yolo_label_crop_experimental_v1(weights_path: str | Path | None = None) -> PreprocessConfig:
    """YOLO label-localization crop preset, requires medicine_preprocess[yolo].
    Defaults to the bundled weights when weights_path is omitted. No-crop
    fallback on low-confidence/near-full-frame detections. Perspective
    correction is off by default; opt in via `replace`."""
    resolved_weights = Path(weights_path) if weights_path is not None else default_yolo_weights_path()
    return PreprocessConfig(
        preset_name="yolo_label_crop_experimental",
        preset_version="1",
        quality=QualityConfig(profile_name="frozen-v2", pre_crop_analysis_max_long_side=1024),
        crop=CropConfig(
            mode="yolo_label",
            padding_x_fraction=0.05,
            padding_y_fraction=0.06,
            yolo_weights_path=resolved_weights,
        ),
        geometry=GeometryConfig(deskew_enabled=True, perspective_enabled=False),
        enhancement=EnhancementConfig(
            gamma_mode="automatic",
            contrast_mode="automatic",
            denoise_mode="automatic",
            sharpen_mode="automatic",
            white_balance_automatic=True,
        ),
        resize=ResizeConfig(mode="upscale_if_small", pre_enhancement_max_long_side=2048),
    )
