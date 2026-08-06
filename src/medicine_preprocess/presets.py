from __future__ import annotations

import json
from importlib import resources

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


def build_grabcut_experimental_v1() -> PreprocessConfig:
    """GrabCut foreground-crop preset, time-boxed with a no-crop fallback.
    Perspective correction is off by default; opt in via `replace`."""
    return PreprocessConfig(
        preset_name="grabcut_experimental",
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
