import numpy as np
import pytest

from medicine_preprocess import preprocess_image
from medicine_preprocess.config import CropConfig, GeometryConfig, PreprocessConfig, QualityConfig


def test_configuration_is_immutable() -> None:
    config = PreprocessConfig.grabcut_experimental()
    with pytest.raises((AttributeError, TypeError)):
        config.preset_name = "changed"  # type: ignore[misc]


def test_preprocess_image_defaults_to_grabcut_experimental_when_config_omitted() -> None:
    image = np.full((32, 40, 3), 120, dtype=np.uint8)
    result = preprocess_image(image, source_id="no-config-arg")
    assert result.preset_name == "grabcut_experimental"


def test_right_angle_requires_an_allowed_integer() -> None:
    with pytest.raises(ValueError, match="right_angle_degrees"):
        GeometryConfig(right_angle_degrees=90.0)  # type: ignore[arg-type]


def test_minimum_padding_pixels_allows_zero_but_not_negative() -> None:
    assert CropConfig(minimum_padding_pixels=0).minimum_padding_pixels == 0
    with pytest.raises(ValueError, match="minimum_padding_pixels"):
        CropConfig(minimum_padding_pixels=-1)


def test_grabcut_foreground_is_an_accepted_crop_mode() -> None:
    assert CropConfig(mode="grabcut_foreground").mode == "grabcut_foreground"


def test_grabcut_area_ratio_bounds_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="grabcut_min_area_ratio"):
        CropConfig(grabcut_min_area_ratio=0.9, grabcut_max_area_ratio=0.1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("grabcut_work_max_dim", 0),
        ("grabcut_work_max_dim", -100),
        ("grabcut_max_iterations", 0),
        ("grabcut_soft_budget_ms", 0.0),
        ("grabcut_soft_budget_ms", -1.0),
        ("grabcut_time_budget_ms", 0.0),
        ("grabcut_time_budget_ms", -1.0),
    ],
)
def test_grabcut_numeric_fields_reject_non_positive_values(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        CropConfig(**{field: value})


def test_grabcut_soft_budget_must_not_exceed_hard_timeout() -> None:
    with pytest.raises(ValueError, match="grabcut_soft_budget_ms"):
        CropConfig(grabcut_soft_budget_ms=3000.0, grabcut_time_budget_ms=2500.0)


def test_grabcut_experimental_preset_selects_grabcut_foreground_mode() -> None:
    config = PreprocessConfig.grabcut_experimental()
    assert config.preset_name == "grabcut_experimental"
    assert config.preset_version == "1"
    assert config.crop.mode == "grabcut_foreground"
    assert config.geometry.deskew_enabled is True
    assert config.geometry.perspective_enabled is False


def test_grabcut_experimental_preset_caps_pre_crop_quality_analysis() -> None:
    config = PreprocessConfig.grabcut_experimental()
    assert config.quality.pre_crop_analysis_max_long_side == 1024
    assert config.resize.pre_enhancement_max_long_side == 2048


def test_pre_crop_analysis_max_long_side_defaults_to_full_resolution() -> None:
    assert QualityConfig().pre_crop_analysis_max_long_side is None


@pytest.mark.parametrize("value", [0, -100])
def test_pre_crop_analysis_max_long_side_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(ValueError, match="pre_crop_analysis_max_long_side"):
        QualityConfig(pre_crop_analysis_max_long_side=value)
