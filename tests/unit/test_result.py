import json

import numpy as np
import pytest

from medicine_preprocess.config import PreprocessConfig
from medicine_preprocess.result import (
    CropMetadata,
    OperationRecord,
    OperationStatus,
    PreprocessResult,
    RuntimeEnvironment,
    canonical_config_json,
    hash_array_input,
)


def _minimal_result(quality_before: object, quality_after: object) -> PreprocessResult:
    config = PreprocessConfig(preset_name="baseline", preset_version="1")
    config_json = canonical_config_json(config)
    transform = np.eye(3, dtype=np.float32)
    return PreprocessResult(
        image=np.array([[[1, 2, 3]]], dtype=np.uint8),
        operations=(OperationRecord("canonicalize", OperationStatus.APPLIED, "decoded"),),
        original_to_final_transform=transform,
        final_to_original_transform=transform,
        crop=CropMetadata(),
        resize_scale_factor=1.0,
        resize_scale_x=1.0,
        resize_scale_y=1.0,
        warnings=(),
        fallback_used=False,
        quality_before=quality_before,
        quality_after=quality_after,
        pipeline_version="0.1.0",
        result_schema_version="1.0.0",
        preset_name=config.preset_name,
        preset_version=config.preset_version,
        config_json=config_json,
        config_hash="config-hash",
        input_hash="input-hash",
        canonical_image_hash="canonical-hash",
        output_image_hash="output-hash",
        runtime_environment=RuntimeEnvironment.current(),
        source_id="fixture",
    )


def test_config_json_and_hash_are_stable() -> None:
    config = PreprocessConfig(preset_name="baseline", preset_version="1")
    first = canonical_config_json(config)
    second = canonical_config_json(config)
    assert first == second
    assert first.startswith('{"debug":')


def test_array_hash_includes_declared_shape_dtype_and_color_order() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    assert hash_array_input(image, "BGR") == hash_array_input(image.copy(), "BGR")
    assert hash_array_input(image, "BGR") != hash_array_input(image, "RGB")


def test_operation_record_serializes_enum_value() -> None:
    record = OperationRecord("canonicalize", OperationStatus.APPLIED, "decoded")
    assert record.to_dict()["status"] == "applied"


def test_result_serializes_quality_metadata_without_raw_image() -> None:
    result = _minimal_result(
        {"mean_luma": 100.0, "classification": {"exposure": "dark"}},
        {"mean_luma": 120.0, "classification": {"exposure": "normal"}},
    )

    serialized = result.to_dict()

    assert serialized["quality_before"] == {
        "mean_luma": 100.0,
        "classification": {"exposure": "dark"},
    }
    assert serialized["quality_after"] == {
        "mean_luma": 120.0,
        "classification": {"exposure": "normal"},
    }
    assert "image" not in serialized
    json.dumps(serialized)


def test_operation_record_detaches_nested_details_from_caller() -> None:
    details = {"metrics": {"edge_loss": 0.2}, "labels": ["initial"]}
    record = OperationRecord("quality", OperationStatus.APPLIED, "measured", details=details)

    details["metrics"]["edge_loss"] = 0.9
    details["labels"].append("mutated")
    details["new"] = "caller-only"

    assert record.to_dict()["details"] == {
        "metrics": {"edge_loss": 0.2},
        "labels": ["initial"],
    }
    with pytest.raises(TypeError):
        record.details["new"] = "not allowed"  # type: ignore[index]


def test_operation_record_to_dict_uses_json_safe_plain_containers() -> None:
    record = OperationRecord(
        "quality",
        OperationStatus.APPLIED,
        "measured",
        details={"metrics": {"edge_loss": 0.2}, "labels": ["initial"]},
    )

    serialized = record.to_dict()

    assert serialized["details"] == {
        "metrics": {"edge_loss": 0.2},
        "labels": ["initial"],
    }
    assert isinstance(serialized["details"], dict)
    assert isinstance(serialized["details"]["labels"], list)
    json.dumps(serialized)


@pytest.mark.parametrize("raw_value", [b"raw", np.zeros((1,), dtype=np.uint8)])
def test_operation_record_rejects_non_json_detail_values(raw_value: object) -> None:
    with pytest.raises(TypeError, match="JSON-serializable"):
        OperationRecord(
            "quality",
            OperationStatus.APPLIED,
            "measured",
            details={"raw": raw_value},
        )
