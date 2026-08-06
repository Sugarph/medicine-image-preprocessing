from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import PIL

from .config import PreprocessConfig

if TYPE_CHECKING:
    from .quality import QualityReport

PIPELINE_VERSION = "1.0.0"
STABLE_PIPELINE_VERSION = "1.0.0"
RESULT_SCHEMA_VERSION = "1.0.0"


class OperationStatus(str, Enum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    REVERTED = "reverted"
    FAILED_SAFELY = "failed_safely"


@dataclass(frozen=True)
class OperationRecord:
    name: str
    status: OperationStatus
    reason: str
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        object.__setattr__(self, "details", _freeze_json(self.details, path="details"))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status.value, "reason": self.reason,
                "duration_ms": float(self.duration_ms), "details": _thaw_json(self.details)}


@dataclass(frozen=True)
class RuntimeEnvironment:
    python: str
    numpy: str
    opencv: str
    pillow: str
    operating_system: str
    platform: str
    architecture: str

    @classmethod
    def current(cls) -> "RuntimeEnvironment":
        return cls(sys.version.split()[0], np.__version__, cv2.__version__, PIL.__version__,
                   platform.system(), platform.platform(), platform.machine())


@dataclass(frozen=True)
class CropMetadata:
    crop_box_working: tuple[int, int, int, int] | None = None
    crop_polygon_original: tuple[tuple[float, float], ...] | None = None
    crop_polygon_canonical: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class PreprocessResult:
    image: np.ndarray
    operations: tuple[OperationRecord, ...]
    original_to_final_transform: np.ndarray
    final_to_original_transform: np.ndarray
    crop: CropMetadata
    resize_scale_factor: float
    resize_scale_x: float
    resize_scale_y: float
    warnings: tuple[str, ...]
    fallback_used: bool
    quality_before: QualityReport | None
    quality_after: QualityReport | None
    pipeline_version: str
    result_schema_version: str
    preset_name: str
    preset_version: str
    config_json: str
    config_hash: str
    input_hash: str
    canonical_image_hash: str
    output_image_hash: str
    runtime_environment: RuntimeEnvironment
    source_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operations": [record.to_dict() for record in self.operations],
            "original_to_final_transform": self.original_to_final_transform.tolist(),
            "final_to_original_transform": self.final_to_original_transform.tolist(),
            "crop": asdict(self.crop),
            "resize_scale_factor": self.resize_scale_factor,
            "resize_scale_x": self.resize_scale_x,
            "resize_scale_y": self.resize_scale_y,
            "warnings": list(self.warnings),
            "fallback_used": self.fallback_used,
            "quality_before": _serialize_quality_metadata(self.quality_before, path="quality_before"),
            "quality_after": _serialize_quality_metadata(self.quality_after, path="quality_after"),
            "pipeline_version": self.pipeline_version,
            "result_schema_version": self.result_schema_version,
            "preset_name": self.preset_name,
            "preset_version": self.preset_version,
            "config_json": self.config_json,
            "config_hash": self.config_hash,
            "input_hash": self.input_hash,
            "canonical_image_hash": self.canonical_image_hash,
            "output_image_hash": self.output_image_hash,
            "runtime_environment": asdict(self.runtime_environment),
            "source_id": self.source_id,
        }


def _freeze_json(value: Any, *, path: str, seen: set[int] | None = None) -> Any:
    """Copy JSON-safe details into immutable containers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} must contain only JSON-serializable values")
        return value
    if isinstance(value, (bytes, bytearray, memoryview, np.ndarray, np.generic)):
        raise TypeError(f"{path} must contain only JSON-serializable values")

    active = seen if seen is not None else set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise TypeError(f"{path} must not contain recursive values")
        active.add(marker)
        try:
            frozen: dict[str, Any] = {}
            for key in sorted(value):
                if not isinstance(key, str):
                    raise TypeError(f"{path} keys must be strings")
                frozen[key] = _freeze_json(value[key], path=f"{path}.{key}", seen=active)
            return MappingProxyType(frozen)
        finally:
            active.remove(marker)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise TypeError(f"{path} must not contain recursive values")
        active.add(marker)
        try:
            return tuple(_freeze_json(item, path=f"{path}[{index}]", seen=active)
                         for index, item in enumerate(value))
        finally:
            active.remove(marker)
    raise TypeError(f"{path} must contain only JSON-serializable values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _serialize_quality_metadata(value: Any, *, path: str) -> Any:
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return _json_safe_metadata(value, path=path)


def _json_safe_metadata(value: Any, *, path: str, seen: set[int] | None = None) -> Any:
    """Convert supported metadata, including dataclass reports, to JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} must contain only JSON-serializable values")
        return value
    if isinstance(value, Enum):
        return _json_safe_metadata(value.value, path=path, seen=seen)
    if isinstance(value, (bytes, bytearray, memoryview, np.ndarray, np.generic)):
        raise TypeError(f"{path} must contain only JSON-serializable values")

    active = seen if seen is not None else set()
    if is_dataclass(value) and not isinstance(value, type):
        marker = id(value)
        if marker in active:
            raise TypeError(f"{path} must not contain recursive values")
        active.add(marker)
        try:
            return {
                item.name: _json_safe_metadata(getattr(value, item.name),
                                               path=f"{path}.{item.name}", seen=active)
                for item in fields(value)
            }
        finally:
            active.remove(marker)
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise TypeError(f"{path} must not contain recursive values")
        active.add(marker)
        try:
            converted: dict[str, Any] = {}
            for key in sorted(value):
                if not isinstance(key, str):
                    raise TypeError(f"{path} keys must be strings")
                converted[key] = _json_safe_metadata(value[key], path=f"{path}.{key}", seen=active)
            return converted
        finally:
            active.remove(marker)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise TypeError(f"{path} must not contain recursive values")
        active.add(marker)
        try:
            return [_json_safe_metadata(item, path=f"{path}[{index}]", seen=active)
                    for index, item in enumerate(value)]
        finally:
            active.remove(marker)
    raise TypeError(f"{path} must contain only JSON-serializable values")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_config_json(config: PreprocessConfig) -> str:
    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if hasattr(value, "as_posix"):
            return value.as_posix()
        return value
    normalized = normalize(asdict(config))
    # debug first for a stable representation, then lexical order.
    ordered = {"debug": normalized["debug"]}
    ordered.update({key: normalized[key] for key in sorted(normalized) if key != "debug"})
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def hash_array_input(image: np.ndarray, color_order: str) -> str:
    contiguous = np.ascontiguousarray(image)
    header = json.dumps({"shape": contiguous.shape, "dtype": str(contiguous.dtype),
                         "color_order": color_order}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(header + b"\0" + contiguous.tobytes())


def hash_image_pixels(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    header = f"{contiguous.shape}|{contiguous.dtype}".encode("ascii")
    return sha256_bytes(header + b"\0" + contiguous.tobytes())
