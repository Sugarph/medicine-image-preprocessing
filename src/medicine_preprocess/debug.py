from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class DebugError(RuntimeError):
    """A debug artifact could not be staged or finalized safely."""


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_name(value: str, *, fallback: str) -> str:
    if not isinstance(value, str):
        raise DebugError("debug name must be a string")
    sanitized = _SAFE_NAME.sub("_", value).strip("._")
    return sanitized or fallback


class DebugSink:
    """Stage optional debug artifacts in a private directory, then publish atomically."""

    def __init__(self, *, enabled: bool, output_dir: Path | None, source_key: str) -> None:
        self.enabled = bool(enabled)
        self.root = None if output_dir is None else Path(output_dir)
        self.key = _sanitize_name(source_key, fallback="image")
        self.final = None if self.root is None else self.root / self.key
        self.temporary: Path | None = None
        # Track stems, not filenames, so a name can't be reused across artifact types.
        self._written: set[str] = set()
        self._finalized = False

        if not self.enabled:
            return
        if self.root is None:
            raise DebugError("enabled debug sink requires output_dir")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.final is not None and self.final.exists():
                raise DebugError(f"debug output already exists: {self.final}")
            self.temporary = self.root / f".{self.key}.tmp-{uuid.uuid4().hex}"
            self.temporary.mkdir(parents=False, exist_ok=False)
        except DebugError:
            raise
        except (OSError, ValueError) as exc:
            raise DebugError(f"unable to initialize debug output: {self.key}") from exc

    def _artifact_path(self, name: str, suffix: str) -> Path:
        if not self.enabled or self.temporary is None:
            raise DebugError("debug sink is disabled")
        safe_name = _sanitize_name(name, fallback="artifact")
        filename = f"{safe_name}{suffix}"
        if safe_name in self._written:
            raise DebugError(f"debug artifact already exists: {safe_name}")
        self._written.add(safe_name)
        return self.temporary / filename

    @staticmethod
    def _atomic_write(path: Path, data: bytes, *, label: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            temporary.write_bytes(data)
            # Reopen r+b: fsync on a read-only descriptor fails on Windows.
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except (OSError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            raise DebugError(label) from exc

    def write_image(self, stage: str, image: np.ndarray) -> Path | None:
        if not self.enabled:
            return None
        path = self._artifact_path(stage, ".png")
        try:
            ok, encoded = cv2.imencode(".png", image)
        except Exception as exc:
            self._written.discard(path.name[: -len(".png")])
            raise DebugError(f"unable to encode debug image: {stage}") from exc
        if not ok or encoded is None:
            self._written.discard(path.name[: -len(".png")])
            raise DebugError(f"unable to encode debug image: {stage}")
        try:
            self._atomic_write(path, encoded.tobytes(), label=f"unable to write debug image: {stage}")
        except DebugError:
            self._written.discard(path.name[: -len(".png")])
            raise
        return path

    def write_json(self, name: str, payload: dict[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        path = self._artifact_path(name, ".json")
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8") + b"\n"
        except (TypeError, ValueError, OverflowError) as exc:
            self._written.discard(path.name[: -len(".json")])
            raise DebugError(f"unable to write debug JSON: {name}") from exc
        try:
            self._atomic_write(path, encoded, label=f"unable to write debug JSON: {name}")
        except DebugError:
            self._written.discard(path.name[: -len(".json")])
            raise
        return path

    def finalize(self) -> Path | None:
        if not self.enabled:
            return None
        if self._finalized:
            raise DebugError("debug sink is already finalized")
        if self.temporary is None or self.final is None or self.root is None:
            raise DebugError("debug sink is not initialized")
        try:
            if self.final.exists():
                raise DebugError(f"debug output already exists: {self.final}")
            os.replace(self.temporary, self.final)
            self._finalized = True
            self.temporary = None
            return self.final
        except DebugError:
            raise
        except (OSError, ValueError) as exc:
            raise DebugError(f"unable to finalize debug output: {self.key}") from exc

    def abort(self) -> None:
        if not self.enabled or self.temporary is None or self.root is None:
            return
        try:
            temporary = self.temporary.resolve()
            root = self.root.resolve()
            temporary.relative_to(root)
        except (OSError, ValueError):
            return
        if not temporary.name.startswith(f".{self.key}.tmp-"):
            return
        shutil.rmtree(temporary, ignore_errors=True)
        self.temporary = None
