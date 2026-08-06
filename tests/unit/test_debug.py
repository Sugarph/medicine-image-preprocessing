from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from medicine_preprocess import PreprocessConfig, preprocess_image
from medicine_preprocess.debug import DebugError, DebugSink
from medicine_preprocess.image_io import PreprocessError
from medicine_preprocess.validation import ValidationVerdict


def _operation_semantics(result):
    return [
        {key: value for key, value in operation.items() if key != "duration_ms"}
        for operation in (record.to_dict() for record in result.operations)
    ]


def test_disabled_debug_sink_writes_nothing(tmp_path: Path) -> None:
    sink = DebugSink(enabled=False, output_dir=tmp_path, source_key="item")

    assert sink.write_image("canonical", np.zeros((2, 2, 3), dtype=np.uint8)) is None
    assert sink.write_json("report", {"status": "ok"}) is None
    assert sink.finalize() is None
    sink.abort()
    assert not list(tmp_path.iterdir())


def test_enabled_sink_atomically_writes_png_and_json(tmp_path: Path) -> None:
    sink = DebugSink(enabled=True, output_dir=tmp_path, source_key="item")

    sink.write_image("canonical", np.zeros((2, 2, 3), dtype=np.uint8))
    sink.write_json("report", {"status": "ok"})
    final = sink.finalize()

    assert final == tmp_path / "item"
    assert (tmp_path / "item" / "canonical.png").is_file()
    assert (tmp_path / "item" / "report.json").is_file()
    assert json.loads((tmp_path / "item" / "report.json").read_text(encoding="utf-8")) == {
        "status": "ok"
    }
    assert not list((tmp_path / "item").glob("*.tmp"))
    assert not list(tmp_path.glob(".item.tmp-*"))


def test_source_key_is_sanitized_and_contained(tmp_path: Path) -> None:
    sink = DebugSink(enabled=True, output_dir=tmp_path, source_key="../unsafe name")

    assert sink.key == "unsafe_name"
    assert sink.final is not None
    assert sink.final.parent == tmp_path
    sink.abort()
    assert not (tmp_path / "unsafe_name").exists()
    assert not list(tmp_path.glob(".unsafe_name.tmp-*"))


def test_existing_final_directory_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "item").mkdir()

    with pytest.raises(DebugError, match="already exists"):
        DebugSink(enabled=True, output_dir=tmp_path, source_key="item")


def test_duplicate_artifact_name_is_rejected(tmp_path: Path) -> None:
    sink = DebugSink(enabled=True, output_dir=tmp_path, source_key="item")
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    sink.write_image("same", image)

    with pytest.raises(DebugError, match="already exists"):
        sink.write_image("same", image)
    with pytest.raises(DebugError, match="already exists"):
        sink.write_json("same", {"status": "duplicate"})
    sink.abort()


def test_debug_encode_failure_is_typed(monkeypatch, tmp_path: Path) -> None:
    sink = DebugSink(enabled=True, output_dir=tmp_path, source_key="../unsafe name")
    monkeypatch.setattr(cv2, "imencode", lambda *args, **kwargs: (False, None))

    with pytest.raises(DebugError, match="unable to encode"):
        sink.write_image("canonical", np.zeros((2, 2, 3), dtype=np.uint8))
    sink.abort()
    assert not list(tmp_path.glob(".unsafe_name.tmp-*"))


def test_debug_json_serialization_failure_is_typed(tmp_path: Path) -> None:
    sink = DebugSink(enabled=True, output_dir=tmp_path, source_key="item")

    with pytest.raises(DebugError, match="unable to write debug JSON"):
        sink.write_json("bad", {"object": object()})
    sink.abort()
    assert not list(tmp_path.glob(".item.tmp-*"))


def test_debug_filesystem_failure_is_typed_and_abortable(monkeypatch, tmp_path: Path) -> None:
    sink = DebugSink(enabled=True, output_dir=tmp_path, source_key="item")
    monkeypatch.setattr(Path, "write_bytes", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk")))

    with pytest.raises(DebugError, match="unable to write debug image"):
        sink.write_image("canonical", np.zeros((2, 2, 3), dtype=np.uint8))
    sink.abort()
    assert not list(tmp_path.glob(".item.tmp-*"))


def test_debug_finalize_failure_is_typed_and_abortable(monkeypatch, tmp_path: Path) -> None:
    sink = DebugSink(enabled=True, output_dir=tmp_path, source_key="item")
    sink.write_json("report", {"status": "ok"})
    monkeypatch.setattr("medicine_preprocess.debug.os.replace", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("rename")))

    with pytest.raises(DebugError, match="unable to finalize"):
        sink.finalize()
    sink.abort()
    assert not list(tmp_path.glob(".item.tmp-*"))
    assert not (tmp_path / "item").exists()


def test_pipeline_debug_writes_only_accepted_checkpoints(tmp_path: Path) -> None:
    image = np.full((48, 64, 3), 100, dtype=np.uint8)
    config = replace(
        PreprocessConfig(preset_name="baseline", preset_version="1"),
        debug=replace(PreprocessConfig(preset_name="baseline", preset_version="1").debug, enabled=True, output_dir=tmp_path),
    )

    result = preprocess_image(image, config, source_id="debug-item")
    output = tmp_path / "debug-item"

    assert (output / "canonical.png").is_file()
    assert (output / "final.png").is_file()
    assert not (output / "crop.png").exists()
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["output_image_hash"] == result.output_image_hash
    assert not list(tmp_path.glob(".debug-item.tmp-*"))


def test_debug_does_not_change_processing_result(tmp_path: Path) -> None:
    image = np.full((48, 64, 3), 100, dtype=np.uint8)
    plain = PreprocessConfig(preset_name="baseline", preset_version="1")
    debug = replace(
        plain,
        debug=replace(plain.debug, enabled=True, output_dir=tmp_path),
    )

    first = preprocess_image(image, plain, source_id="same")
    second = preprocess_image(image, debug, source_id="same")

    assert np.array_equal(first.image, second.image)
    assert first.output_image_hash == second.output_image_hash
    assert first.input_hash == second.input_hash
    assert first.canonical_image_hash == second.canonical_image_hash
    assert first.original_to_final_transform.tolist() == second.original_to_final_transform.tolist()
    assert first.final_to_original_transform.tolist() == second.final_to_original_transform.tolist()
    assert _operation_semantics(first) == _operation_semantics(second)
    assert first.crop == second.crop
    assert first.resize_scale_factor == second.resize_scale_factor
    assert first.resize_scale_x == second.resize_scale_x
    assert first.resize_scale_y == second.resize_scale_y
    assert first.warnings == second.warnings == ()


def test_debug_failure_adds_warning_without_changing_processing(monkeypatch, tmp_path: Path) -> None:
    import medicine_preprocess.debug as debug

    image = np.full((48, 64, 3), 100, dtype=np.uint8)
    plain = PreprocessConfig(preset_name="baseline", preset_version="1")
    config = replace(plain, debug=replace(plain.debug, enabled=True, output_dir=tmp_path))
    first = preprocess_image(image, plain, source_id="same")
    monkeypatch.setattr(
        debug.DebugSink,
        "write_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(DebugError("forced debug")),
    )

    second = preprocess_image(image.copy(), config, source_id="same")

    assert np.array_equal(first.image, second.image)
    assert first.output_image_hash == second.output_image_hash
    assert first.original_to_final_transform.tolist() == second.original_to_final_transform.tolist()
    assert first.final_to_original_transform.tolist() == second.final_to_original_transform.tolist()
    assert _operation_semantics(first) == _operation_semantics(second)
    assert second.warnings == ("debug:canonical:DebugError",)
    assert not list(tmp_path.glob(".same.tmp-*"))


def test_pipeline_debug_aborts_on_processing_error(monkeypatch, tmp_path: Path) -> None:
    image = np.full((48, 64, 3), 100, dtype=np.uint8)
    plain = PreprocessConfig(preset_name="baseline", preset_version="1")
    config = replace(
        plain,
        debug=replace(plain.debug, enabled=True, output_dir=tmp_path),
    )

    def fail_terminal_validation(*args, **kwargs):
        return ValidationVerdict(False, "forced terminal validation", {})

    monkeypatch.setattr(
        "medicine_preprocess.pipeline.validate_final_structure",
        fail_terminal_validation,
    )

    with pytest.raises(PreprocessError, match="forced terminal validation"):
        preprocess_image(image, config, source_id="terminal-failure")

    assert not list(tmp_path.glob(".terminal-failure.tmp-*"))
    assert not (tmp_path / "terminal-failure").exists()
