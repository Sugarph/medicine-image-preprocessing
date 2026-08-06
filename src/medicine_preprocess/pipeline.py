from __future__ import annotations

from dataclasses import replace
import time

import cv2
import numpy as np

from .config import PreprocessConfig, ResizeConfig
from .crop import apply_crop
from .debug import DebugError, DebugSink
from .enhancement import (
    apply_clahe_luminance,
    apply_denoise,
    apply_gamma,
    apply_gray_world_white_balance,
    apply_percentile_exposure,
    apply_unsharp_mask,
    automatic_gamma,
    choose_operations,
)
from .geometry import (
    TransformState,
    apply_explicit_rotation,
    apply_perspective_correction,
    estimate_small_skew,
    rotate_expand,
)
from .image_io import ImageSource, PreprocessError, decode_and_canonicalize
from .presets import load_quality_thresholds_v2
from .quality import QualityReport, analyze_image_quality
from .resize import resize_preserving_aspect_ratio
from .result import (
    PIPELINE_VERSION,
    RESULT_SCHEMA_VERSION,
    STABLE_PIPELINE_VERSION,
    CropMetadata,
    OperationRecord,
    OperationStatus,
    PreprocessResult,
    RuntimeEnvironment,
    canonical_config_json,
    hash_image_pixels,
    sha256_bytes,
)
from .validation import (
    validate_clahe_change,
    validate_denoise_change,
    validate_exposure_change,
    validate_final_structure,
    validate_deskew,
    validate_white_balance_change,
    validate_sharpen_change,
)


def _exception_details(exc: Exception) -> dict[str, str]:
    return {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }


def _timed_record(record: OperationRecord, started: float) -> OperationRecord:
    """Attach the measured stage duration without changing its semantics."""
    return replace(record, duration_ms=(time.perf_counter() - started) * 1000.0)


def _capped_copy_for_quality_preview(image: np.ndarray, max_long_side: int) -> np.ndarray:
    """Downscale-only copy for quality_before; never touches the real output."""
    height, width = image.shape[:2]
    scale = min(1.0, max_long_side / max(height, width))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


class _DebugTransaction:
    """Own the optional sink for the full preprocessing transaction."""

    def __init__(self) -> None:
        self.sink: DebugSink | None = None

    def abort(self) -> None:
        if self.sink is None:
            return
        try:
            self.sink.abort()
        except Exception:
            # Cleanup must never mask a normal preprocessing exception.
            pass


def _run_validated_pixel_stage(name, image, *, enabled, apply, validate):
    started = time.perf_counter()
    if not enabled:
        return (
            image,
            _timed_record(
                OperationRecord(name, OperationStatus.SKIPPED, "disabled_or_not_required"),
                started,
            ),
            False,
        )
    try:
        candidate = np.ascontiguousarray(apply(image.copy()), dtype=np.uint8)
        verdict = validate(image, candidate)
    except Exception as exc:
        return (
            image,
            _timed_record(
                OperationRecord(
                    name,
                    OperationStatus.FAILED_SAFELY,
                    type(exc).__name__,
                    details=_exception_details(exc),
                ),
                started,
            ),
            True,
        )
    if not verdict.accepted:
        return (
            image,
            _timed_record(
                OperationRecord(
                    name,
                    OperationStatus.REVERTED,
                    verdict.reason,
                    details=verdict.details,
                ),
                started,
            ),
            True,
        )
    return (
        candidate,
        _timed_record(
            OperationRecord(
                name,
                OperationStatus.APPLIED,
                verdict.reason,
                details=verdict.details,
            ),
            started,
        ),
        False,
    )


def _run_validated_resize_stage(image, state, config):
    started = time.perf_counter()
    checkpoint = np.ascontiguousarray(image.copy(), dtype=np.uint8)
    checkpoint_state = state
    try:
        outcome = resize_preserving_aspect_ratio(checkpoint, config, checkpoint_state)
        verdict = validate_final_structure(
            outcome.image,
            outcome.transform.forward,
            outcome.transform.inverse,
        )
    except Exception as exc:
        return (
            checkpoint,
            checkpoint_state,
            1.0,
            1.0,
            1.0,
            _timed_record(
                OperationRecord(
                    "resize",
                    OperationStatus.FAILED_SAFELY,
                    type(exc).__name__,
                    details=_exception_details(exc),
                ),
                started,
            ),
            True,
        )
    if not verdict.accepted:
        return (
            checkpoint,
            checkpoint_state,
            1.0,
            1.0,
            1.0,
            _timed_record(
                OperationRecord(
                    "resize",
                    OperationStatus.REVERTED,
                    verdict.reason,
                    details=getattr(verdict, "details", {}),
                ),
                started,
            ),
            True,
        )
    return (
        np.ascontiguousarray(outcome.image, dtype=np.uint8),
        outcome.transform,
        outcome.resize_scale_factor,
        outcome.resize_scale_x,
        outcome.resize_scale_y,
        _timed_record(outcome.record, started),
        False,
    )


def preprocess_image(
    image: ImageSource,
    config: PreprocessConfig | None = None,
    *,
    source_id: str | None = None,
) -> PreprocessResult:
    transaction = _DebugTransaction()
    try:
        return _preprocess_image_impl(
            image,
            config,
            source_id=source_id,
            debug_transaction=transaction,
        )
    finally:
        transaction.abort()


def _preprocess_image_impl(
    image: ImageSource,
    config: PreprocessConfig | None = None,
    *,
    source_id: str | None = None,
    debug_transaction: _DebugTransaction,
) -> PreprocessResult:
    active = PreprocessConfig.grabcut_experimental() if config is None else config
    if not isinstance(active, PreprocessConfig):
        raise TypeError("config must be a PreprocessConfig")

    canonical_started = time.perf_counter()
    canonical = decode_and_canonicalize(image, active.input)
    working = np.ascontiguousarray(canonical.image.copy(), dtype=np.uint8)
    state = TransformState(canonical.original_to_canonical, np.eye(3, dtype=np.float64))
    canonical_record = OperationRecord("canonicalize", OperationStatus.APPLIED, "valid_bgr_uint8")
    canonical_record = _timed_record(canonical_record, canonical_started)
    operations = [canonical_record]
    experimental_document_geometry = (
        active.preset_name == "grabcut_experimental" and active.preset_version == "1"
    )
    warnings: list[str] = []
    debug_sink: DebugSink | None = None
    if active.debug.enabled:
        try:
            debug_sink = DebugSink(
                enabled=True,
                output_dir=active.debug.output_dir,
                source_key=source_id or canonical.input_hash,
            )
        except Exception as exc:
            warnings.append(f"debug:initialize:{type(exc).__name__}")
    debug_transaction.sink = debug_sink

    def debug_checkpoint(
        stage: str,
        image_checkpoint: np.ndarray,
        record: OperationRecord | None = None,
    ) -> None:
        nonlocal debug_sink
        if debug_sink is None:
            return
        if record is not None and record.status is not OperationStatus.APPLIED:
            return
        try:
            debug_sink.write_image(
                stage,
                np.ascontiguousarray(image_checkpoint.copy(), dtype=np.uint8),
            )
        except Exception as exc:
            warnings.append(f"debug:{stage}:{type(exc).__name__}")
            try:
                debug_sink.abort()
            except Exception:
                pass
            debug_sink = None

    debug_checkpoint("canonical", working)
    fallback_used = False
    thresholds = load_quality_thresholds_v2() if experimental_document_geometry else None
    pre_crop_analysis_cap = active.quality.pre_crop_analysis_max_long_side
    quality_before: QualityReport | None = None
    if thresholds is not None:
        if pre_crop_analysis_cap is not None:
            # Observability only; no thresholds, so classifications stays None.
            quality_before = analyze_image_quality(
                _capped_copy_for_quality_preview(canonical.image, pre_crop_analysis_cap)
            )
        else:
            quality_before = analyze_image_quality(canonical.image, thresholds)

    crop_outcome = None
    crop_started = time.perf_counter()
    try:
        crop_outcome = apply_crop(
            working,
            active.crop,
            state,
            experimental=experimental_document_geometry,
        )
        working, state, crop_metadata = (
            crop_outcome.image,
            crop_outcome.transform,
            crop_outcome.metadata,
        )
        crop_record = _timed_record(crop_outcome.record, crop_started)
        operations.append(crop_record)
        debug_checkpoint("crop", working, crop_record)
        fallback_used |= crop_outcome.fallback_used
    except (cv2.error, ValueError, ArithmeticError) as exc:
        crop_metadata = CropMetadata()
        operations.append(
            _timed_record(
                OperationRecord("crop", OperationStatus.FAILED_SAFELY, type(exc).__name__),
                crop_started,
            )
        )
        fallback_used = True

    if (
        experimental_document_geometry
        and active.geometry.perspective_enabled
        and crop_outcome is not None
        and crop_outcome.quadrilateral is not None
    ):
        perspective_started = time.perf_counter()
        try:
            perspective_outcome = apply_perspective_correction(
                working,
                crop_outcome.quadrilateral.corners,
                state,
            )
            working, state = perspective_outcome.image, perspective_outcome.transform
            perspective_record = _timed_record(perspective_outcome.record, perspective_started)
            operations.append(perspective_record)
            debug_checkpoint("perspective", working, perspective_record)
        except (cv2.error, ValueError, ArithmeticError) as exc:
            crop_metadata = CropMetadata()
            operations.append(
                _timed_record(
                    OperationRecord(
                        "perspective",
                        OperationStatus.FAILED_SAFELY,
                        type(exc).__name__,
                    ),
                    perspective_started,
                )
            )
            fallback_used = True

    if experimental_document_geometry:
        deskew_started = time.perf_counter()
        try:
            deskew_estimate = estimate_small_skew(working, active.geometry)
            if not deskew_estimate.accepted:
                operations.append(
                    _timed_record(
                        OperationRecord(
                            "deskew",
                            OperationStatus.SKIPPED,
                            deskew_estimate.reason,
                            details={
                                "measured_degrees": deskew_estimate.measured_degrees,
                                "weighted_mad_degrees": deskew_estimate.weighted_mad_degrees,
                            },
                        ),
                        deskew_started,
                    )
                )
            else:
                deskew_outcome = rotate_expand(
                    working,
                    deskew_estimate.correction_degrees,
                    state,
                )
                deskew_verdict = validate_deskew(working, deskew_outcome)
                if not deskew_verdict.accepted:
                    operations.append(
                        _timed_record(
                            OperationRecord(
                                "deskew",
                                OperationStatus.REVERTED,
                                deskew_verdict.reason,
                                details=deskew_verdict.details,
                            ),
                            deskew_started,
                        )
                    )
                    fallback_used = True
                else:
                    working, state = deskew_outcome.image, deskew_outcome.transform
                    deskew_record = _timed_record(deskew_outcome.record, deskew_started)
                    operations.append(deskew_record)
                    debug_checkpoint("deskew", working, deskew_record)
        except Exception as exc:
            operations.append(
                _timed_record(
                    OperationRecord(
                        "deskew",
                        OperationStatus.FAILED_SAFELY,
                        type(exc).__name__,
                        details=_exception_details(exc),
                    ),
                    locals().get("deskew_started", time.perf_counter()),
                )
            )
            fallback_used = True

    rotation_started = time.perf_counter()
    try:
        rotation_outcome = apply_explicit_rotation(working, active.geometry.right_angle_degrees, state)
        working, state = rotation_outcome.image, rotation_outcome.transform
        rotation_record = _timed_record(rotation_outcome.record, rotation_started)
        operations.append(rotation_record)
        debug_checkpoint("explicit_rotation", working, rotation_record)
    except (cv2.error, ValueError, ArithmeticError) as exc:
        operations.append(
            _timed_record(
                OperationRecord("explicit_rotation", OperationStatus.FAILED_SAFELY, type(exc).__name__),
                rotation_started,
            )
        )
        fallback_used = True

    pre_resize_scale_x, pre_resize_scale_y = 1.0, 1.0
    if active.resize.pre_enhancement_max_long_side is not None:
        pre_cap_config = ResizeConfig(
            mode="fit_long_side",
            maximum_long_side=active.resize.pre_enhancement_max_long_side,
            maximum_upscale_factor=1.0,
        )
        (
            working,
            state,
            _pre_resize_scale_factor,
            pre_resize_scale_x,
            pre_resize_scale_y,
            pre_resize_record,
            pre_resize_fallback,
        ) = _run_validated_resize_stage(working, state, pre_cap_config)
        pre_resize_record = replace(pre_resize_record, name="pre_enhancement_resize")
        operations.append(pre_resize_record)
        debug_checkpoint("pre_enhancement_resize", working, pre_resize_record)
        fallback_used |= pre_resize_fallback

    working_quality: QualityReport | None = None
    decision = None
    if thresholds is not None:
        working_quality = (
            quality_before
            if (
                pre_crop_analysis_cap is None
                and quality_before is not None
                and np.array_equal(working, canonical.image)
            )
            else analyze_image_quality(working, thresholds)
        )
        decision = choose_operations(
            working_quality,
            thresholds,
            glare_warning=(
                working_quality.measurements.glare_fraction > 0.02
                or working_quality.measurements.largest_glare_fraction > 0.01
            ),
            color_cast_strong=working_quality.measurements.channel_mean_spread > 15.0,
            short_side=min(working.shape[:2]),
            minimum_short_side=active.resize.minimum_short_side,
        )

    if decision is None:
        working, record, reverted = _run_validated_pixel_stage(
            "gamma",
            working,
            enabled=active.enhancement.gamma_mode == "manual" and active.enhancement.gamma != 1.0,
            apply=lambda source: apply_gamma(source, active.enhancement.gamma),
            validate=lambda before, after: validate_exposure_change(
                before,
                after,
                target_luminance=128.0,
                maximum_new_clipped_fraction=active.validation.maximum_new_clipped_fraction,
            ),
        )
        operations.append(record)
        debug_checkpoint("gamma", working, record)
        fallback_used |= reverted

        working, record, reverted = _run_validated_pixel_stage(
            "clahe",
            working,
            enabled=active.enhancement.contrast_mode == "clahe",
            apply=lambda source: apply_clahe_luminance(
                source,
                clip_limit=active.enhancement.clahe_clip_limit,
                grid_size=active.enhancement.clahe_grid_size,
            ),
            validate=lambda before, after: validate_clahe_change(
                before,
                after,
                maximum_new_clipped_fraction=active.validation.maximum_new_clipped_fraction,
            ),
        )
        operations.append(record)
        debug_checkpoint("clahe", working, record)
        fallback_used |= reverted

        working, record, reverted = _run_validated_pixel_stage(
            "denoise",
            working,
            enabled=active.enhancement.denoise_mode in {"median", "bilateral", "nlmeans"},
            apply=lambda source: apply_denoise(
                source,
                active.enhancement.denoise_mode,
                active.enhancement.denoise_strength,
            ),
            validate=lambda before, after: validate_denoise_change(
                before,
                after,
                maximum_edge_loss_fraction=active.validation.maximum_edge_loss_fraction,
            ),
        )
        operations.append(record)
        debug_checkpoint("denoise", working, record)
        fallback_used |= reverted

        working, record, reverted = _run_validated_pixel_stage(
            "sharpen",
            working,
            enabled=active.enhancement.sharpen_mode == "unsharp",
            apply=lambda source: apply_unsharp_mask(
                source,
                sigma=active.enhancement.unsharp_sigma,
                amount=active.enhancement.unsharp_amount,
                threshold=active.enhancement.unsharp_threshold,
            ),
            validate=lambda before, after: validate_sharpen_change(
                before,
                after,
                maximum_new_clipped_fraction=active.validation.maximum_new_clipped_fraction,
                maximum_noise_increase_fraction=active.validation.maximum_noise_increase_fraction,
            ),
        )
        operations.append(record)
        debug_checkpoint("sharpen", working, record)
        fallback_used |= reverted
    else:
        assert working_quality is not None
        working, record, reverted = _run_validated_pixel_stage(
            "white_balance",
            working,
            enabled=active.enhancement.white_balance_automatic and decision.white_balance,
            apply=apply_gray_world_white_balance,
            validate=lambda before, after: validate_white_balance_change(
                before,
                after,
                maximum_new_clipped_fraction=active.validation.maximum_new_clipped_fraction,
            ),
        )
        operations.append(record)
        debug_checkpoint("white_balance", working, record)
        fallback_used |= reverted

        working, record, reverted = _run_validated_pixel_stage(
            "percentile_exposure",
            working,
            enabled=active.enhancement.gamma_mode == "automatic" and decision.percentile_exposure,
            apply=apply_percentile_exposure,
            validate=lambda before, after: validate_exposure_change(
                before,
                after,
                target_luminance=128.0,
                maximum_new_clipped_fraction=active.validation.maximum_new_clipped_fraction,
            ),
        )
        operations.append(record)
        debug_checkpoint("percentile_exposure", working, record)
        fallback_used |= reverted

        working, record, reverted = _run_validated_pixel_stage(
            "gamma",
            working,
            enabled=active.enhancement.gamma_mode == "automatic" and decision.gamma,
            apply=lambda source: apply_gamma(
                source, automatic_gamma(working_quality.measurements.luminance_p50)
            ),
            validate=lambda before, after: validate_exposure_change(
                before,
                after,
                target_luminance=128.0,
                maximum_new_clipped_fraction=active.validation.maximum_new_clipped_fraction,
            ),
        )
        operations.append(record)
        debug_checkpoint("gamma", working, record)
        fallback_used |= reverted

        working, record, reverted = _run_validated_pixel_stage(
            "clahe",
            working,
            enabled=active.enhancement.contrast_mode == "automatic" and decision.clahe,
            apply=lambda source: apply_clahe_luminance(
                source,
                clip_limit=active.enhancement.clahe_clip_limit,
                grid_size=active.enhancement.clahe_grid_size,
            ),
            validate=lambda before, after: validate_clahe_change(
                before,
                after,
                maximum_new_clipped_fraction=active.validation.maximum_new_clipped_fraction,
            ),
        )
        operations.append(record)
        debug_checkpoint("clahe", working, record)
        fallback_used |= reverted

        working, record, reverted = _run_validated_pixel_stage(
            "denoise",
            working,
            enabled=active.enhancement.denoise_mode == "automatic" and decision.denoise,
            apply=lambda source: apply_denoise(
                source,
                decision.denoise_mode,
                active.enhancement.denoise_strength,
            ),
            validate=lambda before, after: validate_denoise_change(
                before,
                after,
                maximum_edge_loss_fraction=active.validation.maximum_edge_loss_fraction,
            ),
        )
        operations.append(record)
        debug_checkpoint("denoise", working, record)
        fallback_used |= reverted

        working, record, reverted = _run_validated_pixel_stage(
            "sharpen",
            working,
            enabled=active.enhancement.sharpen_mode == "automatic" and decision.sharpen,
            apply=lambda source: apply_unsharp_mask(
                source,
                sigma=active.enhancement.unsharp_sigma,
                amount=active.enhancement.unsharp_amount,
                threshold=active.enhancement.unsharp_threshold,
            ),
            validate=lambda before, after: validate_sharpen_change(
                before,
                after,
                maximum_new_clipped_fraction=active.validation.maximum_new_clipped_fraction,
                maximum_noise_increase_fraction=active.validation.maximum_noise_increase_fraction,
            ),
        )
        operations.append(record)
        debug_checkpoint("sharpen", working, record)
        fallback_used |= reverted

    resize_config = active.resize
    automatic_resize_not_selected = decision is not None and not decision.resize
    if automatic_resize_not_selected:
        resize_config = replace(active.resize, mode="none")
    (
        working,
        state,
        resize_scale_factor,
        resize_scale_x,
        resize_scale_y,
        resize_record,
        resize_fallback,
    ) = _run_validated_resize_stage(working, state, resize_config)
    if automatic_resize_not_selected and resize_record.status == OperationStatus.SKIPPED:
        resize_record = replace(resize_record, reason="automatic_not_selected")
    operations.append(resize_record)
    debug_checkpoint("resize", working, resize_record)
    fallback_used |= resize_fallback

    # Combine with any pre-enhancement cap so resize_scale_* reflects the
    # full canonical-to-final scale, not just this (possibly no-op) stage.
    resize_scale_x *= pre_resize_scale_x
    resize_scale_y *= pre_resize_scale_y
    resize_scale_factor = (resize_scale_x + resize_scale_y) / 2.0

    final_started = time.perf_counter()
    final_verdict = validate_final_structure(working, state.forward, state.inverse)
    if not final_verdict.accepted:
        raise PreprocessError(final_verdict.reason)
    final_record = _timed_record(
        OperationRecord(
            "final_validation",
            OperationStatus.APPLIED,
            final_verdict.reason,
            details=final_verdict.details,
        ),
        final_started,
    )
    operations.append(final_record)
    debug_checkpoint("final", working, final_record)

    quality_after = (
        None
        if thresholds is None
        else (
            quality_before
            if (
                pre_crop_analysis_cap is None
                and quality_before is not None
                and np.array_equal(working, canonical.image)
            )
            else analyze_image_quality(working, thresholds)
        )
    )

    config_json = canonical_config_json(active)
    config_hash = sha256_bytes(config_json.encode("utf-8"))
    pipeline_version = PIPELINE_VERSION if experimental_document_geometry else STABLE_PIPELINE_VERSION

    def build_result() -> PreprocessResult:
        return PreprocessResult(
            image=np.ascontiguousarray(working, dtype=np.uint8),
            operations=tuple(operations),
            original_to_final_transform=state.forward,
            final_to_original_transform=state.inverse,
            crop=crop_metadata,
            resize_scale_factor=resize_scale_factor,
            resize_scale_x=resize_scale_x,
            resize_scale_y=resize_scale_y,
            warnings=tuple(warnings),
            fallback_used=fallback_used,
            quality_before=quality_before,
            quality_after=quality_after,
            pipeline_version=pipeline_version,
            result_schema_version=RESULT_SCHEMA_VERSION,
            preset_name=active.preset_name,
            preset_version=active.preset_version,
            config_json=config_json,
            config_hash=config_hash,
            input_hash=canonical.input_hash,
            canonical_image_hash=canonical.canonical_image_hash,
            output_image_hash=hash_image_pixels(working),
            runtime_environment=RuntimeEnvironment.current(),
            source_id=source_id,
        )

    result = build_result()
    if debug_sink is not None:
        try:
            try:
                report_payload = result.to_dict()
            except Exception as exc:
                raise DebugError("unable to serialize debug report") from exc
            debug_sink.write_json("report", report_payload)
        except Exception as exc:
            warnings.append(f"debug:report:{type(exc).__name__}")
            try:
                debug_sink.abort()
            except Exception:
                pass
            debug_sink = None
            return build_result()
        try:
            debug_sink.finalize()
        except Exception as exc:
            warnings.append(f"debug:finalize:{type(exc).__name__}")
            try:
                debug_sink.abort()
            except Exception:
                pass
            debug_sink = None
            return build_result()
    return result
