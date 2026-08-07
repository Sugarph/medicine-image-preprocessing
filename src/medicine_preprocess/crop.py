from __future__ import annotations

import multiprocessing as mp
import queue as queue_module
import threading
from dataclasses import dataclass, replace
from functools import lru_cache
import math

import cv2
import numpy as np

from .config import CropConfig
from .geometry import TransformState, map_points
from .result import CropMetadata, OperationRecord, OperationStatus


@dataclass(frozen=True)
class CropOutcome:
    image: np.ndarray
    transform: TransformState
    metadata: CropMetadata
    record: OperationRecord
    fallback_used: bool
    quadrilateral: "QuadrilateralCandidate | None" = None


@dataclass(frozen=True)
class QuadrilateralCandidate:
    corners: tuple[tuple[float, float], ...]
    confidence: float
    area_ratio: float
    rectangularity: float
    edge_support: float
    minimum_internal_angle: float
    maximum_internal_angle: float
    boundary_contacts: int
    opposing_edge_ratio: float
    perturbation_index: int


def _clamped_box(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    box = (max(0, left), max(0, top), min(width, right), min(height, bottom))
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("crop box must contain at least one pixel")
    return box


def _apply_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    transform: TransformState,
    reason: str,
) -> CropOutcome:
    left, top, right, bottom = box
    cropped = np.ascontiguousarray(image[top:bottom, left:right].copy())
    current_to_next = np.array(
        [[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float64
    )
    next_transform = transform.then(current_to_next)
    working_polygon = (
        (left - 0.5, top - 0.5),
        (right - 0.5, top - 0.5),
        (right - 0.5, bottom - 0.5),
        (left - 0.5, bottom - 0.5),
    )
    original_polygon = map_points(working_polygon, transform.inverse)
    canonical_polygon = map_points(working_polygon, transform.canonical_inverse)
    metadata = CropMetadata(box, original_polygon, canonical_polygon)
    record = OperationRecord(
        "crop", OperationStatus.APPLIED, reason, details={"box": box}
    )
    return CropOutcome(cropped, next_transform, metadata, record, False)


def _ordered_corners(points: np.ndarray | tuple[tuple[float, float], ...]) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if values.shape != (4, 2) or not np.isfinite(values).all():
        raise ValueError("quadrilateral corners must contain four finite points")
    center = values.mean(axis=0)
    angles = np.arctan2(values[:, 1] - center[1], values[:, 0] - center[0])
    ordered = values[np.argsort(angles, kind="mergesort")]
    start = int(np.argmin(np.sum(ordered, axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if cv2.contourArea(ordered.astype(np.float32)) < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def _polygon_area(corners: np.ndarray) -> float:
    return abs(float(cv2.contourArea(corners.astype(np.float32))))


def _internal_angles(corners: np.ndarray) -> tuple[float, ...]:
    angles = []
    for index in range(4):
        current = corners[index]
        previous = corners[(index - 1) % 4] - current
        following = corners[(index + 1) % 4] - current
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
        if denominator <= 1e-12:
            angles.append(0.0)
            continue
        cosine = float(np.dot(previous, following) / denominator)
        angles.append(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))
    return tuple(angles)


def _opposing_edge_ratio(corners: np.ndarray) -> float:
    lengths = [
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
        for index in range(4)
    ]
    if min(lengths) <= 1e-12:
        return float("inf")
    return max(lengths[0] / lengths[2], lengths[2] / lengths[0], lengths[1] / lengths[3], lengths[3] / lengths[1])


def _boundary_contacts(corners: np.ndarray, width: int, height: int) -> int:
    tolerance = max(2.0, 0.01 * min(width, height))
    contacts = 0
    if float(np.min(corners[:, 0])) <= tolerance:
        contacts += 1
    if float(np.min(corners[:, 1])) <= tolerance:
        contacts += 1
    if float(np.max(corners[:, 0])) >= width - 1 - tolerance:
        contacts += 1
    if float(np.max(corners[:, 1])) >= height - 1 - tolerance:
        contacts += 1
    return contacts


def _edge_support(corners: np.ndarray, edges: np.ndarray) -> float:
    height, width = edges.shape[:2]
    samples: list[bool] = []
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        length = max(2, int(round(float(np.linalg.norm(end - start)))))
        for fraction in np.linspace(0.0, 1.0, length, endpoint=True):
            point = np.rint(start + (end - start) * fraction).astype(np.int32)
            x, y = int(point[0]), int(point[1])
            left, right = max(0, x - 5), min(width, x + 6)
            top, bottom = max(0, y - 5), min(height, y + 6)
            samples.append(bool(np.any(edges[top:bottom, left:right] > 0)))
    return float(np.mean(samples)) if samples else 0.0


def _candidate_from_corners(
    corners: np.ndarray,
    image_shape: tuple[int, ...],
    perturbation_index: int,
    edge_support: float,
) -> QuadrilateralCandidate:
    height, width = image_shape[:2]
    ordered = _ordered_corners(corners)
    area_ratio = _polygon_area(ordered) / float(width * height)
    rectangle = cv2.minAreaRect(ordered.astype(np.float32))
    rectangle_area = max(1e-12, float(rectangle[1][0] * rectangle[1][1]))
    rectangularity = min(1.0, _polygon_area(ordered) / rectangle_area)
    internal = _internal_angles(ordered)
    minimum_angle, maximum_angle = min(internal), max(internal)
    boundary_contacts = _boundary_contacts(ordered, width, height)
    opposing_ratio = _opposing_edge_ratio(ordered)
    area_plausibility = 1.0
    corner_quality = float(
        np.clip(1.0 - np.mean(np.abs(np.asarray(internal) - 90.0)) / 45.0, 0.0, 1.0)
    )
    boundary_safety = max(0.0, 1.0 - boundary_contacts / 4.0)
    perspective_safety = float(np.clip(2.0 / max(1.0, opposing_ratio), 0.0, 1.0))
    confidence = float(
        0.25 * area_plausibility
        + 0.20 * rectangularity
        + 0.20 * edge_support
        + 0.15 * corner_quality
        + 0.10 * boundary_safety
        + 0.10 * perspective_safety
    )
    return QuadrilateralCandidate(
        corners=tuple((float(x), float(y)) for x, y in ordered),
        confidence=confidence,
        area_ratio=area_ratio,
        rectangularity=rectangularity,
        edge_support=float(edge_support),
        minimum_internal_angle=float(minimum_angle),
        maximum_internal_angle=float(maximum_angle),
        boundary_contacts=boundary_contacts,
        opposing_edge_ratio=float(opposing_ratio),
        perturbation_index=perturbation_index,
    )


@dataclass(frozen=True)
class _GrabCutMaskDiagnostics:
    has_contour: bool
    largest: np.ndarray | None
    box: tuple[int, int, int, int]  # bx, by, bw, bh in working coords
    area_ratio: float
    edge_touch_count: int
    dominant_share: float
    aspect_ratio: float
    near_init_rect_ratio: float


def _diagnose_grabcut_mask(
    fg_mask: np.ndarray, rect: tuple[int, int, int, int], margin: int, sw: int, sh: int
) -> _GrabCutMaskDiagnostics:
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _GrabCutMaskDiagnostics(False, None, (0, 0, 0, 0), 0.0, 0, 0.0, 0.0, 0.0)

    largest = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest)
    area_ratio = (bw * bh) / (sw * sh)

    edge_touch_count = sum(
        (
            bx <= margin,
            by <= margin,
            (bx + bw) >= (sw - margin),
            (by + bh) >= (sh - margin),
        )
    )

    total_fg = float(fg_mask.sum())
    dominant_share = float(cv2.contourArea(largest)) / total_fg if total_fg > 0 else 0.0
    aspect_ratio = max(bw, bh) / max(1, min(bw, bh))

    rx, ry, rw, rh = rect
    rect_slice = fg_mask[ry : ry + rh, rx : rx + rw]
    near_init_rect_ratio = float(rect_slice.mean()) if rect_slice.size else 0.0

    return _GrabCutMaskDiagnostics(
        True, largest, (bx, by, bw, bh), area_ratio, edge_touch_count, dominant_share, aspect_ratio, near_init_rect_ratio
    )


def _grabcut_early_reject_reason(diag: _GrabCutMaskDiagnostics, config: CropConfig) -> str | None:
    """Sanity checks after one GrabCut iteration; any hit skips the second pass."""
    if not diag.has_contour:
        return "no_foreground_contour"
    if diag.area_ratio < config.grabcut_min_area_ratio * 0.7 or diag.area_ratio > min(1.0, config.grabcut_max_area_ratio * 1.05):
        return "area_ratio_out_of_range"
    if diag.edge_touch_count >= 3:
        return "bbox_touches_all_edges"
    if diag.dominant_share < 0.5:
        return "no_dominant_component"
    if diag.aspect_ratio > 6.0:
        return "bbox_extremely_narrow"
    if diag.near_init_rect_ratio > 0.97:
        return "mask_matches_init_rectangle"
    return None


def _grabcut_is_clearly_confident(diag: _GrabCutMaskDiagnostics, config: CropConfig) -> bool:
    """Comfortable margin inside the accept thresholds; skips the second iteration."""
    comfortable_min = config.grabcut_min_area_ratio + 0.05
    comfortable_max = config.grabcut_max_area_ratio - 0.05
    return (
        comfortable_min <= diag.area_ratio <= comfortable_max
        and diag.edge_touch_count == 0
        and diag.dominant_share >= 0.85
        and diag.aspect_ratio <= 4.0
    )


def _grabcut_prepare_working_image(
    image: np.ndarray, config: CropConfig
) -> tuple[np.ndarray, float]:
    """Downscales to the GrabCut working resolution, before the process boundary."""
    height, width = image.shape[:2]
    work_max_dim = config.grabcut_work_max_dim
    scale = min(1.0, work_max_dim / max(height, width))
    small = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else np.ascontiguousarray(image.copy())
    )
    return small, scale


def _grabcut_detect_box_on_working(
    small: np.ndarray, scale: float, config: CropConfig
) -> tuple[
    tuple[int, int, int, int] | None,
    str,
    float | None,
    tuple[tuple[float, float], ...] | None,
]:
    """Runs GrabCut on a downscaled working image; returns box, reason, and area_ratio."""
    started = cv2.getTickCount()
    sh, sw = small.shape[:2]
    margin = max(2, int(round(0.03 * min(sh, sw))))
    rect = (margin, margin, sw - 2 * margin, sh - 2 * margin)

    mask = np.zeros((sh, sw), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    kernel = np.ones((5, 5), np.uint8)

    def cleaned_mask() -> np.ndarray:
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    try:
        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None, "grabcut_error", None, None

    fg_mask = cleaned_mask()
    diag = _diagnose_grabcut_mask(fg_mask, rect, margin, sw, sh)

    early_reject = _grabcut_early_reject_reason(diag, config)
    if early_reject is not None:
        return None, early_reject, (diag.area_ratio if diag.has_contour else None), None

    if not _grabcut_is_clearly_confident(diag, config):
        elapsed_ms = (cv2.getTickCount() - started) / cv2.getTickFrequency() * 1000
        max_iterations = max(1, config.grabcut_max_iterations)
        if max_iterations > 1 and elapsed_ms < config.grabcut_soft_budget_ms:
            try:
                cv2.grabCut(small, mask, None, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)
            except cv2.error:
                pass
            else:
                fg_mask = cleaned_mask()
                diag = _diagnose_grabcut_mask(fg_mask, rect, margin, sw, sh)
                early_reject = _grabcut_early_reject_reason(diag, config)
                if early_reject is not None:
                    return None, early_reject, (diag.area_ratio if diag.has_contour else None), None

    if not (config.grabcut_min_area_ratio <= diag.area_ratio <= config.grabcut_max_area_ratio):
        return None, "area_ratio_out_of_range", diag.area_ratio, None
    if diag.edge_touch_count >= 3:
        return None, "bbox_touches_all_edges", diag.area_ratio, None

    bx, by, bw, bh = diag.box
    inv_scale = 1.0 / scale
    full_box = (
        int(round(bx * inv_scale)),
        int(round(by * inv_scale)),
        int(round((bx + bw) * inv_scale)),
        int(round((by + bh) * inv_scale)),
    )

    quad_corners: tuple[tuple[float, float], ...] | None = None
    rotated_rect = cv2.minAreaRect(diag.largest)
    rect_area = max(1e-6, float(rotated_rect[1][0] * rotated_rect[1][1]))
    contour_area = float(cv2.contourArea(diag.largest))
    extent = contour_area / rect_area if rect_area > 0 else 0.0
    if extent >= config.grabcut_quad_min_extent:
        box_points = cv2.boxPoints(rotated_rect)
        quad_corners = tuple((float(x * inv_scale), float(y * inv_scale)) for x, y in box_points)

    return full_box, "grabcut_confident_bbox", diag.area_ratio, quad_corners


def _grabcut_worker_loop(task_queue: "mp.Queue", result_queue: "mp.Queue") -> None:
    """Worker process entry point: loops on tasks until killed."""
    while True:
        task = task_queue.get()
        if task is None:
            return
        small, scale, config = task
        try:
            result = _grabcut_detect_box_on_working(small, scale, config)
        except Exception as exc:  # noqa: BLE001 -- report, don't crash silently
            result_queue.put(("error", repr(exc)))
        else:
            result_queue.put(("ok", result))


class _GrabCutWorkerPool:
    """Persistent worker process for GrabCut detection; respawned on timeout or crash."""

    def __init__(self, worker_target=_grabcut_worker_loop) -> None:
        self._worker_target = worker_target
        self._process: mp.Process | None = None
        self._task_queue: "mp.Queue | None" = None
        self._result_queue: "mp.Queue | None" = None
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._process = mp.Process(
            target=self._worker_target,
            args=(self._task_queue, self._result_queue),
            daemon=True,
        )
        self._process.start()

    def _kill(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)
        self._process = None
        self._task_queue = None
        self._result_queue = None

    def run(self, task: object, timeout_s: float) -> tuple[str, object]:
        """Returns ("ok", payload), ("timeout", None), or ("error", message)."""
        with self._lock:
            self._ensure_started()
            assert self._task_queue is not None and self._result_queue is not None
            try:
                self._task_queue.put(task)
            except Exception as exc:  # noqa: BLE001 -- dead process, restart
                self._kill()
                return "error", repr(exc)
            try:
                status, payload = self._result_queue.get(timeout=timeout_s)
            except queue_module.Empty:
                self._kill()
                return "timeout", None
            if status == "error":
                self._kill()
            return status, payload

    def shutdown(self) -> None:
        with self._lock:
            if self._process is not None and self._process.is_alive() and self._task_queue is not None:
                try:
                    self._task_queue.put(None)
                    self._process.join(timeout=1.0)
                except Exception:  # noqa: BLE001 -- best-effort graceful stop
                    pass
            self._kill()


_grabcut_pool = _GrabCutWorkerPool()


def _run_grabcut_with_budget(
    image: np.ndarray, config: CropConfig
) -> tuple[tuple[int, int, int, int] | None, str, tuple[tuple[float, float], ...] | None]:
    """Runs GrabCut detection with a hard wall-clock timeout."""
    small, scale = _grabcut_prepare_working_image(image, config)
    status, payload = _grabcut_pool.run((small, scale, config), config.grabcut_time_budget_ms / 1000.0)
    if status == "ok":
        box, reason, _area_ratio, quad_corners = payload
        return box, reason, quad_corners
    if status == "timeout":
        return None, "grabcut_timeout", None
    return None, "grabcut_worker_error", None


def _grabcut_quadrilateral_candidate(
    quad_corners_full: tuple[tuple[float, float], ...],
    padded_box: tuple[int, int, int, int],
    cropped_image: np.ndarray,
) -> QuadrilateralCandidate | None:
    """Builds a QuadrilateralCandidate in the cropped image's coordinate space."""
    left, top, right, bottom = padded_box
    height, width = cropped_image.shape[:2]
    corners = np.array(
        [(x - left, y - top) for x, y in quad_corners_full], dtype=np.float64
    )
    corners[:, 0] = np.clip(corners[:, 0], 0.0, max(0.0, width - 1.0))
    corners[:, 1] = np.clip(corners[:, 1], 0.0, max(0.0, height - 1.0))
    if not np.isfinite(corners).all():
        return None
    try:
        ordered = _ordered_corners(corners)
    except ValueError:
        return None
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    support = _edge_support(ordered, edges)
    return _candidate_from_corners(ordered, cropped_image.shape, 0, support)


def _apply_grabcut_foreground(
    image: np.ndarray,
    config: CropConfig,
    transform: TransformState,
) -> CropOutcome:
    box, reason, quad_corners_full = _run_grabcut_with_budget(image, config)
    if box is None:
        # No-crop fallback by design, not a guessed center-crop.
        return CropOutcome(
            np.ascontiguousarray(image.copy()),
            transform,
            CropMetadata(),
            OperationRecord("crop", OperationStatus.SKIPPED, reason),
            True,
        )
    padded_box = _add_hybrid_padding(box, image.shape, config)
    outcome = _apply_box(image, padded_box, transform, "grabcut_foreground_detected")
    if quad_corners_full is None:
        return outcome
    candidate = _grabcut_quadrilateral_candidate(quad_corners_full, padded_box, outcome.image)
    if candidate is None:
        return outcome
    return replace(outcome, quadrilateral=candidate)


def _passes_geometry_check(
    box: tuple[float, float, float, float], img_w: int, img_h: int, config: CropConfig
) -> bool:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return False
    area_ratio = (w * h) / (img_w * img_h)
    if not (config.yolo_min_box_area_ratio <= area_ratio <= config.yolo_max_box_area_ratio):
        return False
    aspect = w / h
    return config.yolo_min_aspect_ratio <= aspect <= config.yolo_max_aspect_ratio


def _select_yolo_crop_box(
    boxes_xyxy: list[tuple[float, float, float, float]],
    confs: list[float],
    img_w: int,
    img_h: int,
    config: CropConfig,
) -> tuple[tuple[float, float, float, float] | None, str]:
    """Confidence-tiered candidate selection with union-of-distinct-boxes,
    matching the policy validated in yolo/apply_crop.py's crop-quality
    evaluation (0 CLIPPED, 0 WRONG on 163 images)."""
    accepted = []
    for box, conf in zip(boxes_xyxy, confs):
        if conf >= config.yolo_confidence_high:
            accepted.append(box)
        elif conf >= config.yolo_confidence_low and _passes_geometry_check(box, img_w, img_h, config):
            accepted.append(box)
    if not accepted:
        return None, "no_confident_detection"

    ux1 = min(b[0] for b in accepted)
    uy1 = min(b[1] for b in accepted)
    ux2 = max(b[2] for b in accepted)
    uy2 = max(b[3] for b in accepted)
    union_area_ratio = ((ux2 - ux1) * (uy2 - uy1)) / (img_w * img_h)
    if union_area_ratio >= config.yolo_union_max_area_ratio:
        return None, "union_near_full_frame"

    reason = "yolo_single_box_detected" if len(accepted) == 1 else f"yolo_{len(accepted)}_boxes_united"
    return (ux1, uy1, ux2, uy2), reason


@lru_cache(maxsize=4)
def _load_yolo_model(weights_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "yolo_label_crop_experimental() requires the optional 'yolo' extra: "
            "pip install medicine_preprocess[yolo]"
        ) from exc
    return YOLO(weights_path)


def _apply_yolo_label(
    image: np.ndarray,
    config: CropConfig,
    transform: TransformState,
) -> CropOutcome:
    model = _load_yolo_model(str(config.yolo_weights_path))
    results = model.predict(source=image, conf=config.yolo_confidence_low, verbose=False)
    result = results[0]
    height, width = image.shape[:2]
    if result.boxes is not None and len(result.boxes):
        boxes_xyxy = [tuple(b) for b in result.boxes.xyxy.cpu().numpy().tolist()]
        confs = [float(c) for c in result.boxes.conf.cpu().numpy().tolist()]
    else:
        boxes_xyxy, confs = [], []

    details = {
        "method": "yolo_label",
        "num_detections": len(boxes_xyxy),
        "confidences": [round(c, 3) for c in confs],
    }
    box, reason = _select_yolo_crop_box(boxes_xyxy, confs, width, height, config)
    if box is None:
        # No-crop fallback by design, not a guessed center-crop.
        return CropOutcome(
            np.ascontiguousarray(image.copy()),
            transform,
            CropMetadata(),
            OperationRecord("crop", OperationStatus.SKIPPED, reason, details=details),
            True,
        )
    int_box = (int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3])))
    padded_box = _add_hybrid_padding(int_box, image.shape, config)
    outcome = _apply_box(image, padded_box, transform, reason)
    record = replace(outcome.record, details={**outcome.record.details, **details})
    return replace(outcome, record=record)


def apply_crop(
    image: np.ndarray,
    config: CropConfig,
    transform: TransformState,
    *,
    experimental: bool = False,
) -> CropOutcome:
    if config.mode == "none":
        return CropOutcome(
            np.ascontiguousarray(image.copy()),
            transform,
            CropMetadata(),
            OperationRecord("crop", OperationStatus.SKIPPED, "crop_disabled"),
            False,
        )
    if experimental and config.mode == "grabcut_foreground":
        return _apply_grabcut_foreground(image, config, transform)
    if experimental and config.mode == "yolo_label":
        return _apply_yolo_label(image, config, transform)
    return CropOutcome(
        np.ascontiguousarray(image.copy()),
        transform,
        CropMetadata(),
        OperationRecord(
            "crop", OperationStatus.SKIPPED, "experimental_crop_not_enabled_in_v1"
        ),
        True,
    )


def _add_hybrid_padding(
    box: tuple[int, int, int, int], shape: tuple[int, ...], config: CropConfig
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    candidate_width, candidate_height = right - left, bottom - top
    pad_x = max(
        int(round(candidate_width * config.padding_x_fraction)),
        config.minimum_padding_pixels,
    )
    pad_y = max(
        int(round(candidate_height * config.padding_y_fraction)),
        config.minimum_padding_pixels,
    )
    return _clamped_box(
        left - pad_x,
        top - pad_y,
        right + pad_x,
        bottom + pad_y,
        shape[1],
        shape[0],
    )
