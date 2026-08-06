from __future__ import annotations

import time

import cv2
import numpy as np

from medicine_preprocess import crop as crop_module
from medicine_preprocess.config import CropConfig
from medicine_preprocess.crop import apply_crop
from medicine_preprocess.geometry import TransformState
from medicine_preprocess.result import OperationStatus


def _distinct_foreground_image() -> np.ndarray:
    image = np.full((200, 300, 3), 200, dtype=np.uint8)
    image[50:150, 60:240] = (10, 10, 200)
    return image


def _rotated_foreground_image(angle_degrees: float = 15.0) -> np.ndarray:
    image = np.full((240, 320, 3), 200, dtype=np.uint8)
    rect = ((160.0, 120.0), (150.0, 90.0), angle_degrees)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(image, box, (10, 10, 200))
    return image


def _slow_worker_loop(task_queue, result_queue) -> None:
    """Module-level (picklable) stand-in for the real worker loop, used to
    simulate a hung GrabCut call. Must live at module scope: multiprocessing
    spawn sends this by qualified import path, and a function nested inside
    a test function has no such path."""
    while True:
        task = task_queue.get()
        if task is None:
            return
        time.sleep(2.0)
        result_queue.put(("ok", ((0, 0, 10, 10), "grabcut_confident_bbox", 1.0, None)))


def test_grabcut_foreground_crops_a_confident_high_contrast_region() -> None:
    image = _distinct_foreground_image()
    outcome = apply_crop(
        image,
        CropConfig(mode="grabcut_foreground"),
        TransformState(np.eye(3)),
        experimental=True,
    )
    assert outcome.record.status is OperationStatus.APPLIED
    assert outcome.record.reason == "grabcut_foreground_detected"
    left, top, right, bottom = outcome.metadata.crop_box_working
    assert 0 <= left < 60
    assert 240 < right <= 300
    assert 0 <= top < 50
    assert 150 < bottom <= 200
    assert outcome.image.shape[0] < image.shape[0]
    assert outcome.image.shape[1] < image.shape[1]
    assert outcome.fallback_used is False
    # A near-perfect axis-aligned rectangle should also clear the extent
    # gate and get a quadrilateral candidate for the perspective stage.
    assert outcome.quadrilateral is not None
    assert outcome.quadrilateral.rectangularity > 0.9


def test_grabcut_foreground_populates_quadrilateral_for_tilted_rectangle() -> None:
    image = _rotated_foreground_image(15.0)
    outcome = apply_crop(
        image,
        CropConfig(mode="grabcut_foreground"),
        TransformState(np.eye(3)),
        experimental=True,
    )
    assert outcome.record.status is OperationStatus.APPLIED
    assert outcome.quadrilateral is not None
    assert outcome.quadrilateral.rectangularity > 0.9
    corners = np.asarray(outcome.quadrilateral.corners, dtype=np.float64)
    # A genuinely tilted rectangle's own area is smaller than its
    # axis-aligned bounding box (that gap is exactly zero for an
    # axis-aligned rectangle) -- confirms the quad captures real tilt,
    # not just a repackaged axis-aligned box.
    xs, ys = corners[:, 0], corners[:, 1]
    axis_aligned_area = (xs.max() - xs.min()) * (ys.max() - ys.min())
    polygon_area = float(cv2.contourArea(corners.astype(np.float32)))
    assert polygon_area / axis_aligned_area < 0.95


def test_grabcut_foreground_falls_back_to_untouched_frame_not_center_crop() -> None:
    blank = np.full((100, 160, 3), 128, dtype=np.uint8)
    outcome = apply_crop(
        blank,
        CropConfig(mode="grabcut_foreground"),
        TransformState(np.eye(3)),
        experimental=True,
    )
    assert outcome.record.status is OperationStatus.SKIPPED
    assert outcome.fallback_used is True
    assert outcome.metadata.crop_box_working is None
    # The point of the no-crop fallback: an off-center object survives
    # untouched instead of being clipped by a fixed center-fraction crop.
    assert np.array_equal(outcome.image, blank)


def test_grabcut_foreground_timeout_falls_back_within_budget() -> None:
    # The detection now runs in a separate OS process (see _GrabCutWorkerPool
    # in crop.py), so monkeypatching a function in this test process's copy
    # of crop_module cannot make the worker's own detection call slow -- a
    # spawned child re-imports the module fresh and never sees this
    # process's monkeypatches. Instead this simulates timeout at the level
    # that's actually process-boundary-safe: swap in a slow *worker loop*
    # target (_slow_worker_loop, module-level so it can be pickled to the
    # child at process-start).
    pool = crop_module._GrabCutWorkerPool(worker_target=_slow_worker_loop)
    try:
        started = time.perf_counter()
        status, payload = pool.run(task=("dummy_small", 1.0, CropConfig(mode="grabcut_foreground")), timeout_s=0.2)
        elapsed = time.perf_counter() - started

        assert status == "timeout"
        assert payload is None
        assert elapsed < 1.0  # well under the 2s sleep; bounded by the 200ms budget

        # The hard-killed worker must actually be gone, not just abandoned:
        # a fresh call spins up a brand new process rather than reusing a
        # zombie/stuck one.
        assert pool._process is None
    finally:
        pool.shutdown()


def test_grabcut_foreground_applies_end_to_end_through_process_isolated_worker() -> None:
    # Integration check that the real pipeline wiring -- prepare working
    # image in the parent, pickle (small, scale, config) across the
    # process boundary, run real GrabCut in the child, return a real box --
    # still produces the same outcome as before process isolation was
    # introduced, with a generous budget so this doesn't race real timing.
    image = _distinct_foreground_image()
    config = CropConfig(mode="grabcut_foreground", grabcut_time_budget_ms=5000.0)
    outcome = apply_crop(image, config, TransformState(np.eye(3)), experimental=True)

    assert outcome.record.status is OperationStatus.APPLIED
    assert outcome.record.reason == "grabcut_foreground_detected"
    assert outcome.fallback_used is False
