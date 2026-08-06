from __future__ import annotations

import numpy as np

from medicine_preprocess.config import CropConfig
from medicine_preprocess.crop import (
    _diagnose_grabcut_mask,
    _grabcut_early_reject_reason,
    _grabcut_is_clearly_confident,
)

SW, SH = 200, 160
MARGIN = 5
RECT = (MARGIN, MARGIN, SW - 2 * MARGIN, SH - 2 * MARGIN)


def _mask_with_rect(box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((SH, SW), dtype=np.uint8)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = 1
    return mask


def test_diagnose_clean_centered_rectangle() -> None:
    mask = _mask_with_rect((55, 45, 145, 115))
    diag = _diagnose_grabcut_mask(mask, RECT, MARGIN, SW, SH)
    assert diag.has_contour is True
    assert diag.edge_touch_count == 0
    # cv2.contourArea (polygon shoelace) slightly underestimates a
    # rasterized single-blob mask's pixel count -- not exactly 1.0.
    assert diag.dominant_share > 0.95
    assert 0.0 < diag.area_ratio < 1.0


def test_diagnose_empty_mask_has_no_contour() -> None:
    mask = np.zeros((SH, SW), dtype=np.uint8)
    diag = _diagnose_grabcut_mask(mask, RECT, MARGIN, SW, SH)
    assert diag.has_contour is False


def test_early_reject_no_contour() -> None:
    mask = np.zeros((SH, SW), dtype=np.uint8)
    diag = _diagnose_grabcut_mask(mask, RECT, MARGIN, SW, SH)
    assert _grabcut_early_reject_reason(diag, CropConfig()) == "no_foreground_contour"


def test_early_reject_touches_three_edges() -> None:
    # Spans nearly the full width and from the top edge down to mid-height:
    # touches left, right, and top -- three of four edges.
    mask = _mask_with_rect((0, 0, SW, SH // 2))
    diag = _diagnose_grabcut_mask(mask, RECT, MARGIN, SW, SH)
    assert diag.edge_touch_count >= 3
    assert _grabcut_early_reject_reason(diag, CropConfig()) == "bbox_touches_all_edges"


def test_early_reject_no_dominant_component() -> None:
    # Three equal-sized, well-separated components: each individually clears
    # the area-ratio band, but none dominates the total foreground.
    mask = np.zeros((SH, SW), dtype=np.uint8)
    mask[20:75, 20:75] = 1       # 55x55
    mask[20:75, 100:155] = 1     # 55x55
    mask[90:145, 20:75] = 1      # 55x55
    diag = _diagnose_grabcut_mask(mask, RECT, MARGIN, SW, SH)
    assert diag.dominant_share < 0.5
    assert _grabcut_early_reject_reason(diag, CropConfig()) == "no_dominant_component"


def test_early_reject_extremely_narrow_bbox() -> None:
    mask = _mask_with_rect((89, 10, 111, 150))  # 22px wide, 140px tall
    diag = _diagnose_grabcut_mask(mask, RECT, MARGIN, SW, SH)
    assert diag.aspect_ratio > 6.0
    assert _grabcut_early_reject_reason(diag, CropConfig()) == "bbox_extremely_narrow"


def test_early_reject_mask_matches_init_rectangle() -> None:
    # A GrabCut init rect that doesn't itself touch the image edges (unlike
    # the module-level RECT), so this isolates the near-init-rect signal
    # from the edge-touch signal, which would otherwise fire first.
    inner_rect = (40, 30, 120, 100)
    rx, ry, rw, rh = inner_rect
    mask = _mask_with_rect((rx, ry, rx + rw, ry + rh))
    diag = _diagnose_grabcut_mask(mask, inner_rect, MARGIN, SW, SH)
    assert diag.edge_touch_count == 0
    assert diag.near_init_rect_ratio > 0.97
    assert _grabcut_early_reject_reason(diag, CropConfig()) == "mask_matches_init_rectangle"


def test_clean_centered_rectangle_is_not_early_rejected() -> None:
    mask = _mask_with_rect((55, 45, 145, 115))
    diag = _diagnose_grabcut_mask(mask, RECT, MARGIN, SW, SH)
    assert _grabcut_early_reject_reason(diag, CropConfig()) is None


def test_clearly_confident_requires_comfortable_margins() -> None:
    config = CropConfig()
    confident_mask = _mask_with_rect((55, 45, 145, 115))
    confident_diag = _diagnose_grabcut_mask(confident_mask, RECT, MARGIN, SW, SH)
    assert confident_diag.area_ratio > config.grabcut_min_area_ratio + 0.05
    assert _grabcut_is_clearly_confident(confident_diag, config) is True

    # Area ratio just barely inside the accept range -- ambiguous, not
    # "clearly" confident, so a second iteration should still be considered.
    tiny_mask = _mask_with_rect((90, 70, 110, 90))
    tiny_diag = _diagnose_grabcut_mask(tiny_mask, RECT, MARGIN, SW, SH)
    assert tiny_diag.area_ratio < config.grabcut_min_area_ratio + 0.05
    assert _grabcut_is_clearly_confident(tiny_diag, config) is False
