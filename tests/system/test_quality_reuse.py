from __future__ import annotations

from dataclasses import replace

import numpy as np

from medicine_preprocess import PreprocessConfig, preprocess_image


def test_capped_quality_before_is_never_reused_as_working_quality() -> None:
    # grabcut() sets pre_crop_analysis_max_long_side=1024, so quality_before
    # is measured on a downscaled preview even when crop and resize are both
    # no-ops and the real working image is untouched -- that preview must
    # never stand in for working_quality/quality_after, since it was never
    # calibrated against thresholds at this resolution.
    base = PreprocessConfig.grabcut()
    assert base.quality.pre_crop_analysis_max_long_side == 1024
    config = replace(
        base,
        crop=replace(base.crop, mode="none"),
        geometry=replace(base.geometry, deskew_enabled=False, perspective_enabled=False),
        resize=replace(base.resize, mode="none", pre_enhancement_max_long_side=None),
    )
    image = np.full((1200, 1600, 3), 120, dtype=np.uint8)

    result = preprocess_image(image, config, source_id="quality-no-reuse-when-capped")

    assert result.quality_before is not None and result.quality_after is not None
    assert result.quality_before.classifications is None
    assert result.quality_after.classifications is not None
    # The real working image kept its full 1600px width; the quality_before
    # preview was capped to a 1024px long side -- proof it measured a
    # downscaled copy, not the real (unchanged) working image.
    assert result.quality_before.measurements.width == 1024
    assert result.quality_after.measurements.width == 1600
