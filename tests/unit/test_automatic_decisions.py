from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from medicine_preprocess.enhancement import (
    AutomaticDecision,
    apply_gray_world_white_balance,
    apply_percentile_exposure,
    automatic_gamma,
    choose_operations,
)
from medicine_preprocess.quality import (
    QualityClassifications,
    QualityReport,
    QualityThresholds,
    analyze_image_quality,
)


THRESHOLDS = QualityThresholds(40, 210, 12, 8, 5, 25)


def _measure(level: int = 80):
    return analyze_image_quality(np.full((64, 64, 3), level, dtype=np.uint8)).measurements


def _report(
    *,
    level: int = 80,
    exposure: str = "normal",
    contrast: str = "normal",
    noise: str = "normal",
    sharpness: str = "sharp",
    **measurement_changes: float,
) -> QualityReport:
    measurements = replace(_measure(level), **measurement_changes)
    classifications = QualityClassifications(exposure, contrast, noise, sharpness)
    return QualityReport(measurements, classifications)


def test_dark_low_contrast_clean_soft_image_selects_expected_operations() -> None:
    report = QualityReport(
        _measure(30),
        QualityClassifications("dark", "low", "normal", "slightly_soft"),
    )
    decision = choose_operations(
        report,
        THRESHOLDS,
        glare_warning=False,
        color_cast_strong=False,
        short_side=64,
        minimum_short_side=960,
    )
    assert decision == AutomaticDecision(
        white_balance=False,
        gamma=True,
        percentile_exposure=False,
        clahe=True,
        denoise=False,
        sharpen=True,
        resize=True,
    )


def test_unusably_blurred_or_high_noise_image_is_not_sharpened() -> None:
    report = QualityReport(
        _measure(100),
        QualityClassifications("normal", "normal", "high", "unusably_blurred"),
    )
    decision = choose_operations(
        report,
        THRESHOLDS,
        glare_warning=False,
        color_cast_strong=False,
        short_side=1200,
        minimum_short_side=960,
    )
    assert decision.denoise
    assert not decision.sharpen
    assert not decision.resize


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (
            # "bright" never triggers automatic gamma -- whole-frame median
            # luminance can't distinguish a naturally bright/light-colored
            # subject from a genuinely overexposed photo.
            _report(exposure="bright"),
            AutomaticDecision(False, False, False, False, False, False, False),
        ),
        (
            _report(
                exposure="dark",
                luminance_p50=20.0,
            ),
            AutomaticDecision(False, False, True, False, False, False, False),
        ),
        (
            _report(
                exposure="dark",
                luminance_p50=20.0,
                clipped_highlight_fraction=0.200001,
            ),
            AutomaticDecision(False, False, False, False, False, False, False),
        ),
        (
            _report(
                contrast="low",
            ),
            AutomaticDecision(False, False, False, True, False, False, False),
        ),
        (
            # High noise + not "sharp" (no impulse confirmed) -> mild
            # bilateral, not the more destructive median filter.
            _report(
                noise="high",
                sharpness="slightly_soft",
            ),
            AutomaticDecision(False, False, False, False, True, False, False, denoise_mode="bilateral"),
        ),
        (
            _report(),
            AutomaticDecision(False, False, False, False, False, False, False),
        ),
    ],
)
def test_decision_table_rows(report: QualityReport, expected: AutomaticDecision) -> None:
    decision = choose_operations(
        report,
        THRESHOLDS,
        glare_warning=False,
        color_cast_strong=False,
        short_side=1200,
        minimum_short_side=960,
    )
    assert decision == expected


def test_glare_blocks_white_balance_and_clahe() -> None:
    report = _report(contrast="low", glare_fraction=0.25)
    decision = choose_operations(
        report,
        THRESHOLDS,
        glare_warning=True,
        color_cast_strong=True,
        short_side=1200,
        minimum_short_side=960,
    )
    assert not decision.white_balance
    assert not decision.clahe


def test_short_side_at_minimum_skips_resize_and_clipping_boundary_is_inclusive() -> None:
    report = _report(
        exposure="dark",
        luminance_p50=20.0,
        clipped_highlight_fraction=0.20,
    )
    decision = choose_operations(
        report,
        THRESHOLDS,
        glare_warning=False,
        color_cast_strong=False,
        short_side=960,
        minimum_short_side=960,
    )
    assert decision.percentile_exposure
    assert not decision.gamma
    assert not decision.resize


def test_color_cast_selects_white_balance() -> None:
    report = _report()
    decision = choose_operations(
        report,
        THRESHOLDS,
        glare_warning=False,
        color_cast_strong=True,
        short_side=1200,
        minimum_short_side=960,
    )
    assert decision.white_balance


def test_automatic_decisions_require_classifications() -> None:
    report = QualityReport(_measure(), None)
    with pytest.raises(ValueError, match="classifications"):
        choose_operations(
            report,
            THRESHOLDS,
            glare_warning=False,
            color_cast_strong=False,
            short_side=64,
            minimum_short_side=960,
        )


def test_gray_world_white_balance_balances_channels_without_mutating_input() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, 0] = 40
    image[:, :, 1] = 100
    image[:, :, 2] = 180
    before = image.copy()

    result = apply_gray_world_white_balance(image)

    assert np.array_equal(image, before)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    assert float(np.ptp(result.reshape(-1, 3).mean(axis=0))) < float(
        np.ptp(image.reshape(-1, 3).mean(axis=0))
    )


def test_percentile_exposure_stretches_luminance_without_mutating_input() -> None:
    ramp = np.tile(np.arange(40, 120, dtype=np.uint8), (64, 1))
    image = np.dstack((ramp, ramp, ramp))
    before = image.copy()

    result = apply_percentile_exposure(image)

    assert np.array_equal(image, before)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    assert int(result.min()) < int(image.min())
    assert int(result.max()) > int(image.max())


def test_percentile_exposure_rejects_flat_luminance() -> None:
    with pytest.raises(ValueError, match="percentile span"):
        apply_percentile_exposure(np.full((8, 8, 3), 100, dtype=np.uint8))


def test_automatic_gamma_uses_bounded_exposure_ratio() -> None:
    # Near-middle-gray (128) rounds up to the 1.0 floor: it's not dark
    # enough to warrant brightening, and darkening is never allowed.
    assert automatic_gamma(128.0) == 1.0
    assert automatic_gamma(0.0) == 1.55
    # A fully bright median floors at 1.0 (no-op), never darkens below it.
    assert automatic_gamma(255.0) == 1.0
    # A moderately dark median (100, unclamped by either bound) still
    # brightens by the real formula.
    assert automatic_gamma(100.0) == pytest.approx(np.log(100.0 / 255.0) / np.log(0.5))


def test_high_noise_sharp_image_skips_denoise_as_text_like_residual() -> None:
    # noise=high + sharpness=sharp, no confirmed impulse noise, is exactly
    # the signature of fine text being misread as noise -- skip denoising
    # rather than blur text that was never actually noisy.
    report = _report(noise="high", sharpness="sharp")
    decision = choose_operations(
        report,
        THRESHOLDS,
        glare_warning=False,
        color_cast_strong=False,
        short_side=1200,
        minimum_short_side=960,
    )
    assert decision.denoise is False
    assert decision.denoise_mode == "off"


def test_confirmed_impulse_noise_overrides_text_like_skip_to_median() -> None:
    # Even with sharpness=sharp, a confirmed high impulse-noise fraction
    # (real salt-and-pepper noise) should still use the median filter.
    report = _report(noise="high", sharpness="sharp", impulse_noise_fraction=0.05)
    decision = choose_operations(
        report,
        THRESHOLDS,
        glare_warning=False,
        color_cast_strong=False,
        short_side=1200,
        minimum_short_side=960,
    )
    assert decision.denoise is True
    assert decision.denoise_mode == "median"


def test_high_noise_not_sharp_without_impulse_uses_mild_bilateral() -> None:
    report = _report(noise="high", sharpness="unusably_blurred")
    decision = choose_operations(
        report,
        THRESHOLDS,
        glare_warning=False,
        color_cast_strong=False,
        short_side=1200,
        minimum_short_side=960,
    )
    assert decision.denoise is True
    assert decision.denoise_mode == "bilateral"
