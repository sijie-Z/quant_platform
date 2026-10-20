"""Exact ICIR convention: mean IC over *sample* std (ddof=1).

``ic_summary`` reports ``mean / std``. pandas' default ``Series.std`` is the
sample standard deviation (ddof=1); the audit mutant switched it to ddof=0,
a 2.6% change at 20 observations and 5.4% at the 10-observation minimum,
which can flip the ICIR gate at its 0.2 threshold.
"""

import numpy as np
import pandas as pd
import pytest
from quant_platform.factors.evaluation import MIN_IC_OBSERVATIONS, ic_summary


def _series(values) -> pd.Series:
    return pd.Series(
        np.asarray(values, dtype=float),
        index=pd.bdate_range("2024-01-02", periods=len(values)),
    )


def test_icir_is_mean_over_sample_std_on_twenty_points():
    values = np.linspace(0.01, 0.10, 20)
    summary = ic_summary(_series(values))

    sample_std = float(np.std(values, ddof=1))
    assert summary["n_obs"] == 20
    assert summary["mean_ic"] == pytest.approx(float(np.mean(values)), rel=1e-12)
    assert summary["std_ic"] == pytest.approx(sample_std, rel=1e-12)
    assert summary["icir"] == pytest.approx(
        float(np.mean(values)) / sample_std, rel=1e-12,
    )

    # The population-std convention is a measurably different number; the
    # assertion above must not admit both.
    population_icir = float(np.mean(values)) / float(np.std(values, ddof=0))
    assert abs(summary["icir"] - population_icir) > 0.01


def test_icir_on_a_mixed_sign_series_uses_the_same_denominator():
    values = np.array([0.04, -0.02, 0.03, -0.01, 0.05, 0.01, -0.03, 0.02,
                       0.06, -0.02, 0.01, 0.04, -0.01, 0.03, 0.02])
    summary = ic_summary(_series(values))

    sample_std = float(np.std(values, ddof=1))
    assert summary["std_ic"] == pytest.approx(sample_std, rel=1e-12)
    assert summary["icir"] == pytest.approx(
        float(np.mean(values)) / sample_std, rel=1e-12,
    )


class TestMinimumObservationCount:
    def test_the_minimum_is_ten(self):
        assert MIN_IC_OBSERVATIONS == 10

    def test_nine_observations_report_nan_statistics(self):
        summary = ic_summary(_series(np.linspace(0.01, 0.09, 9)))

        assert summary["n_obs"] == 9
        assert np.isnan(summary["mean_ic"])
        assert np.isnan(summary["std_ic"])
        assert np.isnan(summary["icir"])
        assert np.isnan(summary["ic_positive_ratio"])

    def test_ten_observations_report_finite_statistics(self):
        summary = ic_summary(_series(np.linspace(0.01, 0.10, 10)))

        assert summary["n_obs"] == 10
        assert np.isfinite(summary["mean_ic"])
        assert np.isfinite(summary["std_ic"])
        assert np.isfinite(summary["icir"])

    def test_nan_observations_do_not_count_toward_the_minimum(self):
        values = np.array([0.01] * 9 + [np.nan] * 50)

        summary = ic_summary(_series(values))

        assert summary["n_obs"] == 9
        assert np.isnan(summary["icir"])
