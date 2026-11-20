"""Regression tests for the research-validation factor-store writer (R11).

The writer used to stamp every evaluation row with the configured run end
date (a constant) and persist stability statistics as zeros, so the factor
store was not a time series and 30% of the health score (the consistency
term) was structurally zero. These tests pin the corrected behaviour:

* signal dates come from the evaluated panels (distinct panels -> distinct
  dates, and the run id carries the actual date);
* stability statistics are computed from real walk-forward fold metrics;
* a factor with no walk-forward data is stored as NULL statistics plus a
  machine-readable ``stability_unavailable`` warning, never as measured
  zeros;
* R10's provenance columns are populated on every row;
* legacy rows without provenance remain readable alongside new rows.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from quant_platform.factors.store import FactorResearchStore
from quant_platform.strategy.research_validation import (
    persist_validation_results,
    signal_date_for_panel,
    stability_from_fold_metrics,
)

DATES_A = pd.bdate_range("2024-01-02", periods=40)
DATES_B = pd.bdate_range("2024-01-02", "2024-06-28")


def _panel(index: pd.DatetimeIndex, last_valid: str | None = None) -> pd.DataFrame:
    """A factor panel with values on every row up to ``last_valid``."""
    index = pd.DatetimeIndex(index)
    panel = pd.DataFrame(1.0, index=index, columns=[f"S{i}" for i in range(5)])
    if last_valid is not None:
        panel.loc[panel.index > pd.Timestamp(last_valid)] = np.nan
    return panel


def _ic_results(**overrides) -> dict:
    base = {
        "mean_ic": 0.03,
        "std_ic": 0.1,
        "icir": 0.3,
        "pearson_ic": 0.02,
        "coverage": 0.95,
        "n_assets": 5,
    }
    base.update(overrides)
    return base


def _store(tmp_path) -> FactorResearchStore:
    return FactorResearchStore(str(tmp_path / "factor_research.db"))


def _stability_rows(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM factor_stability_history ORDER BY id"
    )]
    conn.close()
    return rows


class TestSignalDates:
    def test_signal_date_comes_from_the_panel_not_the_run_end(self, tmp_path):
        store = _store(tmp_path)
        panels = {
            "alpha": _panel(DATES_A, last_valid="2024-02-09"),
            "beta": _panel(DATES_B, last_valid="2024-06-14"),
        }
        result = persist_validation_results(
            {name: _ic_results() for name in panels}, panels, store=store,
        )

        assert result["records_saved"] == 2
        assert result["signal_dates"] == {
            "alpha": "2024-02-09", "beta": "2024-06-14",
        }

        conn = sqlite3.connect(str(tmp_path / "factor_research.db"))
        conn.row_factory = sqlite3.Row
        rows = {
            r["factor_name"]: dict(r) for r in conn.execute(
                "SELECT * FROM factor_evaluation_history"
            )
        }
        conn.close()
        # Two distinct dates: the store is now a time series, not a constant.
        assert rows["alpha"]["signal_date"] == "2024-02-09"
        assert rows["beta"]["signal_date"] == "2024-06-14"
        assert rows["alpha"]["run_id"] == "research_validation_2024-02-09"
        assert rows["beta"]["run_id"] == "research_validation_2024-06-14"

    def test_panel_without_observations_is_skipped_not_dated(self, tmp_path):
        store = _store(tmp_path)
        empty = pd.DataFrame(index=DATES_A, columns=["S0"], dtype=float)
        result = persist_validation_results(
            {"dead": _ic_results()}, {"dead": empty}, store=store,
        )
        assert result["records_saved"] == 0
        assert result["stability_status"] == {}
        assert "dead" in result["skipped"][0]
        assert signal_date_for_panel(None) is None


class TestStabilityFromFolds:
    def test_real_fold_metrics_produce_nonzero_statistics(self, tmp_path):
        store = _store(tmp_path)
        wf_results = {
            "alpha": {
                "fold_metrics": [
                    {"sharpe_ratio": 0.5},
                    {"sharpe_ratio": -0.2},
                    {"sharpe_ratio": 0.8},
                ]
            }
        }
        result = persist_validation_results(
            {"alpha": _ic_results()}, {"alpha": _panel(DATES_A)},
            wf_results, store=store,
        )
        assert result["stability_status"] == {"alpha": "computed"}

        rows = _stability_rows(tmp_path / "factor_research.db")
        assert len(rows) == 1
        row = rows[0]
        arr = np.array([0.5, -0.2, 0.8])
        assert row["sharpe_mean"] == pytest.approx(arr.mean())
        assert row["sharpe_std"] == pytest.approx(arr.std())
        assert row["sharpe_consistency"] == pytest.approx(2 / 3)
        assert row["positive_folds"] == 2
        assert row["total_folds"] == 3
        # The old writer stored zeros here, making the consistency term of the
        # health score structurally zero.
        assert row["sharpe_consistency"] > 0
        assert row["total_folds"] > 0

    def test_no_walk_forward_data_is_null_and_flagged(self, tmp_path):
        store = _store(tmp_path)
        persist_validation_results(
            {"alpha": _ic_results()}, {"alpha": _panel(DATES_A)},
            None, store=store,
        )
        row = _stability_rows(tmp_path / "factor_research.db")[0]

        assert row["sharpe_mean"] is None
        assert row["sharpe_std"] is None
        assert row["sharpe_consistency"] is None
        assert row["positive_folds"] is None
        assert row["total_folds"] is None
        warnings = json.loads(row["warnings"])
        assert any(w.startswith("stability_unavailable") for w in warnings)

    def test_non_finite_and_missing_fold_sharpes_are_skipped(self):
        stability = stability_from_fold_metrics([
            {"sharpe_ratio": 0.4},
            {"sharpe_ratio": float("nan")},
            {"sharpe_ratio": None},
            {},
            {"sharpe_ratio": -0.4},
        ])
        assert stability["status"] == "computed"
        assert stability["total_folds"] == 2
        assert stability["positive_folds"] == 1
        assert stability["n_skipped_folds"] == 3

        unavailable = stability_from_fold_metrics([{"sharpe_ratio": np.nan}])
        assert unavailable["status"] == "unavailable"
        assert unavailable["positive_folds"] is None
        assert np.isnan(unavailable["sharpe_mean"])


class TestProvenance:
    def test_evaluation_and_stability_rows_carry_lineage(self, tmp_path):
        store = _store(tmp_path)
        persist_validation_results(
            {"alpha": _ic_results()}, {"alpha": _panel(DATES_A)},
            {"alpha": {"fold_metrics": [{"sharpe_ratio": 0.1}]}},
            store=store, data_source="synthetic", adjust="qfq", pit=False,
            validity="valid", warnings=["synthetic_data_not_market_data"],
        )

        conn = sqlite3.connect(str(tmp_path / "factor_research.db"))
        conn.row_factory = sqlite3.Row
        evaluation = dict(conn.execute(
            "SELECT * FROM factor_evaluation_history"
        ).fetchone())
        stability = dict(conn.execute(
            "SELECT * FROM factor_stability_history"
        ).fetchone())
        conn.close()

        for row in (evaluation, stability):
            assert row["data_source"] == "synthetic"
            assert row["adjust"] == "qfq"
            assert row["pit"] == 0
            assert row["validity"] == "valid"
            assert "synthetic_data_not_market_data" in json.loads(row["warnings"])

    def test_nan_metrics_are_stored_as_null_not_zero(self, tmp_path):
        store = _store(tmp_path)
        persist_validation_results(
            {"alpha": _ic_results(mean_ic=float("nan"), icir=float("nan"))},
            {"alpha": _panel(DATES_A)}, store=store,
        )
        conn = sqlite3.connect(str(tmp_path / "factor_research.db"))
        row = conn.execute(
            "SELECT rank_ic, icir FROM factor_evaluation_history"
        ).fetchone()
        conn.close()
        assert row[0] is None and row[1] is None

    def test_old_rows_stay_readable_next_to_new_ones(self, tmp_path):
        store = _store(tmp_path)
        # A row written by an older checkout: no provenance columns filled.
        conn = sqlite3.connect(str(tmp_path / "factor_research.db"))
        conn.execute(
            "INSERT INTO factor_evaluation_history "
            "(factor_name, signal_date, rank_ic, pearson_ic, icir, coverage, "
            " n_assets, run_id, timestamp) "
            "VALUES ('legacy', '2020-01-02', 0.01, 0.0, 0.1, 0.9, 10, "
            "        'old_run', '2020-01-02T00:00:00')"
        )
        conn.commit()
        conn.close()

        persist_validation_results(
            {"alpha": _ic_results()}, {"alpha": _panel(DATES_A)}, store=store,
        )

        history = store.get_evaluation_history("legacy")
        assert len(history) == 1
        assert history[0].signal_date == "2020-01-02"
        assert history[0].data_source == ""  # NULL round-trips as empty
        assert len(store.get_evaluation_history("alpha")) == 1


class TestValidatorPlumbing:
    """The production step must forward the walk-forward fold metrics."""

    def test_save_step_persists_computed_stability(self, tmp_path):
        from quant_platform.strategy.research_validation import ResearchValidator

        validator = ResearchValidator(
            factor_store_path=str(tmp_path / "factor_research.db"),
        )
        step = validator._step_save_to_store(
            {"alpha": _ic_results()},
            {"alpha": _panel(DATES_A)},
            {"alpha": {"fold_metrics": [
                {"sharpe_ratio": 0.4},
                {"sharpe_ratio": -0.2},
                {"sharpe_ratio": 0.6},
            ]}},
            data_source="synthetic", adjust="qfq", pit=False,
            warnings=["synthetic_data_not_market_data"],
        )

        assert step.status == "completed"
        assert step.details["records_saved"] == 1
        assert step.details["stability_status"] == {"alpha": "computed"}
        row = _stability_rows(tmp_path / "factor_research.db")[0]
        assert row["sharpe_consistency"] == pytest.approx(2 / 3)
        assert row["positive_folds"] == 2
        assert row["total_folds"] == 3

    def test_save_step_without_folds_marks_unavailable(self, tmp_path):
        from quant_platform.strategy.research_validation import ResearchValidator

        validator = ResearchValidator(
            factor_store_path=str(tmp_path / "factor_research.db"),
        )
        step = validator._step_save_to_store(
            {"alpha": _ic_results()}, {"alpha": _panel(DATES_A)},
            {}, data_source="synthetic",
        )

        assert step.status == "completed"
        assert step.details["stability_status"] == {"alpha": "unavailable"}
        row = _stability_rows(tmp_path / "factor_research.db")[0]
        assert row["sharpe_consistency"] is None
        assert row["positive_folds"] is None
        assert row["total_folds"] is None
