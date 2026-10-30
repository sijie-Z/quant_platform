"""Factor research store: health-score sign, lineage columns, ordering.

Covers R1 (abs(ICIR) promoted wrong-sign factors), R10 (no provenance /
validity columns on any table), R25 (naive local timestamps and a
latest-row join on timestamp text) and the R11 follow-up (NULL stability is
missing evidence, not a measured zero).
"""

from __future__ import annotations

import json
import math
import sqlite3

import pytest
from quant_platform.factors.store import (
    FactorEvalRecord,
    FactorResearchStore,
    FactorStabilityRecord,
)


@pytest.fixture
def store(tmp_path):
    return FactorResearchStore(str(tmp_path / "factor_research.db"))


def _stability(name, icir, coverage=0.9, consistency=0.8):
    return FactorStabilityRecord(
        factor_name=name, ic_mean=icir / 10, ic_std=0.1, icir=icir,
        sharpe_mean=0.1, sharpe_std=0.1, sharpe_consistency=consistency,
        positive_folds=int(consistency * 10), total_folds=10, coverage=coverage,
    )


class TestHealthScoreSign:
    def test_negative_icir_ranks_below_weak_positive(self, store):
        """R1: abs(ICIR) gave ICIR=-0.5 a score of 0.91 vs 0.59 for +0.1."""
        store.save_stability(_stability("inverse", -0.5))
        store.save_stability(_stability("weak_pos", 0.1))

        ranking = store.get_factor_ranking()
        by_name = {r["factor_name"]: r for r in ranking}

        assert ranking[0]["factor_name"] == "weak_pos"
        # Independent health calculation with a sign-aware ICIR term:
        # coverage 0.9, consistency 0.8.
        assert by_name["weak_pos"]["health_score"] == pytest.approx(
            min(1.0, 0.1 / 0.5) * 0.4 + 0.9 * 0.3 + 0.8 * 0.3, abs=1e-4
        )
        assert by_name["inverse"]["health_score"] == pytest.approx(
            0.0 * 0.4 + 0.9 * 0.3 + 0.8 * 0.3, abs=1e-4
        )
        assert by_name["inverse"]["health_score"] < by_name["weak_pos"]["health_score"]

    def test_icir_score_is_capped_at_one(self, store):
        store.save_stability(_stability("strong", 2.0))
        assert store.get_factor_ranking()[0]["health_score"] == pytest.approx(
            0.4 + 0.9 * 0.3 + 0.8 * 0.3, abs=1e-4
        )


class TestLineageColumns:
    EXPECTED = {"data_source", "adjust", "pit", "validity", "warnings"}

    def test_all_data_tables_have_provenance_columns(self, store, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "factor_research.db"))
        for table in (
            "factor_values", "factor_evaluation_history",
            "factor_backtest_history", "factor_walk_forward_history",
            "factor_stability_history", "factor_regime_history",
        ):
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            assert cols >= self.EXPECTED, table
        conn.close()

    def test_evaluation_round_trips_provenance(self, store):
        record = FactorEvalRecord(
            factor_name="momentum_1m", signal_date="2024-01-02",
            rank_ic=0.02, pearson_ic=0.01, icir=0.4, coverage=0.95,
            n_assets=300, run_id="run_x", data_source="akshare",
            adjust="hfq", pit=False, validity="affected",
            warnings=["coverage_drop"],
        )
        store.save_evaluation(record)

        rows = store.get_evaluation_history("momentum_1m")
        assert len(rows) == 1
        row = rows[0]
        assert row.data_source == "akshare"
        assert row.adjust == "hfq"
        assert row.pit is False
        assert row.validity == "affected"
        assert row.warnings == ["coverage_drop"]
        assert row.to_dict()["data_source"] == "akshare"

    def test_values_round_trip_provenance(self, store, tmp_path):
        inserted = store.save_values(
            "momentum_1m",
            [{"symbol": "000001", "signal_date": "2024-01-02", "value": 0.1}],
            data_source="akshare", adjust="hfq", pit=False,
            validity="valid", warnings=["w"], run_id="run_y",
        )
        assert inserted == 1
        conn = sqlite3.connect(str(tmp_path / "factor_research.db"))
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM factor_values").fetchone())
        conn.close()
        assert row["data_source"] == "akshare"
        assert row["adjust"] == "hfq"
        assert row["pit"] == 0
        assert row["run_id"] == "run_y"
        assert json.loads(row["warnings"]) == ["w"]

    def test_old_database_is_migrated_to_full_schema(self, tmp_path):
        """A store created by an older checkout must accept lineage writes."""
        db = str(tmp_path / "old_factor.db")
        conn = sqlite3.connect(db)
        conn.execute(
            """CREATE TABLE factor_stability_history (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   factor_name TEXT NOT NULL, ic_mean REAL, ic_std REAL,
                   icir REAL, sharpe_mean REAL, sharpe_std REAL,
                   sharpe_consistency REAL, positive_folds INTEGER DEFAULT 0,
                   total_folds INTEGER DEFAULT 0, coverage REAL DEFAULT 0,
                   run_id TEXT DEFAULT '', timestamp TEXT NOT NULL)"""
        )
        conn.commit()
        conn.close()

        store = FactorResearchStore(db)
        store.save_stability(_stability("legacy_factor", 0.3))

        ranking = store.get_factor_ranking()
        assert [r["factor_name"] for r in ranking] == ["legacy_factor"]

    def test_timestamps_are_timezone_aware_utc(self, store):
        record = FactorStabilityRecord(
            factor_name="f", ic_mean=0.0, ic_std=0.0, icir=0.0,
            sharpe_mean=0.0, sharpe_std=0.0, sharpe_consistency=0.0,
            positive_folds=0, total_folds=0, coverage=0.0,
        )
        assert record.timestamp.endswith("+00:00")


class TestLatestRowSelection:
    def test_latest_row_is_selected_by_id_not_timestamp_text(self, store, tmp_path):
        """Legacy naive timestamps and new UTC ones cannot be ordered as text;
        the join must use the monotonic id."""
        db = str(tmp_path / "factor_research.db")
        store.save_stability(_stability("f", 0.2))
        conn = sqlite3.connect(db)
        # A lexicographically larger timestamp inserted later in the text
        # order but with a *higher* id would still be latest; insert it FIRST
        # with a huge timestamp and then save a new row.
        conn.execute(
            "INSERT INTO factor_stability_history "
            "(factor_name, icir, sharpe_consistency, coverage, timestamp) "
            "VALUES ('g', 9.0, 1.0, 1.0, '9999-12-31T00:00:00')"
        )
        conn.commit()
        conn.close()
        store.save_stability(_stability("g", 0.2))

        ranking = {r["factor_name"]: r for r in store.get_factor_ranking()}
        assert ranking["g"]["icir"] == pytest.approx(0.2)
        assert ranking["g"]["health_score"] < 0.9

    def test_invalidated_latest_row_is_excluded(self, store):
        store.save_stability(_stability("f", 0.3))
        invalid = _stability("f", -0.9)
        invalid.validity = "invalidated"
        store.save_stability(invalid)

        ranking = {r["factor_name"]: r for r in store.get_factor_ranking()}
        assert ranking["f"]["icir"] == pytest.approx(0.3)


class TestStabilityAvailability:
    """R11 follow-up: NULL stability must not look like a measured zero."""

    def test_null_consistency_is_unavailable_and_not_scored_as_zero(self, store):
        # A factor that WAS validated and scored consistency=0 ...
        store.save_stability(_stability("measured_zero", 0.2, consistency=0.0))
        # ... versus one whose run had no walk-forward folds (NULL).
        store.save_stability(FactorStabilityRecord(
            factor_name="no_walk_forward", ic_mean=0.02, ic_std=0.1, icir=0.2,
            sharpe_mean=float("nan"), sharpe_std=float("nan"),
            sharpe_consistency=float("nan"), positive_folds=None,
            total_folds=None, coverage=0.9,
        ))

        ranking = {r["factor_name"]: r for r in store.get_factor_ranking()}
        assert ranking["no_walk_forward"]["stability_status"] == "unavailable"
        assert ranking["no_walk_forward"]["consistency"] is None
        assert ranking["measured_zero"]["stability_status"] == "measured"
        assert ranking["measured_zero"]["consistency"] == 0.0

        # Missing stability is excluded and the measured terms renormalized
        # (ICIR 40% + coverage 30% -> /0.7) instead of counting as a zero.
        no_wf_health = (0.2 / 0.5 * 0.4 + 0.9 * 0.3) / 0.7
        assert ranking["no_walk_forward"]["health_score"] == pytest.approx(
            no_wf_health, abs=1e-4
        )
        measured_health = 0.2 / 0.5 * 0.4 + 0.9 * 0.3 + 0.0 * 0.3
        assert ranking["measured_zero"]["health_score"] == pytest.approx(
            measured_health, abs=1e-4
        )
        assert (ranking["no_walk_forward"]["health_score"]
                > ranking["measured_zero"]["health_score"])

    def test_null_evaluation_metrics_read_back_as_nan(self, store, tmp_path):
        db = str(tmp_path / "factor_research.db")
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO factor_evaluation_history "
            "(factor_name, signal_date, rank_ic, pearson_ic, icir, coverage, "
            " n_assets, run_id, timestamp) "
            "VALUES ('unmeasured', '2024-01-02', NULL, NULL, NULL, NULL, 0, "
            "        'run_x', '2024-01-02T00:00:00')"
        )
        conn.commit()
        conn.close()

        row = store.get_evaluation_history("unmeasured")[0]
        # The DB stores NULL; the reader must not turn it into measured 0.0.
        assert math.isnan(row.rank_ic)
        assert math.isnan(row.icir)
        assert math.isnan(row.coverage)
