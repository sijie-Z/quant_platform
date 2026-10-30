"""Report generation must surface validity and stored warnings (R2/R3/R27).

An invalidated run (e.g. marked after a bug fix) used to produce a clean
Markdown report: the generator read neither `validity` nor `affected_by`, and
the stored `warnings` column was ignored. The cross-run comparison likewise
listed invalidated runs as plain "success runs".
"""

from __future__ import annotations

import json

import pytest
from quant_platform.lab.registry import RunStore
from quant_platform.lab.reports import factor_comparison, generate_report


def _success_run(store: RunStore, factor: str = "roe") -> str:
    run_id = store.begin_run("slice", {
        "data_source": "akshare",
        "factor": factor,
        "data_meta": {"provider": "akshare", "adjust": "hfq"},
        "universe_meta": {"provider": "test", "pit": False,
                          "bias_warning": ["survivorship_bias_possible"]},
    })
    store.finish_run(run_id, status="success",
                     evaluation={"ic_mean": 0.02, "icir": 0.4,
                                 "n_ic_obs": 300})
    return run_id


class TestGenerateReportValidity:
    def test_invalidated_run_gets_a_warning(self, tmp_path):
        store = RunStore(str(tmp_path / "reg.db"))
        run_id = _success_run(store)
        store.mark_validity(run_id, "invalidated", "BUG-99")

        path = generate_report(store.get_run(run_id), out_dir=str(tmp_path))
        text = open(path, encoding="utf-8").read()

        assert "invalidated" in text
        assert "BUG-99" in text
        assert "**Validity**: invalidated" in text

    def test_affected_by_is_shown_even_when_still_valid(self, tmp_path):
        store = RunStore(str(tmp_path / "reg.db"))
        run_id = _success_run(store)
        store.mark_validity(run_id, "valid", "BUG-07")

        path = generate_report(store.get_run(run_id), out_dir=str(tmp_path))
        text = open(path, encoding="utf-8").read()
        assert "BUG-07" in text
        assert "affected_by" in text

    def test_stored_warnings_are_rendered(self, tmp_path):
        store = RunStore(str(tmp_path / "reg.db"))
        run_id = store.begin_run("slice", {
            "data_source": "akshare", "factor": "roe",
            "universe_meta": {"provider": "test", "pit": False},
        })
        store.finish_run(run_id, status="failed", reason="RuntimeError: boom",
                         warnings=["traceback_excerpt: RuntimeError: boom"])
        # Second call without warnings must not wipe the first (R9).
        store.finish_run(run_id, status="failed", reason="RuntimeError: boom")

        path = generate_report(store.get_run(run_id), out_dir=str(tmp_path))
        text = open(path, encoding="utf-8").read()
        assert "traceback_excerpt" in text
        assert "RUN FAILED" in text

    def test_front_adjust_basis_reads_canonical_and_legacy_keys(self, tmp_path):
        store = RunStore(str(tmp_path / "reg.db"))
        for i, key in enumerate(("adjust_basis", "basis")):
            run_id = store.begin_run("slice", {
                "data_source": "akshare", "factor": "roe",
                "data_meta": {"adjust": "front", key: "per_stock_latest"},
                "universe_meta": {"provider": "test", "pit": False},
            })
            store.finish_run(run_id, status="success")
            path = generate_report(store.get_run(run_id), out_dir=str(tmp_path))
            text = open(path, encoding="utf-8").read()
            assert "per_stock_latest" in text, key


class TestFactorComparisonValidity:
    @pytest.fixture
    def registry(self, tmp_path, monkeypatch):
        # The comparison reads the module-level DEFAULT_DB and writes
        # data/reports/factor_comparison.md relative to cwd.
        monkeypatch.chdir(tmp_path)
        db = str(tmp_path / "reg.db")
        monkeypatch.setattr(factor_comparison, "DEFAULT_DB", db)
        return RunStore(db)

    def test_invalidated_success_is_flagged_not_hidden(self, registry, tmp_path):
        _success_run(registry, "valid_factor")
        bad_id = _success_run(registry, "bad_factor")
        registry.mark_validity(bad_id, "invalidated", "BUG-99")

        path = factor_comparison.generate()
        text = open(path, encoding="utf-8").read()

        assert "invalidated" in text
        assert "BUG-99" in text
        assert "Validity warnings" in text
        assert "invalidated" in text
        # Both success-status rows are shown, with the trust flag differing.
        assert "valid_factor" in text and "bad_factor" in text
        valid_row = next(
            line for line in text.splitlines() if line.startswith("| valid_factor")
        )
        assert "| yes |" in valid_row
        assert bad_id in text  # warning section cites the invalidated run id
        # Staleness is detectable: generation timestamp + row counts.
        assert "Generated at" in text
        assert "total rows" in text or "Total runs" in text

    def test_every_status_is_counted(self, registry, tmp_path):
        _success_run(registry, "f_success")
        failed_id = registry.begin_run("slice", {"data_source": "x", "factor": "f"})
        registry.finish_run(failed_id, status="failed", reason="boom")
        diag_id = registry.begin_run("slice", {"data_source": "x", "factor": "g"})
        registry.finish_run(diag_id, status="diagnostic")

        path = factor_comparison.generate()
        text = open(path, encoding="utf-8").read()
        assert "failed=1" in text
        assert "diagnostic=1" in text
        assert "success=1" in text

    def test_pit_limitation_is_derived_from_rows(self, registry, tmp_path):
        _success_run(registry, "f")
        # A row without a declared PIT must not be described as pit=false.
        run_id = registry.begin_run("slice", {"data_source": "x", "factor": "g"})
        registry.finish_run(run_id, status="success")

        path = factor_comparison.generate()
        text = open(path, encoding="utf-8").read()
        assert "All runs use `pit=false`" not in text
        assert "PIT declarations observed" in text
        assert "None" in text

    def test_run_warnings_are_surfaced(self, registry, tmp_path):
        run_id = registry.begin_run("slice", {"data_source": "x", "factor": "f"})
        registry.finish_run(run_id, status="success", warnings=["data_gap_2024"])

        path = factor_comparison.generate()
        text = open(path, encoding="utf-8").read()
        assert "data_gap_2024" in text

    def test_empty_registry_output_declares_generation_time(self, registry):
        path = factor_comparison.generate()
        text = open(path, encoding="utf-8").read()
        assert "No runs in Registry yet" in text
        assert "Generated at" in text

    def test_lineage_less_success_does_not_crash(self, registry):
        """R4: NULL factor/provider rows are kept as untrusted, and the report
        that flags them must not abort while formatting None."""
        run_id = registry.begin_run("slice", {})
        registry.finish_run(run_id, status="success")

        path = factor_comparison.generate()
        text = open(path, encoding="utf-8").read()
        assert "| ?" in text
        assert "NO" in text
        assert "0 trusted" in text

    def test_synthetic_success_is_not_counted_as_trusted(self, registry):
        run_id = registry.begin_run("slice", {
            "data_source": "synthetic", "factor": "f",
            "universe_meta": {"pit": False},
        })
        registry.finish_run(run_id, status="success")

        path = factor_comparison.generate()
        text = open(path, encoding="utf-8").read()
        assert "(0 trusted)" in text
        assert "synthetic" in text


def test_report_warnings_include_explicit_run_warnings():
    from quant_platform.lab.reports import _warnings_from_meta

    run = {
        "status": "success",
        "validity": "valid",
        "affected_by": "[]",
        "warnings": json.dumps(["something_odd"]),
        "universe_meta": json.dumps({"provider": "p", "pit": False,
                                     "bias_warning": ["b"]}),
        "data_meta": json.dumps({"adjust": "hfq"}),
    }
    warnings = _warnings_from_meta(run)
    assert "something_odd" in warnings
