"""End-to-end wiring tests for the seven previously-unavailable endpoints.

The contract under test:

* ``POST /api/walkforward``, ``POST /api/risk/decompose``,
  ``GET /api/analysis/ic-decay``, ``GET /api/analysis/correlation``,
  ``GET /api/monitor/tca-summary``, ``GET /api/monitor/factor-status`` and
  ``GET /api/monitor/capacity-gauge`` return real computations on the
  research context / store, with ``available: false`` and a precise reason
  when an input is genuinely missing.
* The context is bounded (analytics executor; 429 + Retry-After under
  saturation), TTL-cached, and keyed by config + provider + date range.
* Nothing is fabricated: the IC vector is bit-for-bit the value
  ``factors.evaluation.ic_decay`` produces on the same panels, and the risk
  shares are Euler variance shares that sum to one.

Runtime note: the default YAML is used unchanged except for
``universe.n_stocks`` (150 instead of 500) so the suite stays fast; every
data/factor/alpha/backtest code path is the production one.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import threading
import time
import warnings

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.api import monitor, routes
from quant_platform.api import research_context as rc
from quant_platform.core.store import Store
from tests.test_api.research_helpers import make_context, small_config

START = "2024-01-01"
END = "2025-06-30"
WINDOW = {"start_date": START, "end_date": END}


# ── Fixtures ──


@pytest.fixture(scope="module")
def wired_client(tmp_path_factory):
    """Client with both routers, the default config (150 names), temp store."""
    real_load_config = rc.load_config

    def _small_load_config(*args, **kwargs):
        config = real_load_config(*args, **kwargs)
        config.universe.n_stocks = 150
        return config

    rc.load_config = _small_load_config
    rc.reset_context_cache()

    store = Store(str(tmp_path_factory.mktemp("wiring") / "wiring.db"))
    old_monitor_store = monitor._core_store
    old_routes_store = routes._core_store
    monitor._core_store = store
    routes._core_store = store

    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(monitor.router)
    client = TestClient(app)
    try:
        yield client
    finally:
        rc.load_config = real_load_config
        rc.reset_context_cache()
        monitor._core_store = old_monitor_store
        routes._core_store = old_routes_store
        store.close()


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    """A per-test Store so TCA seeds never touch the module store."""
    store = Store(str(tmp_path / "tca.db"))
    monkeypatch.setattr(monitor, "_core_store", store)
    yield store
    store.close()


def _seed_full_trades(store: Store, n: int = 3) -> None:
    for i in range(n):
        code = f"60000{i}"
        store.save_order({
            "order_id": f"o{i}", "code": code, "side": "buy",
            "quantity": 100, "price": 10.0 + i, "status": "filled",
            "created_at": "2024-07-15T10:00:00",
        })
        store.save_trade({
            "trade_id": f"t{i}", "order_id": f"o{i}", "code": code,
            "side": "buy", "quantity": 100, "price": 10.0 + i + 0.05,
            "commission": 0.3, "executed_at": "2024-07-15T10:00:00",
        })


def _assert_json_strict(payload) -> None:
    """Starlette encodes with allow_nan=False; NaN here would 500 in prod."""
    json.dumps(payload, allow_nan=False)


# ── Real values on the default synthetic config ──


class TestICDecay:
    def test_returns_measured_decay(self, wired_client):
        resp = wired_client.get("/api/analysis/ic-decay", params=WINDOW)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        _assert_json_strict(data)

        assert data["available"] is True
        assert data["horizons"] == [1, 5, 10, 20]
        assert data["factors"], "no factor produced an IC decay curve"
        assert any(
            math.isfinite(point["ic"])
            for factor in data["factors"]
            for point in factor["points"]
        ), "the IC decay curve is not a measurement"

        for factor in data["factors"]:
            assert factor["factor"]
            assert factor["lags"] == [p["horizon"] for p in factor["points"]]
            assert factor["ics"] == [p["ic"] for p in factor["points"]]
            for point in factor["points"]:
                assert point["horizon"] in (1, 5, 10, 20)
                assert point["n"] > 0
                assert math.isfinite(point["ic"])

    def test_provenance_is_echoed(self, wired_client):
        data = wired_client.get("/api/analysis/ic-decay", params=WINDOW).json()
        assert data["as_of"] == "2025-06-30"
        assert data["provider"] == "synthetic"
        assert data["synthetic"] is True
        assert len(data["config_hash"]) == 12
        # computed_at is an ISO timestamp, not a placeholder.
        parsed = pd.Timestamp(data["computed_at"])
        assert parsed.tz is not None
        assert data["n_sessions"] > 200
        assert data["n_assets"] == 150

    def test_ic_matches_the_evaluation_function_bit_for_bit(
        self, prices, returns, sample_factor,
    ):
        """The endpoint must not re-derive, smooth or scale the value.

        ``ic_decay`` returns ``rank_ic(period=h).mean()``; the endpoint
        computes exactly that for the reported horizons.
        """
        from quant_platform.factors.evaluation import ic_decay

        factors = {
            "momentum": sample_factor,
            "volatility": returns.rolling(20).std(),
        }
        context = make_context(prices, returns, factors)
        out = routes._compute_ic_decay(context)

        assert out["available"] is True
        assert out["window_sessions"] <= 252
        by_name = {f["factor"]: f for f in out["factors"]}
        for name, panel in factors.items():
            # The endpoint measures the trailing analysis window.
            direct = ic_decay(panel.tail(252), returns, max_periods=20)
            for point in by_name[name]["points"]:
                expected = float(direct.loc[point["horizon"]])
                assert abs(point["ic"] - expected) <= 1e-12, (
                    f"{name} horizon {point['horizon']}: "
                    f"{point['ic']} != {expected}"
                )


class TestFactorCorrelation:
    def test_matrix_is_square_symmetric_and_measured(self, wired_client):
        resp = wired_client.get("/api/analysis/correlation", params=WINDOW)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        _assert_json_strict(data)

        assert data["available"] is True
        names = data["names"]
        matrix = data["matrix"]
        assert len(names) >= 2
        assert len(matrix) == len(names)
        assert all(len(row) == len(names) for row in matrix)

        for i in range(len(names)):
            assert matrix[i][i] == 1.0
            for j in range(len(names)):
                assert matrix[i][j] == matrix[j][i], "matrix is not symmetric"
                value = matrix[i][j]
                if value is not None:
                    assert -1.0 <= value <= 1.0
        # Real correlations, not a constant or a diagonal-only matrix.
        off_diagonal = [
            matrix[i][j]
            for i in range(len(names)) for j in range(len(names))
            if i != j and matrix[i][j] is not None
        ]
        assert off_diagonal
        assert any(abs(value) > 1e-9 for value in off_diagonal)

    def test_fallback_handles_panels_with_different_date_indices(
        self, prices, returns,
    ):
        """Regression: the pairwise fallback used to index the global matrix
        with a per-date k x k correlation block, raising IndexError when a
        panel (e.g. a quarterly fundamental) lacked a date in the union."""
        base = prices.pct_change(fill_method=None)
        factors = {
            "full": base,
            "short": base.tail(120),
            "late": base.iloc[150:],
        }
        context = make_context(prices, returns, factors)
        out = routes._compute_factor_correlation(context)

        assert out["available"] is True
        assert len(out["matrix"]) == len(out["names"]) == 3
        for row in out["matrix"]:
            assert len(row) == 3

    def test_pairwise_fallback_shape_with_disjoint_index(self, prices):
        base = prices.pct_change(fill_method=None)
        corr = routes._pairwise_factor_correlation({
            "full": base,
            "short": base.tail(150),
        })
        assert corr.shape == (2, 2)
        assert corr.index.tolist() == ["full", "short"]


class TestWalkForward:
    def test_returns_real_oos_folds(self, wired_client):
        resp = wired_client.post(
            "/api/walkforward",
            json={
                **WINDOW,
                "train_period": 126,
                "test_period": 21,
                "mode": "rolling",
                "folds": 1,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        _assert_json_strict(data)

        assert data["available"] is True, data.get("reason")
        assert data["n_folds"] >= 1
        assert data["fold_details"]
        assert data["signal_recomputed"] is True, (
            "the signal was not recomputed from train-only data"
        )
        for fold in data["fold_details"]:
            assert fold["oos_days"] > 0
            assert fold["sharpe"] is None or math.isfinite(fold["sharpe"])
        assert any(
            fold["sharpe"] is not None for fold in data["fold_details"]
        ), "no fold produced a Sharpe"
        assert len(data["oos_equity"]) == len(data["oos_dates"])
        assert data["oos_equity"]
        assert data["stability"]["total_folds"] >= 1
        assert set(data["stability"]) >= {
            "mean_sharpe", "std_sharpe", "sharpe_consistency",
            "positive_folds", "total_folds", "mean_return_pct",
        }
        # Response-shape regression: the panel reads all of these.
        assert set(data) >= {
            "n_folds", "mode", "train_period", "test_period",
            "fold_details", "oos_equity", "oos_dates", "stability",
        }
        assert data["train_period"] == 126
        assert data["test_period"] == 21


class TestRiskDecomposition:
    def _weights(self) -> dict[str, float]:
        context = asyncio.run(rc.get_research_context(
            start_date=START, end_date=END,
        ))
        assets = list(context.prices.columns[:5])
        return dict(zip(
            assets, (0.35, 0.25, 0.2, 0.12, 0.08), strict=False,
        ))

    def test_shares_are_euler_contributions_that_sum_to_one(self, wired_client):
        resp = wired_client.post(
            "/api/risk/decompose",
            json={**WINDOW, "weights": self._weights()},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        _assert_json_strict(data)

        assert data["available"] is True, data.get("reason")
        assert data["weights_source"] == "request"
        assert data["total_risk_pct"] > 0
        assert 0.0 <= data["r_squared"] <= 1.0

        shares = [row["risk_share"] for row in data["factors"]]
        alpha_rows = [r for r in data["factors"] if r["contribution"] == "alpha"]
        assert len(alpha_rows) == 1, "idiosyncratic risk row missing"
        total = sum(shares)
        assert abs(total - 1.0) <= 1e-6, f"risk shares sum to {total}"
        assert abs(data["risk_share"]["total"] - 1.0) <= 1e-6
        assert data["idiosyncratic_risk_pct"] > 0

        # Independent checks of the Euler convention: every factor share is
        # its variance contribution over total variance, and the alpha row is
        # the specific variance share (allowing only float residual).
        total_variance = (data["total_risk_pct"] / 100.0) ** 2
        for row in data["factors"]:
            assert row["variance_contribution"] is not None
            assert abs(
                row["risk_share"] - row["variance_contribution"] / total_variance
            ) <= 1e-9
        specific_variance = (data["idiosyncratic_risk_pct"] / 100.0) ** 2
        assert abs(alpha_rows[0]["risk_share"] - specific_variance / total_variance) <= 1e-9

        # Factor rows carry measured exposure where the model produced one.
        factor_rows = [r for r in data["factors"] if r["contribution"] == "factor"]
        assert factor_rows
        assert any(row["beta"] is not None for row in factor_rows)

    def test_degenerate_context_reports_unavailable(self, prices, returns):
        empty_prices = prices.iloc[0:0]
        empty_returns = returns.iloc[0:0]
        context = make_context(empty_prices, empty_returns, {})
        out = routes._decompose_risk(context, {"weights": None})

        assert out["available"] is False
        assert out["factors"] == []
        assert out["risk_share"]["sums_to_one"] is False
        assert "fit failed" in out["reason"]

    def test_unknown_date_is_refused_not_silently_latest(self, wired_client):
        data = wired_client.post(
            "/api/risk/decompose",
            json={**WINDOW, "date": "1999-01-04"},
        ).json()
        assert data["available"] is False
        assert "not in the research context" in data["reason"]

    def test_reuses_the_barra_computation(self):
        """No duplicated math: both endpoints call `_barra_decomposition`."""
        assert "_barra_decomposition" in inspect.getsource(
            routes._decompose_risk,
        )
        assert "_barra_decomposition" in inspect.getsource(
            routes._barra_decompose_impl,
        )


class TestMonitorEndpoints:
    def test_tca_implementation_shortfall_from_seeded_trades(
        self, wired_client, temp_store,
    ):
        _seed_full_trades(temp_store, n=3)
        resp = wired_client.get("/api/monitor/tca-summary")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        _assert_json_strict(data)

        assert data["available"] is True
        assert data["n_orders"] == 3
        assert data["n_trades"] == 3
        assert data["n_analyzed"] == 3
        assert math.isfinite(data["mean_is_bps"])
        assert math.isfinite(data["median_is_bps"])
        expected = sum(0.05 / (10.0 + i) * 1e4 for i in range(3)) / 3.0
        assert data["mean_is_bps"] == pytest.approx(expected, abs=1e-6)
        assert data["daily_trend"] and data["by_ticker"]
        assert data["decomposition"]["fields"]["implementation_shortfall"] is True
        assert data["decomposition"]["fields"]["arrival"] is False
        assert data["decomposition"]["fields"]["delay_impact_timing"] is False
        # Unmeasured decompositions are null, not a plausible zero.
        assert data["mean_arrival_bps"] is None
        assert data["mean_delay_bps"] is None
        assert data["mean_impact_bps"] is None
        assert data["mean_timing_bps"] is None
        assert data["as_of"] == "2024-07-15"

    def test_tca_sell_side_sign_convention(self, wired_client, temp_store):
        temp_store.save_order({
            "order_id": "sell1", "code": "600519", "side": "sell",
            "quantity": 100, "price": 11.0, "status": "filled",
            "created_at": "2024-07-16T10:00:00",
        })
        temp_store.save_trade({
            "trade_id": "sell-t1", "order_id": "sell1", "code": "600519",
            "side": "sell", "quantity": 100, "price": 10.9,
            "executed_at": "2024-07-16T10:00:00",
        })
        data = wired_client.get("/api/monitor/tca-summary").json()

        assert data["available"] is True
        # Sold below the reference price: a positive cost, not a gain.
        assert data["mean_is_bps"] > 0
        expected = (11.0 - 10.9) / 11.0 * 1e4
        assert data["mean_is_bps"] == pytest.approx(expected, abs=1e-6)

    def test_factor_status_has_rolling_ic(self, wired_client):
        resp = wired_client.get("/api/monitor/factor-status", params=WINDOW)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        _assert_json_strict(data)

        assert data["available"] is True, data.get("reason")
        assert data["factors"], "no factor stats"
        assert data["ic_dates"]
        assert len(data["ic_dates"]) <= 60
        for name, series in data["rolling_ic"].items():
            assert len(series) == len(data["ic_dates"])
            assert any(value is not None for value in series)
        for factor in data["factors"]:
            assert factor["name"]
            assert factor["trend"] in ("up", "down")
            assert factor["alert"] in ("green", "yellow", "red")
            assert math.isfinite(factor["current_ic"])
            assert math.isfinite(factor["icir"])
        assert isinstance(data["disabled_factors"], list)

    def test_capacity_has_a_finite_adv_estimate(self, wired_client):
        resp = wired_client.get("/api/monitor/capacity-gauge", params=WINDOW)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        _assert_json_strict(data)

        assert data["available"] is True, data.get("reason")
        assert math.isfinite(data["capacity_aum"])
        assert data["capacity_aum"] > 0
        assert math.isfinite(data["usage_pct"])
        assert data["aum_curve"], "the AUM-Sharpe curve is empty"
        for point in data["aum_curve"]:
            assert math.isfinite(point["aum"])
            assert point["sharpe"] is None or math.isfinite(point["sharpe"])
        policy = data["participation_assumption"]
        assert policy["max_participation"] > 0
        assert policy["adv_window_sessions"] > 0
        assert "ADV" in policy["capacity_definition"]
        # No live book is seeded in this module, so the AUM/usage readout is
        # explicitly marked unavailable instead of a measured 0.
        assert data["aum_source"] == "none"
        assert data["usage_available"] is False


# ── Missing inputs ──


class TestMissingInputs:
    def test_tca_without_trades_says_so(self, wired_client, temp_store):
        data = wired_client.get("/api/monitor/tca-summary").json()
        assert data["available"] is False
        assert data["reason"] == "no stored trades"
        assert data["daily_trend"] == []
        assert data["mean_is_bps"] == 0

    def test_tca_without_reference_price_says_so(self, wired_client, temp_store):
        temp_store.save_trade({
            "trade_id": "t1", "order_id": "missing", "code": "600000",
            "side": "buy", "quantity": 100, "price": 10.0,
            "executed_at": "2024-07-15T10:00:00",
        })
        data = wired_client.get("/api/monitor/tca-summary").json()
        assert data["available"] is False
        assert "reference price" in data["reason"]
        assert data["skipped"]["missing_order"] == 1
        assert data["daily_trend"] == []

    def test_walkforward_sample_too_short_states_the_requirement(self, wired_client):
        data = wired_client.post(
            "/api/walkforward",
            json={
                **WINDOW, "train_period": 2520, "test_period": 1260,
                "mode": "rolling", "folds": 1,
            },
        ).json()
        assert data["available"] is False
        assert "at least 3780 sessions" in data["reason"]
        assert data["n_folds"] == 0
        assert data["fold_details"] == []
        assert data["oos_equity"] == []
        # Shape is preserved for the panel.
        assert set(data) >= {
            "n_folds", "mode", "train_period", "test_period",
            "fold_details", "oos_equity", "oos_dates", "stability",
        }

    def test_walkforward_too_short_context_reports_the_sample(
        self, prices, returns,
    ):
        short_prices = prices.tail(50).copy()
        short_returns = returns.reindex(short_prices.index).copy()
        context = make_context(short_prices, short_returns, {"m": short_prices})
        out = routes._run_walkforward(context, {
            "train_period": 126, "test_period": 21,
            "mode": "rolling", "folds": None,
        })
        assert out["available"] is False
        assert "at least 147 sessions" in out["reason"]
        assert "has 50" in out["reason"]
        assert set(out["stability"].values()) == {0}

    def test_factor_status_insufficient_history(self, prices, returns):
        short_prices = prices.tail(50).copy()
        short_returns = returns.reindex(short_prices.index).copy()
        context = make_context(short_prices, short_returns, {
            "momentum": short_prices.pct_change(fill_method=None),
        })
        out = monitor._compute_factor_status(context)
        assert out.available is False
        assert out.required_sessions > 50
        assert "sessions" in out.reason
        assert out.factors == []

    def test_capacity_without_volume_panel(self, prices, returns, monkeypatch):
        context = make_context(
            prices.tail(300), returns.reindex(prices.tail(300).index),
            {"momentum": prices.tail(300).pct_change(fill_method=None)},
            turnover=None,
            fin_unstacked=None,
        )
        monkeypatch.setattr(monitor, "_get_store", lambda: None)
        out = monitor._compute_capacity_gauge(context)
        assert out.available is False
        assert "no volume panel" in out.reason
        assert out.aum_curve == []


# ── Bounded execution (429) ──


class TestSaturation:
    def test_third_heavy_request_gets_429(self, wired_client, monkeypatch):
        # Unique date windows, so neither request can be served from cache.
        entered = threading.Event()
        release = threading.Event()
        lock = threading.Lock()
        count = {"n": 0}

        def blocking_loader(config, resolution, key):
            with lock:
                count["n"] += 1
                if count["n"] >= 2:
                    entered.set()
            release.wait(timeout=20)
            raise RuntimeError("test loader released")

        monkeypatch.setattr(rc, "_load_context_uncached", blocking_loader)

        results: list[object] = []

        def request(window: dict) -> None:
            # Separate clients: each request gets its own portal/event loop.
            app = FastAPI()
            app.include_router(routes.router)
            client = TestClient(app)
            results.append(client.get("/api/analysis/ic-decay", params=window))

        windows = [
            {"start_date": "2023-01-01", "end_date": "2023-06-30"},
            {"start_date": "2023-07-01", "end_date": "2023-12-31"},
        ]
        threads = [
            threading.Thread(target=request, args=(window,))
            for window in windows
        ]
        try:
            for thread in threads:
                thread.start()
            assert entered.wait(timeout=20), (
                "the two heavy requests never entered the loader"
            )
            third = wired_client.get(
                "/api/analysis/ic-decay",
                params={"start_date": "2022-01-01", "end_date": "2022-06-30"},
            )
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=30)

        assert third.status_code == 429
        assert third.headers.get("Retry-After") == "5"
        assert [resp.status_code for resp in results] == [503, 503]

    def test_fully_cached_response_does_not_take_a_slot(self, wired_client):
        # Prime the context and the derived result for the shared window.
        first = wired_client.get("/api/analysis/ic-decay", params=WINDOW)
        assert first.status_code == 200
        held = 0
        try:
            while routes._analytics_slots.acquire(blocking=False):
                held += 1
                if held >= routes._ANALYTICS_WORKERS:
                    break
            second = wired_client.get("/api/analysis/ic-decay", params=WINDOW)
        finally:
            for _ in range(held):
                routes._analytics_slots.release()
        assert second.status_code == 200
        assert second.json()["cached"] is True


# ── Cache behaviour ──


class TestContextCache:
    def test_second_request_hits_the_cache_without_reloading(
        self, wired_client, monkeypatch,
    ):
        rc.reset_context_cache()
        real_load_data = rc._load_data
        calls: list[int] = []

        def counting_load_data(*args, **kwargs):
            calls.append(1)
            return real_load_data(*args, **kwargs)

        monkeypatch.setattr(rc, "_load_data", counting_load_data)
        window = {"start_date": "2024-03-01", "end_date": "2024-09-30"}

        first = wired_client.get("/api/analysis/correlation", params=window)
        assert first.status_code == 200, first.text
        assert first.json()["cached"] is False
        assert len(calls) == 1

        second = wired_client.get("/api/analysis/correlation", params=window)
        assert second.status_code == 200, second.text
        assert second.json()["cached"] is True
        assert len(calls) == 1, "the cached context reloaded the data pipeline"

        # A different date range is a different key and must reload.
        other = wired_client.get(
            "/api/analysis/correlation",
            params={"start_date": "2024-04-01", "end_date": "2024-10-31"},
        )
        assert other.status_code == 200, other.text
        assert other.json()["cached"] is False
        assert len(calls) == 2
        assert other.json()["config_hash"] != second.json()["config_hash"]

    def test_context_key_changes_with_config_and_dates(self):
        base = small_config(150)
        other_universe = small_config(160)
        other_dates = small_config(150)
        other_dates.data.start_date = "2023-01-01"
        identity = {"provider": "synthetic", "synthetic": True}

        base_key = rc.make_context_key(base, {"identity": identity})
        assert base_key == rc.make_context_key(
            small_config(150), {"identity": identity},
        )
        assert base_key != rc.make_context_key(
            other_universe, {"identity": identity},
        )
        assert base_key != rc.make_context_key(
            other_dates, {"identity": identity},
        )


# ── No fabricated numbers ──


class TestNoFabricatedConstants:
    def test_wired_computations_do_not_generate_values(self):
        functions = (
            routes._compute_ic_decay,
            routes._compute_factor_correlation,
            routes._decompose_risk,
            routes._run_walkforward,
            monitor._compute_tca_summary,
            monitor._compute_factor_status,
            monitor._compute_capacity_gauge,
        )
        banned = (
            "np.random", "random.uniform", "random.random",
            "np.exp(-", "1103515245",
        )
        for function in functions:
            body = inspect.getsource(function)
            for expression in banned:
                assert expression not in body, (
                    f"{function.__name__} generates {expression!r}"
                )

    def test_the_old_hardcoded_decomposition_is_gone(self):
        body = inspect.getsource(routes._decompose_risk)
        for expression in ("15.2", "icir * 15", "0.5 + icir"):
            assert expression not in body

    def test_walkforward_does_not_slice_a_stored_equity_curve(self):
        body = inspect.getsource(routes._run_walkforward)
        assert "chart_data" not in body
        assert "Generate synthetic walk-forward" not in body
        assert "validator.run(" in body
