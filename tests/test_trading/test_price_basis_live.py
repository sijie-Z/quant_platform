"""Paper/live money paths must use raw prices and fail closed on adjusted data.

Controlled scenario: raw close 10 CNY, a 2-for-1 split halves it to 5 while
the adjusted path stays at 10.  A paper buy of 100 shares at raw 10 costs
1000 CNY; after the split the position is worth 500, not 1000 (the hfq
level).  A provider that only offers adjusted levels makes the engine
refuse rather than silently mark the book at a non-tradeable price (D15).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.base import DataProvider
from quant_platform.data.schema import DataQualityError
from quant_platform.trading.live_engine import LiveEngine
from quant_platform.trading.live_runner import LiveRunner

DATES = pd.bdate_range("2024-01-02", periods=50, name="date")
SPLIT_DAY = 43
ASSET = "600519"
CANONICAL = "600519.SH"


def _price_frame(*, adjusted_only: bool = False) -> pd.DataFrame:
    n = len(DATES)
    if adjusted_only:
        close = np.full(n, 10.0)
        divisor = np.ones(n)
    else:
        raw = np.full(n, 10.0)
        raw[SPLIT_DAY:] = 5.0
        divisor = np.ones(n)
        divisor[SPLIT_DAY:] = 0.5
        close = raw

    index = pd.MultiIndex.from_product([DATES, [ASSET]], names=["date", "asset"])
    frame = pd.DataFrame(index=index)
    for column in ("open", "high", "low", "close"):
        frame[column] = np.repeat(close, 1)
    # Varying volume prevents the pipeline from flagging every unchanged
    # close as a stale (suspended) row.
    frame["volume"] = np.repeat(1_000_000.0 + np.arange(len(DATES)), 1)
    frame["turnover"] = 0.01
    frame["adj_factor"] = np.repeat(divisor, 1)
    return frame


class _Provider(DataProvider):
    price_adjustment = "raw"
    adjusted_price_basis = "hfq"
    adjustment_factor_available = True

    def __init__(self, *, adjusted_only: bool = False):
        self._adjusted_only = adjusted_only
        self._prices = _price_frame(adjusted_only=adjusted_only)
        self._metadata = pd.DataFrame(
            {
                "sector": ["食品饮料"],
                "market_cap_group": ["large"],
                "is_st": [False],
                "listing_date": [pd.Timestamp("2020-01-01")],
                "delisting_date": [pd.NaT],
            },
            index=pd.Index([ASSET], name="asset"),
        )

    def get_prices(self, start_date, end_date, fields=None):
        df = self._prices.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
        return df if not fields else df[list(fields)]

    def get_financials(self, start_date, end_date):
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["date", "asset"]),
            columns=[
                "market_cap", "total_assets", "net_assets", "revenue",
                "net_profit", "roe", "pb_ratio", "pe_ratio", "asset_growth",
            ],
        )

    def get_benchmark(self, start_date, end_date):
        return pd.Series(np.zeros(len(DATES)), index=DATES, name="benchmark")

    def get_metadata(self):
        return self._metadata


class _AdjustedOnlyProvider(_Provider):
    price_adjustment = "hfq"
    adjusted_price_basis = "hfq"
    adjustment_factor_available = False

    def __init__(self):
        super().__init__(adjusted_only=True)


def _pipeline(provider) -> DataPipeline:
    pipe = DataPipeline(
        provider,
        str(DATES[0].date()),
        str(DATES[-1].date()),
        exclude_st=False,
        exclude_suspended=False,
    )
    pipe.run()
    return pipe


def _loaded_engine(pipe: DataPipeline) -> LiveEngine:
    engine = LiveEngine()
    engine.load_history(pipe.returns, pipe.get_raw_close(), price_basis="raw")
    return engine


def test_paper_buy_uses_raw_price_and_marks_raw_after_split():
    pipe = _pipeline(_Provider())
    engine = _loaded_engine(pipe)

    # 100 shares at raw 10 CNY costs exactly 1000 CNY.
    engine.state.cash = 1_000.0
    engine._update_equity(DATES[0])
    engine._execute_rebalance(DATES[41])

    assert len(engine.state.positions) == 1
    position = engine.state.positions[0]
    assert position.asset == CANONICAL
    assert position.shares == 100
    assert position.entry_price == pytest.approx(10.0)
    assert engine.state.cash == pytest.approx(0.0)
    # Pre-split mark: cash 0 + 100 x raw 10 = 1000.
    assert engine.state.current_equity == pytest.approx(1_000.0)

    raw = pipe.get_raw_close()
    engine._update_equity(DATES[SPLIT_DAY])
    # Post-split raw price is 5: the book is worth 500 CNY, not the hfq 1000.
    expected = engine.state.cash + position.shares * raw.loc[
        DATES[SPLIT_DAY], CANONICAL
    ]
    assert engine.state.current_equity == pytest.approx(500.0)
    assert engine.state.current_equity == pytest.approx(expected)
    assert engine.state.current_equity != pytest.approx(1_000.0)

    # The research path stayed continuous, so the split produced no return.
    assert pipe.returns[CANONICAL].abs().max() < 1e-12
    assert pipe.get_adjusted_close().loc[
        DATES[SPLIT_DAY], CANONICAL
    ] == pytest.approx(10.0)


def test_live_engine_refuses_an_adjusted_panel():
    pipe = _pipeline(_AdjustedOnlyProvider())

    with pytest.raises(DataQualityError, match="no raw close"):
        pipe.get_raw_close()

    engine = LiveEngine()
    with pytest.raises(ValueError, match="raw"):
        # This is exactly what the old daemon wiring did: mark with get_close.
        engine.load_history(pipe.returns, pipe.get_close())
    # Even an unlabeled panel can be explicitly declared adjusted.
    with pytest.raises(ValueError, match="raw"):
        engine.load_history(pipe.returns, pipe.get_close(), price_basis="hfq")


def test_live_runner_refuses_adjusted_quotes_and_reports_basis():
    runner = LiveRunner(broker_type="simulated", dual_track=False)
    with pytest.raises(ValueError, match="raw"):
        runner.set_prices({"600519": 10.0}, price_basis="hfq")
    with pytest.raises(ValueError, match="raw"):
        runner.set_prices({"600519": 10.0}, price_basis="qfq")

    runner.set_universe(["600519"])
    assert runner._universe == [CANONICAL]
    report = runner.run(days=2)
    payload = report.to_dict()
    assert payload["price_basis"] == "raw"
    assert runner._daily_reports
    assert all(
        daily.price_basis == "raw" for daily in runner._daily_reports
    )
