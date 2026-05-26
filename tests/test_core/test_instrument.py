"""Tests for cross-asset instrument abstraction."""

import pytest

from quant_platform.core.instrument import (
    AssetUniverse,
    Instrument,
    InstrumentType,
)


# ── InstrumentType ──


class TestInstrumentType:
    def test_stock_value(self):
        assert InstrumentType.STOCK == "stock"

    def test_etf_value(self):
        assert InstrumentType.ETF == "etf"

    def test_future_value(self):
        assert InstrumentType.FUTURE == "future"

    def test_option_value(self):
        assert InstrumentType.OPTION == "option"

    def test_index_value(self):
        assert InstrumentType.INDEX == "index"

    def test_from_string(self):
        assert InstrumentType("stock") == InstrumentType.STOCK
        assert InstrumentType("future") == InstrumentType.FUTURE


# ── Instrument defaults ──


class TestInstrumentDefaults:
    def test_default_is_stock(self):
        inst = Instrument()
        assert inst.asset_type == InstrumentType.STOCK

    def test_default_multiplier(self):
        inst = Instrument()
        assert inst.multiplier == 1.0

    def test_default_lot_size(self):
        inst = Instrument()
        assert inst.lot_size == 100

    def test_default_margin_rate(self):
        inst = Instrument()
        assert inst.margin_rate == 1.0

    def test_default_t_plus(self):
        inst = Instrument()
        assert inst.t_plus == 1


# ── Instrument properties ──


class TestInstrumentProperties:
    def test_is_equity_stock(self):
        assert Instrument(symbol="600519").is_equity is True

    def test_is_equity_etf(self):
        inst = Instrument(symbol="510300", asset_type=InstrumentType.ETF)
        assert inst.is_equity is True

    def test_is_derivative_future(self):
        inst = Instrument(symbol="IF2406", asset_type=InstrumentType.FUTURE)
        assert inst.is_derivative is True

    def test_is_derivative_option(self):
        inst = Instrument(symbol="IO2406", asset_type=InstrumentType.OPTION)
        assert inst.is_derivative is True

    def test_stock_not_derivative(self):
        assert Instrument(symbol="600519").is_derivative is False

    def test_future_not_equity(self):
        inst = Instrument(symbol="IF2406", asset_type=InstrumentType.FUTURE)
        assert inst.is_equity is False


# ── Instrument.notional ──


class TestInstrumentNotional:
    def test_stock_notional(self):
        inst = Instrument(symbol="600519", multiplier=1.0)
        assert inst.notional(1800.0, 100) == 180000.0

    def test_future_notional(self):
        inst = Instrument(symbol="IF2406", multiplier=300.0, lot_size=1)
        assert inst.notional(3500.0, 1) == 1050000.0

    def test_etf_notional(self):
        inst = Instrument(symbol="510300", multiplier=1.0, asset_type=InstrumentType.ETF)
        assert inst.notional(4.0, 1000) == 4000.0

    def test_commodity_future_notional(self):
        inst = Instrument(symbol="cu2406", multiplier=5.0, lot_size=1)
        assert inst.notional(70000.0, 2) == 700000.0


# ── Instrument.margin_required ──


class TestInstrumentMargin:
    def test_stock_full_margin(self):
        inst = Instrument(symbol="600519", margin_rate=1.0)
        assert inst.margin_required(100.0, 100) == 10000.0

    def test_future_margin(self):
        inst = Instrument(symbol="IF2406", multiplier=300.0, margin_rate=0.12, lot_size=1)
        assert inst.margin_required(3500.0, 1) == 3500.0 * 300.0 * 0.12


# ── Instrument.round_lot / valid_quantity ──


class TestInstrumentLot:
    def test_round_lot_stock(self):
        inst = Instrument(lot_size=100)
        assert inst.round_lot(150) == 100
        assert inst.round_lot(300) == 300
        assert inst.round_lot(99) == 0

    def test_round_lot_future(self):
        inst = Instrument(lot_size=1)
        assert inst.round_lot(1) == 1
        assert inst.round_lot(5) == 5

    def test_valid_quantity_stock(self):
        inst = Instrument(lot_size=100)
        assert inst.valid_quantity(100) is True
        assert inst.valid_quantity(150) is False
        assert inst.valid_quantity(0) is False

    def test_valid_quantity_future(self):
        inst = Instrument(lot_size=1)
        assert inst.valid_quantity(1) is True
        assert inst.valid_quantity(3) is True


# ── Instrument.tick_round ──


class TestInstrumentTick:
    def test_stock_tick(self):
        inst = Instrument(tick_size=0.01)
        assert inst.tick_round(10.055) == 10.06

    def test_future_tick(self):
        inst = Instrument(tick_size=0.2)
        assert inst.tick_round(3500.3) == 3500.4

    def test_etf_tick(self):
        inst = Instrument(tick_size=0.001)
        assert inst.tick_round(4.0555) == 4.056


# ── Instrument.commission ──


class TestInstrumentCommission:
    def test_stock_commission_rate(self):
        inst = Instrument(commission_rate=0.0003, multiplier=1.0)
        assert abs(inst.commission(100.0, 100) - 3.0) < 0.01

    def test_future_commission_per_lot(self):
        inst = Instrument(commission_per_lot=25.0, lot_size=1)
        assert inst.commission(3500.0, 2) == 50.0

    def test_default_commission_rate(self):
        inst = Instrument()  # commission_rate is None, falls back to 0.0003
        assert abs(inst.commission(100.0, 100) - 3.0) < 0.01

    def test_etf_custom_commission(self):
        inst = Instrument(commission_rate=0.0001, multiplier=1.0)
        assert abs(inst.commission(4.0, 1000) - 0.4) < 0.01


# ── Instrument.stamp_tax ──


class TestInstrumentStampTax:
    def test_stock_sell_stamp_tax(self):
        inst = Instrument(stamp_tax_rate=0.001, multiplier=1.0)
        assert abs(inst.stamp_tax(100.0, 100, "sell") - 10.0) < 0.01

    def test_stock_buy_no_stamp_tax(self):
        inst = Instrument(stamp_tax_rate=0.001)
        assert inst.stamp_tax(100.0, 100, "buy") == 0.0

    def test_etf_no_stamp_tax(self):
        inst = Instrument(stamp_tax_rate=0.0, asset_type=InstrumentType.ETF)
        assert inst.stamp_tax(4.0, 1000, "sell") == 0.0

    def test_future_no_stamp_tax(self):
        inst = Instrument(stamp_tax_rate=0.0, asset_type=InstrumentType.FUTURE)
        assert inst.stamp_tax(3500.0, 1, "sell") == 0.0

    def test_default_stamp_tax(self):
        inst = Instrument()  # stamp_tax_rate is None, falls back to 0.001
        assert abs(inst.stamp_tax(100.0, 100, "sell") - 10.0) < 0.01


# ── Instrument factories ──


class TestInstrumentFactories:
    def test_stock_factory(self):
        inst = Instrument.stock("600519")
        assert inst.symbol == "600519"
        assert inst.asset_type == InstrumentType.STOCK
        assert inst.lot_size == 100
        assert inst.multiplier == 1.0
        assert inst.t_plus == 1
        assert inst.price_limit == 0.10

    def test_stock_chinext_price_limit(self):
        inst = Instrument.stock("300750")
        assert inst.price_limit == 0.20

    def test_stock_star_market_price_limit(self):
        inst = Instrument.stock("688001")
        assert inst.price_limit == 0.20

    def test_etf_factory(self):
        inst = Instrument.etf("510300")
        assert inst.asset_type == InstrumentType.ETF
        assert inst.stamp_tax_rate == 0.0
        assert inst.tick_size == 0.001

    def test_index_future_factory(self):
        inst = Instrument.index_future("IF2406")
        assert inst.asset_type == InstrumentType.FUTURE
        assert inst.multiplier == 300.0
        assert inst.lot_size == 1
        assert inst.margin_rate == 0.12
        assert inst.commission_per_lot == 25.0
        assert inst.stamp_tax_rate == 0.0
        assert inst.t_plus == 0
        assert inst.exchange == "CFFEX"

    def test_index_future_custom_params(self):
        inst = Instrument.index_future("IC2406", multiplier=200, margin_rate=0.14)
        assert inst.multiplier == 200.0
        assert inst.margin_rate == 0.14

    def test_commodity_future_factory(self):
        inst = Instrument.commodity_future("cu2406", multiplier=5, tick_size=10.0)
        assert inst.asset_type == InstrumentType.FUTURE
        assert inst.multiplier == 5.0
        assert inst.tick_size == 10.0
        assert inst.exchange == "SHFE"


# ── Instrument.from_dict ──


class TestInstrumentFromDict:
    def test_basic_stock(self):
        data = {"symbol": "600519", "asset_type": "stock"}
        inst = Instrument.from_dict(data)
        assert inst.symbol == "600519"
        assert inst.asset_type == InstrumentType.STOCK

    def test_future_with_params(self):
        data = {
            "symbol": "IF2406",
            "asset_type": "future",
            "multiplier": 300,
            "lot_size": 1,
            "margin_rate": 0.12,
            "commission_per_lot": 25.0,
            "stamp_tax_rate": 0.0,
            "t_plus": 0,
        }
        inst = Instrument.from_dict(data)
        assert inst.symbol == "IF2406"
        assert inst.multiplier == 300
        assert inst.lot_size == 1
        assert inst.margin_rate == 0.12
        assert inst.t_plus == 0

    def test_etf(self):
        data = {"symbol": "510300", "asset_type": "etf", "stamp_tax_rate": 0.0}
        inst = Instrument.from_dict(data)
        assert inst.asset_type == InstrumentType.ETF
        assert inst.stamp_tax_rate == 0.0

    def test_defaults_for_missing_fields(self):
        data = {"symbol": "TEST"}
        inst = Instrument.from_dict(data)
        assert inst.lot_size == 100
        assert inst.multiplier == 1.0
        assert inst.margin_rate == 1.0


# ── AssetUniverse ──


class TestAssetUniverse:
    def test_empty_universe(self):
        u = AssetUniverse()
        assert len(u) == 0
        assert u.count == 0

    def test_add_and_get(self):
        u = AssetUniverse()
        inst = Instrument.stock("600519")
        u.add(inst)
        assert u.get("600519") is inst
        assert len(u) == 1

    def test_get_missing_returns_none(self):
        u = AssetUniverse()
        assert u.get("NONEXISTENT") is None

    def test_get_or_default_returns_stock(self):
        u = AssetUniverse()
        inst = u.get_or_default("600519")
        assert inst.asset_type == InstrumentType.STOCK
        assert inst.symbol == "600519"
        assert inst.lot_size == 100

    def test_get_or_default_returns_registered(self):
        u = AssetUniverse()
        custom = Instrument(symbol="IF2406", asset_type=InstrumentType.FUTURE, lot_size=1)
        u.add(custom)
        assert u.get_or_default("IF2406") is custom

    def test_filter_by_type(self):
        u = AssetUniverse()
        u.add(Instrument.stock("600519"))
        u.add(Instrument.etf("510300"))
        u.add(Instrument.index_future("IF2406"))

        stocks = u.filter_by_type(InstrumentType.STOCK)
        assert len(stocks) == 1
        assert stocks[0].symbol == "600519"

        futures = u.filter_by_type(InstrumentType.FUTURE)
        assert len(futures) == 1

        etfs = u.filter_by_type(InstrumentType.ETF)
        assert len(etfs) == 1

    def test_symbols_property(self):
        u = AssetUniverse()
        u.add(Instrument.stock("600519"))
        u.add(Instrument.stock("000001"))
        assert set(u.symbols) == {"600519", "000001"}

    def test_contains(self):
        u = AssetUniverse()
        u.add(Instrument.stock("600519"))
        assert "600519" in u
        assert "000001" not in u

    def test_overwrite_on_add(self):
        u = AssetUniverse()
        u.add(Instrument.stock("600519"))
        u.add(Instrument(symbol="600519", lot_size=200))
        assert u.get("600519").lot_size == 200


# ── AssetUniverse.from_config ──


class TestAssetUniverseFromConfig:
    def test_empty_config(self):
        u = AssetUniverse.from_config({})
        assert len(u) == 0

    def test_none_instruments(self):
        u = AssetUniverse.from_config({"instruments": None})
        assert len(u) == 0

    def test_from_dict_config(self):
        config = {
            "instruments": [
                {"symbol": "600519", "asset_type": "stock"},
                {"symbol": "IF2406", "asset_type": "future", "multiplier": 300, "lot_size": 1},
            ]
        }
        u = AssetUniverse.from_config(config)
        assert len(u) == 2
        assert u.get("600519").asset_type == InstrumentType.STOCK
        assert u.get("IF2406").multiplier == 300

    def test_from_object_config(self):
        """Config as an object with instruments attribute."""
        class MockConfig:
            instruments = [
                {"symbol": "510300", "asset_type": "etf"},
            ]
        u = AssetUniverse.from_config(MockConfig())
        assert len(u) == 1
        assert u.get("510300").asset_type == InstrumentType.ETF

    def test_from_config_with_instrument_objects(self):
        config = {
            "instruments": [
                Instrument.stock("600519"),
                Instrument.index_future("IF2406"),
            ]
        }
        u = AssetUniverse.from_config(config)
        assert len(u) == 2

    def test_backward_compat_no_instruments(self):
        """No instruments section → empty universe, get_or_default works."""
        u = AssetUniverse.from_config({"portfolio": {"lot_size": 100}})
        assert len(u) == 0
        inst = u.get_or_default("600519")
        assert inst.lot_size == 100


# ── Integration: PortfolioConstraints with AssetUniverse ──


class TestConstraintsWithUniverse:
    def test_get_lot_size_default(self):
        from quant_platform.portfolio.constraints import PortfolioConstraints
        c = PortfolioConstraints(lot_size=100)
        assert c.get_lot_size("600519") == 100

    def test_get_lot_size_with_universe(self):
        from quant_platform.portfolio.constraints import PortfolioConstraints
        u = AssetUniverse()
        u.add(Instrument(symbol="IF2406", asset_type=InstrumentType.FUTURE, lot_size=1))
        c = PortfolioConstraints(lot_size=100, asset_universe=u)
        assert c.get_lot_size("600519") == 100  # not in universe, fallback
        assert c.get_lot_size("IF2406") == 1     # from universe

    def test_get_multiplier_default(self):
        from quant_platform.portfolio.constraints import PortfolioConstraints
        c = PortfolioConstraints()
        assert c.get_multiplier("600519") == 1.0

    def test_get_multiplier_with_universe(self):
        from quant_platform.portfolio.constraints import PortfolioConstraints
        u = AssetUniverse()
        u.add(Instrument.index_future("IF2406", multiplier=300.0))
        c = PortfolioConstraints(asset_universe=u)
        assert c.get_multiplier("IF2406") == 300.0
        assert c.get_multiplier("600519") == 1.0


# ── Integration: CostModel with AssetUniverse ──


class TestCostModelWithUniverse:
    def test_stock_costs_unchanged(self):
        from quant_platform.backtest.cost_model import CostModel
        cm = CostModel()
        cost = cm.compute_costs(100000, is_sell=None)
        assert cost > 0

    def test_etf_no_stamp_tax(self):
        from quant_platform.backtest.cost_model import CostModel
        u = AssetUniverse()
        u.add(Instrument(symbol="510300", asset_type=InstrumentType.ETF, stamp_tax_rate=0.0))
        cm = CostModel(asset_universe=u)
        import pandas as pd
        is_sell = pd.Series([True])
        cost = cm.compute_costs(pd.Series([100000]), is_sell=is_sell, symbol="510300")
        # With stamp_tax=0, sell cost equals buy cost (no stamp tax difference)
        cost_no_sell = cm.compute_costs(pd.Series([100000]), is_sell=pd.Series([False]), symbol="510300")
        assert abs(float(cost.iloc[0]) - float(cost_no_sell.iloc[0])) < 0.01

    def test_compute_costs_instrument_stock(self):
        from quant_platform.backtest.cost_model import CostModel
        cm = CostModel()
        cost = cm.compute_costs_instrument(100.0, 100, "buy", "600519")
        expected = 100.0 * 100 * 0.0003 + 100.0 * 100 * 0.001  # comm + slippage
        assert abs(cost - expected) < 0.1

    def test_compute_costs_instrument_with_universe(self):
        from quant_platform.backtest.cost_model import CostModel
        u = AssetUniverse()
        u.add(Instrument(symbol="IF2406", asset_type=InstrumentType.FUTURE,
                         commission_per_lot=25.0, stamp_tax_rate=0.0,
                         multiplier=300.0, lot_size=1))
        cm = CostModel(asset_universe=u)
        cost = cm.compute_costs_instrument(3500.0, 1, "sell", "IF2406")
        # commission=25, stamp_tax=0, slippage=3500*300*0.001
        expected = 25.0 + 0.0 + 3500.0 * 300 * 0.001
        assert abs(cost - expected) < 0.1
