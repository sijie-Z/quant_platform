"""Tests for the tick-level backtesting engine."""

import numpy as np
import pandas as pd
from quant_platform.backtest.tick_engine import (
    BacktestConfig,
    BacktestResult,
    Position,
    Tick,
    TickBacktester,
    TickDataSource,
    TWAPAlgorithm,
    VWAPAlgorithm,
)
from quant_platform.execution.market_impact import (
    AlmgrenChrissModel,
    CompositeImpactModel,
    ExecutionCostCalculator,
    KyleModel,
    SquareRootModel,
)

# ── Market Impact Tests ──


class TestAlmgrenChriss:
    def test_basic_estimate(self):
        model = AlmgrenChrissModel()
        impact = model.estimate(
            order_quantity=10000,
            market_volume=1000000,
            volatility=0.02,
            spread=0.01,
            price=100.0,
        )
        assert impact.temporary > 0
        assert impact.permanent > 0
        assert impact.total > 0
        assert impact.participation_rate == 0.01

    def test_larger_order_more_impact(self):
        model = AlmgrenChrissModel()
        small = model.estimate(1000, 1000000, 0.02, 0.01, 100)
        large = model.estimate(100000, 1000000, 0.02, 0.01, 100)
        assert large.total > small.total

    def test_higher_vol_more_timing_risk(self):
        model = AlmgrenChrissModel()
        low_vol = model.estimate(10000, 1000000, 0.01, 0.0, 100)
        high_vol = model.estimate(10000, 1000000, 0.05, 0.0, 100)
        # Almgren-Chriss: timing risk scales with volatility
        assert high_vol.timing_risk > low_vol.timing_risk


class TestSquareRoot:
    def test_basic_estimate(self):
        model = SquareRootModel()
        impact = model.estimate(10000, 1000000, 0.02, 0.01, 100)
        assert impact.temporary > 0
        assert impact.permanent > 0


class TestKyle:
    def test_basic_estimate(self):
        model = KyleModel()
        impact = model.estimate(10000, 1000000, 0.02, 0.01, 100)
        assert impact.total > 0


class TestCompositeImpact:
    def test_ensemble(self):
        model = CompositeImpactModel()
        impact = model.estimate(10000, 1000000, 0.02, 0.01, 100)
        assert impact.total > 0
        assert impact.participation_rate == 0.01


class TestExecutionCost:
    def test_buy_cost(self):
        calc = ExecutionCostCalculator()
        cost = calc.calculate(
            order_quantity=10000,
            price=100.0,
            side="buy",
            market_volume=1000000,
            volatility=0.02,
            spread=0.01,
        )
        assert cost.commission > 0
        assert cost.tax == 0  # No stamp tax on buy
        assert cost.total_cost > 0
        assert cost.total_cost_bps > 0

    def test_sell_has_stamp_tax(self):
        calc = ExecutionCostCalculator()
        cost = calc.calculate(
            order_quantity=10000,
            price=100.0,
            side="sell",
            market_volume=1000000,
            volatility=0.02,
        )
        assert cost.tax > 0  # Sell-side stamp tax


# ── Position Tests ──


class TestPosition:
    def test_update_price(self):
        pos = Position(symbol="SYM", quantity=100, avg_cost=100.0)
        pos.update_price(110.0)
        assert pos.unrealized_pnl == 1000.0
        assert pos.market_value == 11000.0

    def test_total_pnl(self):
        pos = Position(symbol="SYM", quantity=100, avg_cost=100.0, realized_pnl=500)
        pos.update_price(105.0)
        assert pos.total_pnl == 500 + 500  # realized + unrealized


# ── TickDataSource Tests ──


class TestTickDataSource:
    def test_from_dataframe(self):
        df = pd.DataFrame({
            "symbol": ["SYM"] * 3,
            "timestamp_ns": [1, 2, 3],
            "price": [100.0, 101.0, 102.0],
            "quantity": [100, 200, 300],
        })
        source = TickDataSource(data=df)
        ticks = list(source.stream())
        assert len(ticks) == 3
        assert ticks[0].price == 100.0

    def test_from_generator(self):
        def gen():
            for i in range(5):
                yield Tick(symbol="SYM", timestamp_ns=i, price=100.0 + i)

        source = TickDataSource(generator=gen)
        ticks = list(source.stream())
        assert len(ticks) == 5


# ── Execution Algorithm Tests ──


class TestTWAP:
    def test_generate_orders(self):
        algo = TWAPAlgorithm(n_slices=5)
        from quant_platform.execution.order_book import OrderBook, Side
        book = OrderBook("SYM")
        tick = Tick(symbol="SYM", timestamp_ns=1, price=100.0, quantity=10000)

        orders = algo.generate_orders(
            target_quantity=1000,
            symbol="SYM",
            side=Side.BUY,
            current_tick=tick,
            order_book=book,
            config=BacktestConfig(),
        )
        assert len(orders) == 1
        assert orders[0].quantity > 0


class TestVWAP:
    def test_generate_orders(self):
        algo = VWAPAlgorithm(target_participation=0.05)
        from quant_platform.execution.order_book import OrderBook, Side
        book = OrderBook("SYM")
        tick = Tick(symbol="SYM", timestamp_ns=1, price=100.0, quantity=100000)

        orders = algo.generate_orders(
            target_quantity=5000,
            symbol="SYM",
            side=Side.BUY,
            current_tick=tick,
            order_book=book,
            config=BacktestConfig(),
        )
        assert len(orders) == 1
        assert orders[0].quantity > 0


# ── TickBacktester Tests ──


class TestTickBacktester:
    def _make_ticks(self, n=100, symbol="SYM", start_price=100.0):
        """Generate synthetic ticks."""
        ticks = []
        price = start_price
        for i in range(n):
            price += np.random.randn() * 0.5
            price = max(price, 50)
            ticks.append(Tick(
                symbol=symbol,
                timestamp_ns=1000000000 + i * 1000000,
                price=round(price, 2),
                quantity=np.random.randint(100, 10000),
                bid=round(price - 0.01, 2),
                ask=round(price + 0.01, 2),
                volume=np.random.randint(100000, 1000000),
                volatility=0.02,
            ))
        return ticks

    def test_basic_run(self):
        ticks = self._make_ticks(50)
        source = TickDataSource(generator=lambda: iter(ticks))

        engine = TickBacktester(BacktestConfig(initial_capital=1_000_000))

        def strategy(tick, book, positions, capital):
            # Simple: buy 100 shares every 10 ticks
            if tick.timestamp_ns % 10000000 == 0 and capital > tick.price * 100:
                return [(tick.symbol, "buy", 100, "limit", tick.price)]
            return []

        engine.set_strategy(strategy)
        result = engine.run(source)

        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None
        assert result.orders_submitted >= 0

    def test_no_strategy(self):
        ticks = self._make_ticks(20)
        source = TickDataSource(generator=lambda: iter(ticks))

        engine = TickBacktester()
        result = engine.run(source)

        assert result.total_trades == 0
        assert result.orders_submitted == 0

    def test_result_metrics(self):
        ticks = self._make_ticks(100)
        source = TickDataSource(generator=lambda: iter(ticks))

        engine = TickBacktester(BacktestConfig(initial_capital=1_000_000))

        def strategy(tick, book, positions, capital):
            if tick.price < 98 and capital > 10000:
                return [(tick.symbol, "buy", 100, "limit", tick.price)]
            elif tick.price > 102 and tick.symbol in positions:
                return [(tick.symbol, "sell", 100, "limit", tick.price)]
            return []

        engine.set_strategy(strategy)
        result = engine.run(source)

        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
