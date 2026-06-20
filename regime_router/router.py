"""RegimeRouterStub — 最小可交易系统。

输入: 市场数据 (prices, returns)
输出: 组合权重

当前: v1 stub — 固定 S=40/H=80 反转, regime hardcoded
下一步: v1.1 — 加入简单 regime 分类器
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.cost_model import CostModel
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class RegimeRouterStub:
    """Regime Router v1 stub — fixed reversal strategy.

    当前行为:
    - regime classifier: 返回 "unknown" (未实现)
    - operator: 固定 S=40, H=80
    - signal: 过去 40 日反转 → 买入跌幅最大 20% 股票
    - execution: 每 80 日调仓, 等权
    - 100% 仓位 (不降仓)

    设计原则:
    - 无 ML, 无优化
    - 参数硬编码 (不可配置)
    - 结果可完全复现
    """

    # 固定参数 (来自 Market Structure Map v1)
    DEFAULT_SIGNAL_H = 40     # 反转信号窗口
    DEFAULT_HOLD_H = 80       # 持有期
    SELECT_PCT = 0.20          # 选择底部 20%
    MAX_WEIGHT = 0.05          # 单票上限 5%
    INITIAL_CAPITAL = 10_000_000

    def __init__(self):
        self._last_regime = "unknown"

    def classify_regime(self, benchmark_returns: pd.Series | None = None) -> str:
        """Regime classifier stub — 始终返回 'unknown'。

        v1.0: hardcoded. v1.1 将替换为规则分类器.
        """
        return "unknown"

    def select_operator(self, regime: str) -> tuple[int, int]:
        """根据 regime 选择 (signal_h, hold_h).

        Stub: 始终返回 (40, 80) — 全周期最优单元.
        """
        return (self.DEFAULT_SIGNAL_H, self.DEFAULT_HOLD_H)

    def generate_signal(self, returns: pd.DataFrame, signal_h: int) -> pd.DataFrame:
        """生成反转信号。

        计算过去 signal_h 日累计收益 → 选择跌幅最大的股票.
        信号值范围 [-0.5, +0.5], 跌幅最大 → 正值 (超配).
        """
        past_ret = returns.rolling(signal_h, min_periods=signal_h).apply(
            lambda x: np.prod(1 + x) - 1 if len(x) == signal_h else np.nan,
            raw=True,
        )
        # 反转信号: 过去跌越多 → 信号值越大
        signal = -past_ret.rank(axis=1, pct=True)
        signal = signal - 0.5  # 中心化到 [-0.5, 0.5]
        return signal

    def build_portfolio(self, signal_row: pd.Series) -> pd.Series:
        """从单日信号构建等权组合。

        选择信号最强的 top 20% (跌幅最大的股票), 等权配置.
        """
        valid = signal_row.dropna().sort_values(ascending=False)
        n_select = max(1, int(len(valid) * self.SELECT_PCT))
        selected = valid.head(n_select)

        weights = pd.Series(0.0, index=signal_row.index)
        w = min(1.0 / n_select, self.MAX_WEIGHT)
        for asset in selected.index:
            weights[asset] = w
        # 归一化
        weights = weights / weights.sum()
        return weights

    def run_backtest(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark_returns: pd.Series | None = None,
        signal_h: int | None = None,
        hold_h: int | None = None,
    ) -> dict:
        """运行完整回测。

        使用 BacktestEngine (与协议一致).
        """
        signal_h = signal_h or self.DEFAULT_SIGNAL_H
        hold_h = hold_h or self.DEFAULT_HOLD_H

        logger.info("RegimeRouterStub: S=%d H=%d", signal_h, hold_h)

        signal = self.generate_signal(returns, signal_h)

        # 使用协议标准参数  (与 RQ5b / Discovery v3 一致)
        constraints = PortfolioConstraints(
            long_only=True,
            max_weight=self.MAX_WEIGHT,
            max_sector_exposure=0.30,
            max_turnover=0.30,
            lot_size=100,
        )
        cost_model = CostModel(
            commission=0.0003,
            stamp_tax=0.001,
            slippage=0.0005,
            slippage_model="fixed",
        )
        # 持有期转为等效调仓频率
        freq_map = {5: "weekly", 10: "weekly", 20: "monthly", 40: "monthly",
                    60: "monthly", 80: "monthly", 120: "monthly"}
        freq = freq_map.get(hold_h, "monthly")

        engine = BacktestEngine(
            initial_capital=self.INITIAL_CAPITAL,
            rebalance_frequency=freq,
            cost_model=cost_model,
            constraints=constraints,
            optimizer="equal_weight",
            benchmark="equal_weight",
        )
        results = engine.run(
            signal=signal,
            prices=prices,
            returns=returns,
            benchmark_returns=benchmark_returns,
            sector_map=None,
            financials=None,
        )
        return results

    def summary(self, results: dict) -> str:
        """生成可读摘要."""
        s = results.get("summary", {})
        lines = [
            "=" * 60,
            "  RegimeRouterStub — 回测结果",
            "=" * 60,
            f"  Sharpe:      {s.get('sharpe_ratio', 0):.4f}",
            f"  年化收益:    {s.get('annual_return', 0)*100:.2f}%",
            f"  年化波动:    {s.get('annual_volatility', 0)*100:.2f}%",
            f"  最大回撤:    {s.get('max_drawdown', 0)*100:.2f}%",
            f"  总收益:      {s.get('total_return', 0)*100:.2f}%",
            f"  调仓次数:    {s.get('n_rebalances', 0)}",
            "-" * 60,
        ]
        return "\n".join(lines)
