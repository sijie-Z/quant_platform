"""80d Reversal Paper Trader — 基于 RQ5b 已验证回测逻辑.

使用与 RQ5b 完全相同的回测方法 (自定义 loop, 非 BacktestEngine),
在调仓日执行选股和买卖, 每日对持仓逐日盯市.

每日输出: 净值曲线, 持仓明细, 交易记录.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SIGNAL_H = 40
HOLD_H = 80
SELECT_PCT = 0.20
VOL_PERCENTILE = 0.70
INITIAL_CAPITAL = 10_000_000
COST_BPS = 15


def compound(x):
    return np.prod(1 + x) - 1 if len(x) > 0 else 0


@dataclass
class Trade:
    date: str
    action: str  # enter | exit | skip
    n_stocks: int = 0
    capital: float = 0.0


class ReversalPaperTrader:
    """80d Reversal Paper Trader — 基于 RQ5b 验证的回测."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self._returns: pd.DataFrame | None = None
        self._prices: pd.DataFrame | None = None
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []

    def load_data(self, returns: pd.DataFrame, prices: pd.DataFrame):
        self._returns = returns
        self._prices = prices

    def run(self):
        if self._returns is None or self._prices is None:
            raise ValueError("No data")

        returns = self._returns
        prices = self._prices
        dates = returns.index
        n_stocks = len(returns.columns)
        n_select = max(1, int(n_stocks * SELECT_PCT))
        cost = COST_BPS / 10000

        # Vol filter
        market_vol = returns.mean(axis=1).rolling(20).std()
        vol_thresh = market_vol.quantile(VOL_PERCENTILE)

        # 信号
        past_ret = returns.rolling(SIGNAL_H, min_periods=SIGNAL_H).apply(compound, raw=True)
        rebal_dates = [dates[i] for i in range(HOLD_H, len(dates) - SIGNAL_H, HOLD_H)]

        # 持仓记录: {(entry_date, exit_date): {asset: (shares, entry_price)}}
        holdings: dict[tuple, dict] = {}
        cash = self.initial_capital

        for rdate in rebal_dates:
            date_str = str(rdate)[:10]
            day_idx = dates.get_loc(rdate)

            # ── 平到期仓位 ──
            matured = [k for k in holdings if k[1] <= rdate]
            for k in matured:
                h = holdings.pop(k)
                for asset, (shares, ep) in h.items():
                    if asset in prices.columns and rdate in prices.index:
                        xp = prices.loc[rdate, asset]
                        if not pd.isna(xp) and xp > 0:
                            cash += shares * xp * (1 - cost)
                self.trades.append(Trade(date_str, "exit", n_stocks=len(h)))

            # ── Vol filter ──
            if rdate in market_vol.index and market_vol[rdate] > vol_thresh:
                self.trades.append(Trade(date_str, "skip"))
                self.equity_curve.append(self._snapshot(rdate, cash, holdings, prices))
                continue

            # ── 选股 ──
            pr = past_ret.iloc[day_idx]
            valid = pr.dropna().sort_values()
            if len(valid) < n_select:
                self.trades.append(Trade(date_str, "skip"))
                self.equity_curve.append(self._snapshot(rdate, cash, holdings, prices))
                continue

            selected = valid.head(n_select).index.tolist()
            n = len(selected)

            # ── 买入 ──
            cap_per = cash / n
            buy_cost = cap_per * cost * n
            cash -= buy_cost

            new_holding = {}
            for asset in selected:
                if asset in prices.columns and rdate in prices.index:
                    ep = prices.loc[rdate, asset]
                    if not pd.isna(ep) and ep > 0:
                        shares = int(cap_per / ep / 100) * 100
                        if shares >= 100:
                            new_holding[asset] = (shares, ep)
                            cash -= shares * ep

            if new_holding:
                exit_date = dates[min(day_idx + HOLD_H, len(dates) - 1)]
                holdings[(rdate, exit_date)] = new_holding
                self.trades.append(Trade(date_str, "enter", n_stocks=len(new_holding), capital=cap_per * len(new_holding)))

            self.equity_curve.append(self._snapshot(rdate, cash, holdings, prices))

        # ── 最后: 平所有 ──
        for k, h in holdings.items():
            for asset, (shares, ep) in h.items():
                cash += shares * ep * (1 - cost)
        holdings.clear()

        self.trades.append(Trade(str(dates[-1])[:10], "exit"))
        self.equity_curve.append(self._snapshot(dates[-1], cash, holdings, prices))
        logger.info("Done. Final: %.2f", cash)
        return cash

    def _snapshot(self, date, cash, holdings, prices):
        """计算当日总净值."""
        pos_value = 0.0
        for k, h in holdings.items():
            for asset, (shares, ep) in h.items():
                if asset in prices.columns and date in prices.index:
                    px = prices.loc[date, asset]
                    if not pd.isna(px) and px > 0:
                        pos_value += shares * px
        return {"date": str(date)[:10], "cash": cash, "positions": pos_value, "total": cash + pos_value}

    def report(self) -> str:
        if len(self.equity_curve) < 10:
            return "Insufficient data"

        df = pd.DataFrame(self.equity_curve)
        totals = df["total"].values
        daily_rets = pd.Series([totals[t] / totals[t - 1] - 1 for t in range(1, len(totals))])

        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 1e-10 else 0
        ann = (1 + daily_rets.mean()) ** 252 - 1
        cum = totals
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        mdd = dd.min()
        total_ret = totals[-1] / totals[0] - 1
        n_enter = sum(1 for t in self.trades if t.action == "enter")

        return (
            f"\n{'=' * 65}\n"
            f"  80d Reversal Paper Trader — 运行报告\n"
            f"{'=' * 65}\n"
            f"  初始资金:     {self.initial_capital:>12,.0f}\n"
            f"  最终资金:     {totals[-1]:>12,.0f}\n"
            f"  总收益:       {total_ret * 100:>+10.2f}%\n"
            f"  年化收益:     {ann * 100:>+10.2f}%\n"
            f"  Sharpe:       {sharpe:>+12.4f}\n"
            f"  最大回撤:     {mdd * 100:>10.2f}%\n"
            f"  交易次数:     {n_enter:>10d}\n"
            f"  成本:         {COST_BPS} bps each way\n"
            f"{'=' * 65}"
        )
