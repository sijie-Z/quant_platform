"""Vectorized multi-period backtesting engine.

Core of the quant platform: takes signals, portfolio weights, and price data,
then simulates strategy performance with realistic costs and constraints.

Design:
- Vectorized (not event-driven): suitable for daily/weekly/monthly multi-factor
  strategies where execution is at known rebalancing dates.
- Monthly rebalancing: weights computed on the last trading day of each month,
  executed at next day's close, held until next rebalance.
- All P&L is net of transaction costs.

Fill model and lots
-------------------
This is a **weight-based** engine. Quantities are never converted to shares,
so the configured ``lot_size`` (100 shares) is *not* enforced here; lot
rounding is the job of the tick/execution layer. Treat backtest weights as
"research notionals" and the engine as lot-agnostic by design (B8). The
tradability gate below *is* enforced per asset-day when the caller supplies
the panels.

Execution-day tradability
-------------------------
``run`` accepts the pipeline's per-date ``tradable_mask`` (suspension / ST /
listing / delisting), an optional ``universe_mask`` (index membership), plus
``limit_up`` and ``limit_down`` panels. Buys are blocked on non-tradable
days, on non-member days and at limit-up; sells are blocked on non-tradable
days and at limit-down but *not* on non-membership -- a name that leaves the
index must remain sellable. Blocked trades are carried to the next tradable
day; no cost or turnover is booked for them. Without the panels the engine
behaves exactly as before.

Cash constraint (R18)
---------------------
Execution is sell-first: the allowed sells settle, and buys are then funded
from real cash only, ``available_cash = max(0, 1 - sum(weights after sells))``.
When a sell is blocked (limit-down / suspension) while the target still buys,
the buys that would execute can exceed that cash; the engine scales every
buy down by one common factor instead of turning the blocked sell into
cost-free leverage. After execution ``sum(weights) <= 1 + 1e-9`` holds on
every day, and the unfilled buy deltas stay in the deferred queue. The
optimizer's ``max_weight`` remains a target-level constraint: it bounds the
rebalance target, not the held book, so price drift (or a blocked sell) can
still leave a single name above it between rebalances -- the engine does not
force an extra sell to clip it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.backtest.cost_model import CostModel
from quant_platform.backtest.metrics import all_metrics, annualization_years
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.portfolio.covariance import estimate_covariance
from quant_platform.portfolio.optimizers import (
    EqualWeightOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
)
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# ``backtest.benchmark`` values the engine can actually compute. Any other
# value would silently fall through to "use the caller's series", which is how
# `market_cap_weight` used to be accepted-but-not-implemented (B6).
_SUPPORTED_BENCHMARKS = ("equal_weight", "market_cap_weight")

# Tolerance on the cash-funded buy budget. The post-execution gross-exposure
# invariant the engine promises is ``sum(weights) <= 1 + 1e-9``; anything
# inside this tolerance is left untouched so the default (no blocked sells)
# path stays bit-identical to the pre-R18 engine.
_FUNDING_TOL = 1e-12


class BacktestEngine:
    """Vectorized multi-period backtesting engine for A-share strategies."""

    def __init__(
        self,
        initial_capital: float = 10_000_000,
        rebalance_frequency: str = "monthly",
        cost_model: CostModel | None = None,
        constraints: PortfolioConstraints | None = None,
        optimizer: str = "mean_variance",
        benchmark: str = "equal_weight",
        covariance_method: str = "ledoit_wolf",
        covariance_lookback: int = 252,
        covariance_ewma_half_life: int = 63,
        execution_timing: str = "next_close",
        risk_free_rate: float = 0.03,
        periods_per_year: float | None = None,
    ):
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.cost_model = cost_model or CostModel()
        self.constraints = constraints or PortfolioConstraints()
        self.optimizer_name = optimizer
        if benchmark not in _SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark: {benchmark!r}. Supported values: "
                f"{', '.join(_SUPPORTED_BENCHMARKS)}. Pass a pre-computed "
                "series only through benchmark_returns."
            )
        self.benchmark_type = benchmark
        self.covariance_method = covariance_method
        self.covariance_lookback = covariance_lookback
        self.covariance_ewma_half_life = covariance_ewma_half_life
        # B12: annualization basis and risk-free are explicit, not constants.
        # ``None`` for periods_per_year means "infer from the return index".
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # `next_close` (default): the signal is computed from rdate's close and
        # executed at the *next* trading day's close, which is what the module
        # docstring, data/ASHARE_PITFALLS.md and the T+1 notes have always said.
        # `signal_close` reproduces the previous behaviour, where the portfolio
        # was traded at the very close the signal was computed from, and exists
        # so the migration can be measured against it (see
        # `tools/execution_timing_migration.py`). Do not use it for research.
        if execution_timing not in ("next_close", "signal_close"):
            raise ValueError(f"Unknown execution_timing: {execution_timing}")
        self.execution_timing = execution_timing

        # Results
        self.daily_returns: pd.Series | None = None
        self.portfolio_values: pd.Series | None = None
        self.benchmark_returns: pd.Series | None = None
        self.weights_history: dict[pd.Timestamp, pd.Series] = {}
        self.unexecuted_signals: dict[pd.Timestamp, pd.Series] = {}
        self.turnover_history: pd.Series | None = None
        self.optimizer_outcomes: dict[pd.Timestamp, dict] = {}
        self.optimizer_fallback_count = 0
        self.skipped_rebalances = 0
        self.covariance_failures = 0
        # Rebalances where covariance estimation produced nothing and MVO ran
        # on an identity matrix. Kept separate from equal-weight fallbacks so
        # `effective_optimizer` names the right risk model (M27).
        self.identity_covariance_fallbacks = 0
        self._executed_signal_dates: set[pd.Timestamp] = set()
        # Sum of per-rebalance costs in return units. Reported in the summary
        # so the strategy gates can evaluate the configured cost-drag limit.
        self.total_cost_drag = 0.0
        # Execution days on which the cash constraint scaled buys down (R18).
        self.n_funding_scaled_days = 0

    @staticmethod
    def _compute_equal_weight_benchmark(returns: pd.DataFrame) -> pd.Series:
        """Per-date equal weight over the assets with a return that day.

        The old implementation filtered columns with a whole-sample
        `dropna(thresh=50%)`, so an asset's future listing history decided
        whether it entered the benchmark on early dates. Mean over each row
        skips NaN and uses exactly what was observable that day.
        """
        return returns.mean(axis=1)

    @staticmethod
    def _extract_market_cap_panel(financials) -> pd.DataFrame | None:
        """Pull a (date x asset) market-cap panel out of the `financials` input.

        Accepted shapes, in order:
        - a MultiIndex-column frame from ``financials.unstack("asset")``
          (columns are ``(field, asset)``; main.py passes this shape),
        - a long frame indexed by ``(date, asset)`` with a ``market_cap``
          column,
        - a plain (date x asset) market-cap panel.
        """
        if financials is None or not isinstance(financials, pd.DataFrame):
            return None
        if isinstance(financials.columns, pd.MultiIndex):
            fields = financials.columns.get_level_values(0)
            if "market_cap" not in fields:
                return None
            return financials["market_cap"]
        if "market_cap" in financials.columns:
            panel = financials["market_cap"]
            if isinstance(panel, pd.Series) and isinstance(
                panel.index, pd.MultiIndex,
            ):
                panel = panel.unstack("asset")
            if isinstance(panel, pd.DataFrame):
                return panel
        return financials

    @classmethod
    def _compute_market_cap_weight_benchmark(
        cls, returns: pd.DataFrame, financials,
    ) -> pd.Series:
        """Cap-weighted benchmark from point-in-time market caps (B6).

        Market caps are forward-filled over time only: on each date the
        benchmark uses the latest cap observed at or before that date, never a
        later observation. Names without a cap or without a return that day
        are excluded from that date's weights; a date with no usable caps
        yields NaN rather than a fabricated zero.
        """
        caps = cls._extract_market_cap_panel(financials)
        if caps is None or caps.empty:
            raise ValueError(
                "benchmark='market_cap_weight' requires a financials panel "
                "with market_cap data (date x asset)"
            )
        if not isinstance(caps.index, pd.DatetimeIndex):
            raise ValueError(
                "benchmark='market_cap_weight': financials must be indexed by date"
            )
        # Point-in-time: reindex to the union and forward-fill so a value only
        # becomes visible on/after its own date; then project onto the return
        # grid.
        all_dates = caps.index.union(returns.index).sort_values()
        caps = caps.reindex(all_dates).ffill().reindex(
            index=returns.index, columns=returns.columns,
        )
        usable = returns.notna() & caps.notna() & (caps > 0)
        weights = caps.where(usable)
        row_sum = weights.sum(axis=1)
        weights = weights.div(row_sum.where(row_sum > 0), axis=0)
        bench = (returns.fillna(0.0) * weights.fillna(0.0)).sum(axis=1)
        bench = bench.where(row_sum > 0)
        return bench.rename("benchmark_return")

    def run(
        self,
        signal: pd.DataFrame,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        sector_map: pd.Series,
        financials: pd.DataFrame | None = None,
        tradable_mask: pd.DataFrame | None = None,
        universe_mask: pd.DataFrame | None = None,
        limit_up: pd.DataFrame | None = None,
        limit_down: pd.DataFrame | None = None,
        daily_volume: pd.DataFrame | None = None,
        volatility: pd.DataFrame | None = None,
        risk_free_rate: float | None = None,
        periods_per_year: float | None = None,
        raw_prices: pd.DataFrame | None = None,
    ) -> dict:
        """Execute the backtest.

        Args:
            signal: (date x asset) alpha signal (higher = more attractive).
            prices: (date x asset) adjusted close prices.
            returns: (date x asset) daily returns.
            benchmark_returns: Series of benchmark daily returns.
            sector_map: Series mapping asset -> sector name.
            financials: financial data for the market-cap benchmark. Accepts
                the ``financials.unstack("asset")`` shape main.py passes, a
                long (date, asset) frame with a ``market_cap`` column, or a
                plain (date x asset) cap panel.
            tradable_mask: (date x asset) True where a new position is
                allowed. False on suspension / ST / listing / delisting days.
                Missing entries default to tradable.
            universe_mask: (date x asset) True where the asset is an index
                constituent. Gates buys only; sells are not restricted by
                membership so an index exit can still be liquidated. Missing
                entries default to member (True). ``None`` is equivalent to
                an all-True panel.
            limit_up: (date x asset) True where the asset closed limit-up.
                Buys are blocked there.
            limit_down: (date x asset) True where the asset closed
                limit-down. Sells are blocked there.
            daily_volume: (date x asset) traded **volume in shares** (the
                provider boundary unit). Converted to CNY with ``raw_prices``
                (unadjusted traded close) for the R8 capacity cap
                ``adv = (volume * raw_close).rolling(20, min_periods=5).mean()``
                and for the cost model's proportional/impact slippage models.
                ``prices`` is the adjusted research panel and must NOT be
                used as a CNY price; without ``raw_prices`` the capacity cap
                is disabled and the volume is dropped before the cost model
                (both paths log a warning).
            volatility: (date x asset) daily volatility estimate for the
                impact model. When omitted and an impact/proportional model
                is configured, a 20-day rolling std of ``returns`` shifted by
                one bar (observable at the execution close) is used.
            risk_free_rate: Annual risk-free override for this run (defaults
                to the constructor value).
            periods_per_year: Annualization override for this run (defaults to
                the constructor value; ``None`` infers from the return index).
            raw_prices: (date x asset) **unadjusted** close prices (CNY),
                e.g. ``DataPipeline.get_raw_close()``. Required to turn share
                volume into a CNY traded value. Only used when
                ``daily_volume`` is supplied.

        Returns:
            Dict with daily_returns, portfolio_values, benchmark_returns,
            weights_history (executed rebalances only), turnover_history, and
            summary metrics. ``summary["since_first_execution"]`` carries the
            same metric block computed from the first fill onward (B13);
            ``summary["n_rebalances"]`` counts executed rebalances (M13).
        """
        logger.info("Starting backtest: capital=%.0f, freq=%s, optimizer=%s",
                     self.initial_capital, self.rebalance_frequency, self.optimizer_name)

        # External callers derive signals from DataFrames that carry pandas
        # attrs (e.g. ``excluded_mask``); on pandas 3 those attrs propagate
        # into downstream selections and crash inside Series.nlargest. Clear
        # them at the engine boundary on a shallow copy (the caller's object
        # is left untouched).
        signal = signal.copy(deep=False)
        signal.attrs = {}

        # Pre-compute benchmark daily returns
        if self.benchmark_type == "equal_weight":
            self.benchmark_returns = self._compute_equal_weight_benchmark(returns)
        elif self.benchmark_type == "market_cap_weight":
            self.benchmark_returns = self._compute_market_cap_weight_benchmark(
                returns, financials,
            )
        else:  # pragma: no cover - constructor rejects unknown values
            self.benchmark_returns = benchmark_returns

        rebalance_dates = self._get_rebalance_dates(signal.index)
        logger.info("Backtest: %d rebalance dates from %s to %s (freq=%s)",
                     len(rebalance_dates),
                     str(rebalance_dates[0])[:10] if rebalance_dates else "N/A",
                     str(rebalance_dates[-1])[:10] if rebalance_dates else "N/A",
                     self.rebalance_frequency)

        self.weights_history = {}
        self.unexecuted_signals = {}
        self.optimizer_outcomes = {}
        self.optimizer_fallback_count = 0
        self.skipped_rebalances = 0
        self.covariance_failures = 0
        self.identity_covariance_fallbacks = 0
        prev_weights = None
        # One optimizer instance for the whole run, so its fallback counter
        # accumulates; a fresh instance per rebalance hid every fallback.
        optimizer = self._get_optimizer()

        # R8 cost/capacity knobs are implemented by the MVO optimizer only
        # (see portfolio/optimizers.py). Say so instead of silently ignoring
        # them when another optimizer is configured.
        if self.optimizer_name != "mean_variance" and (
            self.constraints.transaction_cost_bps > 0
            or self.constraints.turnover_penalty > 0
            or self.constraints.max_participation > 0
        ):
            logger.warning(
                "transaction_cost_bps/turnover_penalty/max_participation are "
                "implemented for the mean_variance optimizer only; "
                "optimizer=%s ignores them",
                self.optimizer_name,
            )

        # R8 capacity: ADV in CNY is the 20-session mean of volume (shares)
        # x **raw** close. ``prices`` is the adjusted research panel and is
        # not a CNY price (data/pipeline.get_factor_prices docstring), so a
        # share volume cannot be turned into a traded value without
        # ``raw_prices``; the cap is refused rather than computed on the
        # wrong price basis. Only evaluated when the cap is configured; the
        # default path never touches this.
        capacity_adv_panel = None
        capacity_adv_warned = False
        if self.constraints.max_participation > 0:
            if daily_volume is None:
                logger.warning(
                    "max_participation=%.3f is configured but run() received "
                    "no daily_volume panel; the capacity constraint is "
                    "disabled for this backtest",
                    self.constraints.max_participation,
                )
                capacity_adv_warned = True
            elif raw_prices is None:
                logger.warning(
                    "max_participation=%.3f is configured but run() received "
                    "no raw_prices; ``prices`` is the adjusted research "
                    "panel, not a CNY price, so share volume cannot be "
                    "converted to a CNY traded value. The capacity "
                    "constraint is disabled for this backtest (pass "
                    "raw_prices, e.g. pipeline.get_raw_close()).",
                    self.constraints.max_participation,
                )
                capacity_adv_warned = True
            else:
                volume_shares = daily_volume.reindex(
                    index=prices.index, columns=prices.columns,
                )
                raw_close = raw_prices.reindex(
                    index=prices.index, columns=prices.columns,
                )
                capacity_adv_panel = (
                    volume_shares * raw_close
                ).rolling(20, min_periods=5).mean()

        for i, rdate in enumerate(rebalance_dates):
            logger.debug("Rebalance %d/%d: %s", i + 1, len(rebalance_dates), str(rdate)[:10])

            sig = signal.loc[rdate].dropna()
            if len(sig) < 10:
                logger.debug("Rebalance %s: only %d signals — using equal-weight signal "
                             "(the configured optimizer still applies)",
                              str(rdate)[:10], len(sig))
                if rdate in prices.index:
                    valid_assets = prices.loc[rdate].dropna().index
                    if len(valid_assets) >= 10:
                        sig = pd.Series(1.0, index=valid_assets)
                if len(sig) < 10:
                    self.skipped_rebalances += 1
                    continue

            # Estimate covariance matrix.
            #
            # Only bars strictly before rdate may enter. `returns` is built as
            # `close.pct_change().shift(-1)` (data/pipeline.py), so the row
            # dated t is the return from close(t) to close(t+1) -- at rdate's
            # close the row dated rdate has not happened yet. The window used
            # to end at `lookback_end + 1`, which is inclusive of that row:
            # the optimizer was estimating risk from one day of its own
            # future. It was also one row too long (253 for a 252 lookback).
            prior_dates = returns.index[returns.index < rdate]
            if len(prior_dates) == 0:
                self.skipped_rebalances += 1
                continue
            lookback_end = returns.index.get_loc(prior_dates[-1])
            lookback_start = max(0, lookback_end + 1 - self.covariance_lookback)
            ret_window = returns.iloc[lookback_start:lookback_end + 1]

            # Filter to assets with sufficient history
            valid_assets = ret_window.dropna(axis=1, thresh=int(len(ret_window) * 0.5)).columns
            valid_assets = valid_assets.intersection(sig.index)
            sig = sig[valid_assets]

            if len(sig) < 10:
                self.skipped_rebalances += 1
                continue

            ret_window = ret_window[valid_assets]
            try:
                cov = estimate_covariance(
                    ret_window,
                    method=self.covariance_method,
                    lookback=min(self.covariance_lookback, len(ret_window)),
                    ewma_half_life=self.covariance_ewma_half_life,
                )
            except Exception as exc:
                # A silently swallowed failure here ran the whole backtest
                # on an identity covariance (MVO) or equal-weight fallback
                # (risk parity) while the summary still named the configured
                # risk model. The Ledoit-Wolf kernel used to fail to compile,
                # which is exactly how that happened.
                cov = None
                self.covariance_failures += 1
                logger.warning(
                    "Covariance estimation failed at %s (%s): %s",
                    str(rdate)[:10], self.covariance_method, exc,
                )

            # M27: MVO silently substitutes the identity matrix for a missing
            # covariance. Count that as a risk-model fallback so the summary
            # cannot claim a real covariance-driven optimizer ran.
            identity_cov = cov is None and self.optimizer_name == "mean_variance"
            if identity_cov:
                self.identity_covariance_fallbacks += 1

            # Optimize
            adv_for_optimizer = None
            aum_for_optimizer = None
            if self.constraints.max_participation > 0:
                # Decision-time portfolio value: the engine has not replayed
                # P&L yet at this point, so the documented AUM convention is
                # the run's initial capital (constant across rebalances).
                aum_for_optimizer = float(self.initial_capital)
                if capacity_adv_panel is not None and rdate in capacity_adv_panel.index:
                    adv_row = capacity_adv_panel.loc[rdate].reindex(sig.index)
                    if adv_row.notna().any() and bool((adv_row > 0).any()):
                        # NaN ADV entries are passed through: the optimizer
                        # keeps max_weight for those names and logs them.
                        adv_for_optimizer = adv_row
                    elif not capacity_adv_warned:
                        capacity_adv_warned = True
                        logger.warning(
                            "Rebalance %s: no positive ADV is available for "
                            "any candidate asset; optimizing without the "
                            "capacity cap", str(rdate)[:10],
                        )
            fallbacks_before = optimizer.fallback_count
            weights = optimizer.optimize(
                signal=sig,
                cov_matrix=cov,
                prices=prices.loc[rdate].reindex(sig.index) if rdate in prices.index else None,
                prev_weights=prev_weights,
                sector_map=sector_map.reindex(sig.index) if sector_map is not None else None,
                adv=adv_for_optimizer,
                aum=aum_for_optimizer,
            )
            fell_back = optimizer.fallback_count > fallbacks_before
            fallback_reason = optimizer.last_fallback_reason if fell_back else None
            if fell_back:
                self.optimizer_fallback_count += 1
                logger.warning(
                    "Rebalance %s: optimizer fallback #%d — %s",
                    str(rdate)[:10], self.optimizer_fallback_count, fallback_reason,
                )

            self.optimizer_outcomes[rdate] = {
                "optimizer": self.optimizer_name,
                "fallback": fell_back,
                "reason": fallback_reason,
                "identity_covariance": identity_cov,
                "n_assets": int((weights > 0).sum()),
                "gross_exposure": float(weights.sum()),
                "executed": False,
            }

            self.weights_history[rdate] = weights
            prev_weights = weights

        # Simulate daily P&L
        self._simulate_pnl(
            returns,
            tradable_mask=tradable_mask,
            universe_mask=universe_mask,
            limit_up=limit_up,
            limit_down=limit_down,
            daily_volume=daily_volume,
            volatility=volatility,
            raw_prices=raw_prices,
        )

        # M13: only rebalances that actually reached an execution day belong
        # in `weights_history` / `n_rebalances`. A signal whose date is the
        # last bar has no following trading day (dropped by the schedule) and
        # used to be reported as a held portfolio.
        executed = self._executed_signal_dates
        self.unexecuted_signals = {
            rdate: weights for rdate, weights in self.weights_history.items()
            if rdate not in executed
        }
        self.weights_history = {
            rdate: weights for rdate, weights in self.weights_history.items()
            if rdate in executed
        }
        for rdate, outcome in self.optimizer_outcomes.items():
            outcome["executed"] = rdate in executed

        rf = self.risk_free_rate if risk_free_rate is None else risk_free_rate
        ppy = self.periods_per_year if periods_per_year is None else periods_per_year

        # Compute metrics
        summary = all_metrics(
            self.daily_returns, self.benchmark_returns,
            periods_per_year=ppy, risk_free=rf,
        )
        n_rebalances = len(self.weights_history)
        summary["n_rebalances"] = n_rebalances
        summary["n_unexecuted_signals"] = len(self.unexecuted_signals)
        summary["optimizer"] = self.optimizer_name
        summary["initial_capital"] = self.initial_capital
        summary["n_optimizer_fallbacks"] = self.optimizer_fallback_count
        summary["optimizer_fallback_rate"] = (
            self.optimizer_fallback_count / n_rebalances if n_rebalances else 0.0
        )
        summary["n_skipped_rebalances"] = self.skipped_rebalances
        summary["n_covariance_failures"] = self.covariance_failures
        summary["n_identity_covariance_fallbacks"] = self.identity_covariance_fallbacks
        # R18: execution days on which the cash constraint scaled buys down
        # (only possible when a sell was blocked or the book carried no cash).
        # Counted per execution/retry day, not per rebalance: one pending
        # rebalance whose sell stays blocked retries (and is counted) daily.
        summary["n_funding_scaled_days"] = int(self.n_funding_scaled_days)
        if self.optimizer_fallback_count > 0:
            summary["effective_optimizer"] = (
                f"{self.optimizer_name}+equal_weight_fallback"
            )
        elif self.identity_covariance_fallbacks > 0:
            # MVO ran every affected rebalance on eye(n); say so instead of
            # reporting a mean-variance risk model that was not used.
            summary["effective_optimizer"] = f"{self.optimizer_name}(identity_cov)"
        else:
            summary["effective_optimizer"] = self.optimizer_name

        # B13: report the metrics the strategy actually earned once invested,
        # alongside the from-inception block. A month-end strategy sits in
        # cash for the first weeks while the benchmark moves, which distorts
        # Sharpe/vol/excess for short samples.
        first_exec = None
        if self.turnover_history is not None and len(self.turnover_history):
            first_exec = self.turnover_history.index.min()
        if first_exec is not None:
            since_returns = self.daily_returns.loc[first_exec:]
            since_bench = None
            if self.benchmark_returns is not None and len(self.benchmark_returns):
                since_bench = self.benchmark_returns.loc[
                    self.benchmark_returns.index >= first_exec
                ]
            summary["since_first_execution"] = all_metrics(
                since_returns, since_bench,
                periods_per_year=ppy, risk_free=rf,
            )
            summary["first_execution_date"] = str(first_exec.date())
            leading = self.daily_returns.loc[:first_exec]
            summary["n_leading_cash_days"] = int(
                (leading.iloc[:-1] == 0).sum()
            ) if len(leading) else 0

        years = annualization_years(self.daily_returns, ppy)
        summary["annual_turnover"] = (
            float(self.turnover_history.sum()) / years
            if years > 0 and self.turnover_history is not None
            and len(self.turnover_history) else 0.0
        )
        summary["total_cost_drag"] = float(self.total_cost_drag)
        summary["annual_cost_drag"] = (
            float(self.total_cost_drag) / years if years > 0 else 0.0
        )

        logger.info("Backtest complete: total_return=%.2f%%, sharpe=%.2f",
                     summary.get("total_return", 0) * 100,
                     summary.get("sharpe_ratio", 0))

        return {
            "daily_returns": self.daily_returns,
            "portfolio_values": self.portfolio_values,
            "benchmark_returns": self.benchmark_returns,
            "weights_history": self.weights_history,
            "unexecuted_signals": self.unexecuted_signals,
            "turnover_history": self.turnover_history,
            "summary": summary,
        }

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> list:
        """Get rebalancing dates based on frequency."""
        if self.rebalance_frequency == "daily":
            return list(dates)
        elif self.rebalance_frequency == "weekly":
            # Last trading day of each ISO week. Grouping by the *calendar*
            # year put a late-December session (ISO week 1 of the next year)
            # in the same bucket as the first week of the current January:
            # the early-January rebalance vanished, a spurious Dec-31 one
            # appeared, the list came back non-chronological, and the January
            # optimizer received the December target as prev_weights -- a
            # look-ahead through the turnover constraint (B2).
            iso = dates.isocalendar()
            grouped = dates.to_series().groupby([
                iso["year"].to_numpy(), iso["week"].to_numpy(),
            ])
            return sorted(pd.Timestamp(d) for d in grouped.last().values)
        elif self.rebalance_frequency == "monthly":
            # Last trading day of each month (using actual trading calendar)
            grouped = dates.to_series().groupby([dates.year, dates.month])
            return sorted(pd.Timestamp(d) for d in grouped.last().values)
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance_frequency}")

    def _get_optimizer(self):
        if self.optimizer_name == "equal_weight":
            return EqualWeightOptimizer(self.constraints)
        elif self.optimizer_name == "mean_variance":
            return MeanVarianceOptimizer(self.constraints)
        elif self.optimizer_name == "risk_parity":
            return RiskParityOptimizer(self.constraints)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _execution_schedule(
        self, returns: pd.DataFrame
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Map each signal date to the trading day whose close it executes at.

        The signal is computed *from* rdate's close, so it cannot also be
        traded at that same close -- the earliest fill the strategy could have
        got is the next trading day's close. Until the weights are applied the
        portfolio still holds yesterday's positions, and the traded value, its
        cost and the resulting turnover all belong to the execution day, not to
        the day the decision was made.

        `signal_close` returns (rdate, rdate), reproducing the previous
        behaviour exactly so the migration can be measured against it.
        """
        index = returns.index
        schedule: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for rdate in sorted(self.weights_history.keys()):
            if self.execution_timing == "signal_close":
                schedule.append((rdate, rdate))
                continue
            pos = index.get_indexer([rdate], method="ffill")[0]
            if pos < 0 or pos + 1 >= len(index):
                # No trading day follows the signal, so it cannot be executed.
                # Dropping it is the honest outcome; trading it at rdate's own
                # close is exactly the assumption being removed.
                logger.warning(
                    "Rebalance %s has no following trading day -- signal dropped "
                    "rather than executed at its own close",
                    str(rdate)[:10],
                )
                continue
            schedule.append((index[pos + 1], rdate))
        return schedule

    @staticmethod
    def _align_gate(panel, dates, columns, default: bool):
        """Reindex a boolean gate panel onto (dates x columns)."""
        if panel is None:
            return None
        return (
            panel.reindex(index=dates, columns=columns)
            .fillna(default)
            .astype(bool)
        )

    @staticmethod
    def _align_float(panel, dates, columns):
        """Reindex a numeric panel onto (dates x columns)."""
        if panel is None:
            return None
        return panel.reindex(index=dates, columns=columns).astype(float)

    def _simulate_pnl(
        self,
        returns: pd.DataFrame,
        tradable_mask: pd.DataFrame | None = None,
        universe_mask: pd.DataFrame | None = None,
        limit_up: pd.DataFrame | None = None,
        limit_down: pd.DataFrame | None = None,
        daily_volume: pd.DataFrame | None = None,
        volatility: pd.DataFrame | None = None,
        raw_prices: pd.DataFrame | None = None,
    ) -> None:
        """Simulate daily P&L with costs and an execution-day tradability gate.

        ``daily_volume`` is a share-volume panel (provider unit). When
        ``raw_prices`` (unadjusted CNY close) is supplied, their product is
        converted to causal CNY ADV using a 60-session median (minimum 20
        observations) shifted one session before being handed to the cost
        model. Without ``raw_prices`` the panel is dropped (warning) so the
        cost model falls back to its fixed slippage instead of treating share
        counts as CNY.

        Between executions, weights drift with price movements. On an
        execution date, the new weights take effect at that close, turnover is
        computed against the held weights, and the cost is charged for that
        day -- so `turnover_history` is indexed by execution date too.

        When ``tradable_mask`` / ``universe_mask`` / ``limit_up`` /
        ``limit_down`` are supplied (B1, R6b), each asset's trade is checked
        on the *execution* date, not the signal date: buys are blocked where
        the asset is not tradable, not an index member, or closed limit-up;
        sells are blocked where it is not tradable or closed limit-down --
        deliberately *not* by ``universe_mask``, so leaving the index never
        traps a held position. Blocked deltas are retried on following days
        until they fill or a newer rebalance supersedes them; no cost or
        turnover is booked for a blocked trade.

        R18: sells settle before buys and buys are capped first by the gross
        cash released, then by a conservative reserve for their estimated
        transaction costs. If a blocked sell leaves insufficient gross cash,
        the unfilled remainder stays deferred and the day is counted in
        ``n_funding_scaled_days``. Cost-only sizing does not create endless
        retries of an ideal fully-invested target. Executed positions and cash
        are subsequently normalized by NAV after the same single cost debit.
        """
        all_dates = returns.index.sort_values()
        columns = returns.columns
        schedule = self._execution_schedule(returns)

        gates_enabled = any(
            panel is not None
            for panel in (tradable_mask, universe_mask, limit_up, limit_down)
        )
        mask_panel = self._align_gate(tradable_mask, all_dates, columns, True)
        universe_panel = self._align_gate(
            universe_mask, all_dates, columns, True,
        )
        up_panel = self._align_gate(limit_up, all_dates, columns, False)
        down_panel = self._align_gate(limit_down, all_dates, columns, False)
        if gates_enabled:
            # Supplying one gate panel must not dereference the others the
            # caller omitted (the gate check indexes all of them). The
            # neutral values are "tradable"/"member" and "not at the limit".
            if mask_panel is None:
                mask_panel = pd.DataFrame(True, index=all_dates, columns=columns)
            if universe_panel is None:
                universe_panel = pd.DataFrame(
                    True, index=all_dates, columns=columns,
                )
            if up_panel is None:
                up_panel = pd.DataFrame(False, index=all_dates, columns=columns)
            if down_panel is None:
                down_panel = pd.DataFrame(False, index=all_dates, columns=columns)
        if daily_volume is not None and raw_prices is None:
            # A share panel cannot be handed to a cost model that expects
            # CNY. Rather than silently overstating participation by ~price,
            # drop the volume so the cost model falls back to its configured
            # fixed slippage and says so.
            logger.warning(
                "daily_volume is a share panel but raw_prices was not "
                "supplied; dropping the volume so the cost model falls back "
                "to fixed slippage. Pass raw_prices (unadjusted CNY close) "
                "to use volume-dependent slippage."
            )
            volume_panel = None
            price_panel = None
        else:
            volume_panel = self._align_float(daily_volume, all_dates, columns)
            # Only needed to turn share volume into CNY for the cost model;
            # skip the full-panel copy on the default (no-volume) path.
            price_panel = (
                self._align_float(raw_prices, all_dates, columns)
                if daily_volume is not None and raw_prices is not None
                else None
            )
            if volume_panel is not None and price_panel is not None:
                # Cost participation uses causal ADV, not the execution day's
                # realized volume. Row d contains only CNY amounts through
                # d-1; zero/missing results are handled per asset downstream.
                volume_panel = (
                    (volume_panel * price_panel)
                    .rolling(60, min_periods=20)
                    .median()
                    .shift(1)
                )

        vol_panel = self._align_float(volatility, all_dates, columns)
        if (
            vol_panel is None
            and getattr(self.cost_model, "slippage_model", "fixed") == "impact"
        ):
            # At close(d) only returns dated before d are observable (row d is
            # the close(d)->close(d+1) return). Shifted rolling std gives the
            # impact model a causal daily-volatility estimate.
            vol_panel = (
                returns.rolling(20, min_periods=5).std().shift(1).reindex(
                    index=all_dates, columns=columns,
                ).fillna(0.02)
            )

        self.total_cost_drag = 0.0
        self.n_funding_scaled_days = 0
        self._executed_signal_dates = set()
        capital = self.initial_capital
        current_weights = pd.Series(0.0, index=columns)
        daily_ret_list = []
        turnover_records = []

        next_idx = 0
        pending_rdate: pd.Timestamp | None = None
        pending_target: pd.Series | None = None
        pending_assets: pd.Index = pd.Index([])

        for date in all_dates:
            rebalance_cost = 0.0
            # At most one funding-scale warning per execution day (R18).
            funding_scaled_today = False

            # A scheduled rebalance starts on its execution date. If a previous
            # rebalance still has blocked deltas outstanding, the newer signal
            # supersedes it: its target was computed with older information.
            if next_idx < len(schedule) and date >= schedule[next_idx][0]:
                _, rdate = schedule[next_idx]
                if pending_rdate is not None and len(pending_assets):
                    logger.warning(
                        "Rebalance %s superseded by %s before its blocked trades "
                        "could execute (%d asset(s))",
                        str(pending_rdate)[:10], str(rdate)[:10], len(pending_assets),
                    )
                pending_rdate = rdate
                pending_target = self.weights_history.get(
                    rdate, current_weights,
                ).reindex(columns, fill_value=0.0)
                pending_assets = pending_target.index
                next_idx += 1

            if pending_rdate is not None and pending_target is not None:
                target = pending_target
                candidates = pending_assets
                delta = (target - current_weights).reindex(
                    candidates, fill_value=0.0,
                )

                if gates_enabled:
                    allowed = pd.Series(True, index=candidates)
                    buys = delta.index[delta > 0]
                    sells = delta.index[delta < 0]
                    if len(buys):
                        # Buys require both tradability and index membership.
                        allowed.loc[buys] = (
                            mask_panel.loc[date].reindex(buys)
                            & universe_panel.loc[date].reindex(buys)
                            & ~up_panel.loc[date].reindex(buys)
                        )
                    if len(sells):
                        # Sells check tradability and limit-down only: an
                        # index exit must not prevent closing the position.
                        allowed.loc[sells] = (
                            mask_panel.loc[date].reindex(sells)
                            & ~down_panel.loc[date].reindex(sells)
                        )
                    allowed = allowed.fillna(True).astype(bool)
                else:
                    allowed = pd.Series(True, index=candidates)

                executable_delta = delta.where(allowed, 0.0)
                turnover = float(executable_delta.abs().sum()) / 2.0

                # Enforce the turnover cap at execution time too. Optimizer
                # constraints only see target-level turnover, while realized
                # turnover also includes weight drift and candidate-set
                # changes; without this clamp the reported 0.44 could exceed
                # the configured 0.30 even when the optimizer "succeeded".
                # Initial deployment from cash is exempt: the cap governs
                # rebalancing an existing portfolio, and clamping the first
                # purchase would make a fully-invested mandate unreachable.
                if (
                    float(current_weights.sum()) > 1e-12
                    and self.constraints.max_turnover < 1.0
                    and turnover > self.constraints.max_turnover
                ):
                    scale = self.constraints.max_turnover / turnover
                    executable_delta = executable_delta * scale
                    turnover = float(executable_delta.abs().sum()) / 2.0

                # R18: sells settle first; buys are then funded from real cash
                # only. `available_cash` is what the allowed sells leave
                # uninvested. When a sell is blocked (limit-down/suspension)
                # while the target still buys, the requested buys can exceed
                # it; scaling every buy by one common factor preserves the
                # target's relative ordering and never lifts a weight (so the
                # optimizer's max_weight still bounds the rebalance target).
                # The old code applied the buys unconditionally and clipped
                # negative cash to zero in the drift step, i.e. it ran
                # cost-free hidden leverage. The unfilled buy remainder is
                # returned to the deferred queue below.
                funding_scale = 1.0
                total_buys = 0.0
                available_cash = 0.0
                if len(candidates):
                    sell_delta = executable_delta.clip(upper=0.0)
                    buy_delta = executable_delta.clip(lower=0.0)
                    total_buys = float(buy_delta.sum())
                    if total_buys > 0.0:
                        available_cash = max(
                            0.0,
                            1.0 - (
                                float(current_weights.sum())
                                + float(sell_delta.sum())
                            ),
                        )
                        if total_buys > available_cash + _FUNDING_TOL:
                            # Cash below the tolerance is zero. Without this,
                            # a fully-invested book whose drifted sum is
                            # 0.9999999999999999 leaves ~1e-16 of "cash" and
                            # the scaled buys become dust orders, each paying
                            # the cost model's per-order minimum commission.
                            if available_cash <= _FUNDING_TOL:
                                available_cash = 0.0
                            funding_scale = available_cash / total_buys
                            executable_delta = (
                                sell_delta + buy_delta * funding_scale
                            )
                            turnover = (
                                float(executable_delta.abs().sum()) / 2.0
                            )
                            if not funding_scaled_today:
                                funding_scaled_today = True
                                self.n_funding_scaled_days += 1
                                logger.warning(
                                    "Rebalance %s: buy orders at %s scaled to "
                                    "%.4f of the requested amount by available "
                                    "cash (%.4f available vs %.4f requested)",
                                    str(pending_rdate)[:10], str(date)[:10],
                                    funding_scale, available_cash, total_buys,
                                )

                if turnover > 0 and capital > 0:
                    volume_row = None
                    if volume_panel is not None and date in volume_panel.index:
                        volume_row = volume_panel.loc[date]
                        if not volume_row.notna().any():
                            volume_row = None

                    vol_row = None
                    if vol_panel is not None and date in vol_panel.index:
                        vol_row = vol_panel.loc[date]
                        if not vol_row.notna().any():
                            vol_row = None

                    # Estimate once at the gross-funded size. If fees make the
                    # buys unaffordable, subtract that entire deficit from the
                    # buy amount and recompute. Every production cost component
                    # is non-increasing when buys shrink, so using the original
                    # (larger) cost is a conservative deterministic reserve;
                    # no iterative fixed-point solve is needed. R18's separate
                    # funding_scale still governs deferral of gross-cash blocks.
                    for cost_pass in range(2):
                        # The cost model receives actual per-asset buy/sell CNY
                        # notionals, so every component is charged on the right
                        # side and the returned values are fractions of capital.
                        buy_value = executable_delta.clip(lower=0.0)
                        sell_value = (-executable_delta).clip(lower=0.0)
                        buy_notional = buy_value * capital
                        sell_notional = sell_value * capital
                        costs = self.cost_model.compute_costs(
                            float(buy_notional.sum() + sell_notional.sum()),
                            buy_turnover=buy_notional,
                            sell_turnover=sell_notional,
                            daily_volume=volume_row,
                            volatility=vol_row,
                            date=date,
                            portfolio_value=capital,
                        )
                        rebalance_cost = float(np.sum(costs))

                        executed_buys = float(buy_value.sum())
                        cash_after_costs = (
                            available_cash - executed_buys - rebalance_cost
                        )
                        if (
                            cost_pass == 1
                            or executed_buys <= 0.0
                            or cash_after_costs >= 0.0
                        ):
                            break

                        affordable_buys = max(
                            0.0, executed_buys + cash_after_costs,
                        )
                        if affordable_buys <= _FUNDING_TOL:
                            affordable_buys = 0.0
                        executable_delta = (
                            executable_delta.clip(upper=0.0)
                            + buy_value * (affordable_buys / executed_buys)
                        )
                        turnover = float(executable_delta.abs().sum()) / 2.0

                    self.total_cost_drag += rebalance_cost

                    if turnover > 0:
                        # `executable_delta` may cover only a retry subset;
                        # label-aware addition keeps every other position.
                        current_weights = current_weights.add(
                            executable_delta, fill_value=0.0,
                        ).clip(lower=0.0)
                        turnover_records.append((date, turnover))
                        self._executed_signal_dates.add(pending_rdate)
                        # Record the affordable trades, not the pre-clamp target.
                        self.weights_history[pending_rdate] = current_weights.copy()

                if gates_enabled:
                    blocked = allowed[~allowed & (delta.abs() > 1e-12)].index
                else:
                    blocked = pd.Index([])
                if funding_scale < 1.0:
                    # Buys the cash constraint could not fund stay deferred
                    # and are retried like gate-blocked trades, instead of
                    # being dropped (which would silently abandon the target).
                    residual = (target - current_weights).reindex(
                        candidates, fill_value=0.0,
                    )
                    funding_blocked = residual[residual > 1e-12].index
                    if len(funding_blocked):
                        blocked = blocked.union(funding_blocked)
                pending_assets = blocked
                if len(pending_assets) == 0:
                    if (
                        pending_rdate is not None
                        and float((target - current_weights).abs().sum()) < 1e-9
                    ):
                        self._executed_signal_dates.add(pending_rdate)
                    pending_rdate = None
                    pending_target = None

            daily_ret_assets = returns.loc[date].reindex(current_weights.index, fill_value=0.0)
            portfolio_return = (current_weights * daily_ret_assets).sum()
            daily_ret = portfolio_return - rebalance_cost
            daily_ret_list.append(daily_ret)

            capital = capital * (1 + daily_ret)

            # Drift the held weights with prices. The old code reused the
            # target vector forever, which silently rebalanced to target every
            # day: a winner was sold down daily, drift turnover never happened,
            # and every reported backtest was a daily-rebalanced portfolio.
            if current_weights.sum() > 1e-12:
                growth = current_weights * (1.0 + daily_ret_assets.fillna(0.0))
                # Cash also pays the execution cost. The NAV deduction above
                # and this denominator are two views of the same debit, not
                # two charges: growth + cash now equals 1 + daily_ret exactly.
                cash_weight = max(
                    0.0,
                    1.0 - float(current_weights.sum()) - rebalance_cost,
                )
                total_value = float(growth.sum()) + cash_weight
                if total_value > 1e-12:
                    current_weights = growth / total_value

        self.daily_returns = pd.Series(daily_ret_list, index=all_dates, name="strategy_return")
        self.portfolio_values = self.initial_capital * (1 + self.daily_returns).cumprod()
        self.portfolio_values.name = "portfolio_value"
        if turnover_records:
            dates_idx, values = zip(*turnover_records, strict=True)
            self.turnover_history = pd.Series(
                values, index=pd.DatetimeIndex(dates_idx), name="turnover"
            )
        else:
            self.turnover_history = pd.Series(dtype=float)
