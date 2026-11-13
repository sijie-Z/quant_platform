"""Monte Carlo simulation for strategy robustness testing.

Two approaches:
1. Bootstrap: resample historical returns with replacement
2. Parametric: fit distribution (Student-t) and simulate

Used to estimate:
- Confidence intervals for strategy metrics
- Probability of exceeding target returns
- Worst-case scenarios beyond historical data
- Strategy robustness across market regimes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class MonteCarloSimulator:
    """Monte Carlo simulation engine for strategy analysis.

    Generates thousands of synthetic return paths to estimate
    the distribution of strategy outcomes.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        horizon_days: int = 252,
        seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.horizon_days = horizon_days
        self.rng = np.random.RandomState(seed)

    def bootstrap_simulation(
        self,
        returns: pd.Series,
        block_size: int = 21,
    ) -> dict:
        """Block bootstrap simulation.

        Resamples blocks of returns to preserve autocorrelation structure.
        Block size of ~21 (1 month) captures momentum/mean-reversion patterns.

        Args:
            returns: historical daily returns
            block_size: number of days per block

        Returns:
            dict with simulation results and statistics
        """
        n = len(returns)
        ret_values = returns.values
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size!r}")
        n_blocks = self.horizon_days // block_size + 1
        # Start indices 0..n-block_size inclusive. The old high endpoint
        # `n - block_size` was exclusive, so the final block (and therefore
        # the most recent return) could never be sampled; use the inclusive
        # range and fall back to the whole series when it is shorter than one
        # block.
        max_start = n - block_size

        terminal_values = []
        max_drawdowns = []
        annual_returns = []
        sharpe_ratios = []
        all_paths = np.zeros((self.n_simulations, self.horizon_days))

        for sim in range(self.n_simulations):
            # Sample blocks with replacement
            blocks = []
            for _ in range(n_blocks):
                start = self.rng.randint(0, max_start + 1) if max_start > 0 else 0
                blocks.append(ret_values[start:start + block_size])

            path_returns = np.concatenate(blocks)[:self.horizon_days]

            # Compute cumulative returns
            cum_returns = np.cumprod(1 + path_returns)
            terminal_value = cum_returns[-1]
            terminal_values.append(terminal_value)

            # Max drawdown measured from the initial wealth of 1.0. Starting
            # the peak scan at 1+r0 made a first-period loss invisible.
            wealth = np.concatenate(([1.0], cum_returns))
            peak = np.maximum.accumulate(wealth)
            dd = (wealth - peak) / peak
            max_drawdowns.append(float(dd.min()))

            # Annualized return
            ann_ret = terminal_value ** (252 / self.horizon_days) - 1
            annual_returns.append(ann_ret)

            # Sharpe
            if path_returns.std() > 0:
                sharpe = path_returns.mean() / path_returns.std() * np.sqrt(252)
            else:
                sharpe = 0
            sharpe_ratios.append(sharpe)

            all_paths[sim] = cum_returns

        terminal_values = np.array(terminal_values)
        annual_returns = np.array(annual_returns)
        max_drawdowns = np.array(max_drawdowns)
        sharpe_ratios = np.array(sharpe_ratios)

        return {
            "method": "bootstrap",
            "n_simulations": self.n_simulations,
            "horizon_days": self.horizon_days,
            "block_size": block_size,
            "terminal_value": {
                "mean": float(terminal_values.mean()),
                "median": float(np.median(terminal_values)),
                "std": float(terminal_values.std()),
                "p5": float(np.percentile(terminal_values, 5)),
                "p25": float(np.percentile(terminal_values, 25)),
                "p75": float(np.percentile(terminal_values, 75)),
                "p95": float(np.percentile(terminal_values, 95)),
            },
            "annual_return": {
                "mean": float(annual_returns.mean()),
                "median": float(np.median(annual_returns)),
                "std": float(annual_returns.std()),
                "p5": float(np.percentile(annual_returns, 5)),
                "p95": float(np.percentile(annual_returns, 95)),
                "prob_positive": float((annual_returns > 0).mean()),
                "prob_10pct": float((annual_returns > 0.10).mean()),
            },
            "max_drawdown": {
                "mean": float(max_drawdowns.mean()),
                "median": float(np.median(max_drawdowns)),
                "worst": float(max_drawdowns.min()),
                "p5": float(np.percentile(max_drawdowns, 5)),
                "p95": float(np.percentile(max_drawdowns, 95)),
                "prob_20pct": float((max_drawdowns < -0.20).mean()),
                "prob_30pct": float((max_drawdowns < -0.30).mean()),
            },
            "sharpe": {
                "mean": float(sharpe_ratios.mean()),
                "median": float(np.median(sharpe_ratios)),
                "std": float(sharpe_ratios.std()),
                "p5": float(np.percentile(sharpe_ratios, 5)),
                "p95": float(np.percentile(sharpe_ratios, 95)),
                "prob_positive": float((sharpe_ratios > 0).mean()),
            },
            "paths": all_paths.tolist()[:50],  # Store first 50 paths for visualization
            "terminal_values": terminal_values.tolist(),
            "annual_returns": annual_returns.tolist(),
            "max_drawdowns": max_drawdowns.tolist(),
        }

    def parametric_simulation(
        self,
        returns: pd.Series,
        df_range: tuple = (3, 30),
    ) -> dict:
        """Parametric simulation using fitted Student-t distribution.

        Student-t captures fat tails better than normal distribution,
        which is critical for risk management in A-share markets.

        Args:
            returns: historical daily returns
            df_range: range of degrees of freedom to fit

        Returns:
            dict with simulation results
        """
        # Fit Student-t distribution
        sample = returns.dropna().values
        df_raw, loc, scale_raw = sp_stats.t.fit(sample)
        df = float(np.clip(df_raw, df_range[0], df_range[1]))

        # Clipping df without touching the scale changes the distribution's
        # variance: Var(t(df)) = scale^2 * df/(df-2), so a fitted df of 2.3
        # clipped to 3 shrinks the simulated std by ~36% and materially
        # understates fat-tail risk. Rescale so the simulated distribution
        # keeps the fitted distribution's standard deviation (or the sample
        # standard deviation when the fitted df has infinite variance).
        if df_raw > 2:
            target_std = float(scale_raw) * np.sqrt(df_raw / (df_raw - 2))
        else:
            target_std = float(np.std(sample, ddof=1))
        if df > 2 and np.isfinite(target_std) and target_std > 0:
            scale = float(target_std / np.sqrt(df / (df - 2)))
        else:
            scale = float(scale_raw)

        logger.info(
            "Fitted Student-t: df=%.1f (raw %.1f), loc=%.4f, scale=%.4f (raw %.4f)",
            df, df_raw, loc, scale, scale_raw,
        )

        terminal_values = []
        max_drawdowns = []
        annual_returns = []
        all_paths = np.zeros((self.n_simulations, self.horizon_days))

        for sim in range(self.n_simulations):
            # Generate from fitted distribution
            sim_returns = sp_stats.t.rvs(df, loc=loc, scale=scale,
                                          size=self.horizon_days,
                                          random_state=self.rng)

            cum_returns = np.cumprod(1 + sim_returns)
            terminal_values.append(cum_returns[-1])

            # Peak scan includes the initial wealth 1.0 (a first-period loss
            # must count toward max drawdown).
            wealth = np.concatenate(([1.0], cum_returns))
            peak = np.maximum.accumulate(wealth)
            dd = (wealth - peak) / peak
            max_drawdowns.append(float(dd.min()))

            ann_ret = cum_returns[-1] ** (252 / self.horizon_days) - 1
            annual_returns.append(ann_ret)

            all_paths[sim] = cum_returns

        terminal_values = np.array(terminal_values)
        annual_returns = np.array(annual_returns)
        max_drawdowns = np.array(max_drawdowns)

        return {
            "method": "parametric",
            "n_simulations": self.n_simulations,
            "horizon_days": self.horizon_days,
            "fitted_distribution": {
                "type": "student_t",
                "df": round(df, 2),
                "loc": round(loc, 6),
                "scale": round(scale, 6),
                "df_raw": round(float(df_raw), 2),
                "scale_raw": round(float(scale_raw), 6),
                "skew": round(float(sp_stats.t.stats(df, loc=loc, scale=scale, moments='s')), 4),
                "kurtosis": round(float(sp_stats.t.stats(df, loc=loc, scale=scale, moments='k')), 4),
            },
            "terminal_value": {
                "mean": float(terminal_values.mean()),
                "median": float(np.median(terminal_values)),
                "std": float(terminal_values.std()),
                "p5": float(np.percentile(terminal_values, 5)),
                "p95": float(np.percentile(terminal_values, 95)),
            },
            "annual_return": {
                "mean": float(annual_returns.mean()),
                "median": float(np.median(annual_returns)),
                "std": float(annual_returns.std()),
                "p5": float(np.percentile(annual_returns, 5)),
                "p95": float(np.percentile(annual_returns, 95)),
                "prob_positive": float((annual_returns > 0).mean()),
                "prob_10pct": float((annual_returns > 0.10).mean()),
            },
            "max_drawdown": {
                "mean": float(max_drawdowns.mean()),
                "worst": float(max_drawdowns.min()),
                "p5": float(np.percentile(max_drawdowns, 5)),
                "prob_20pct": float((max_drawdowns < -0.20).mean()),
                "prob_30pct": float((max_drawdowns < -0.30).mean()),
            },
            "paths": all_paths.tolist()[:50],
            "terminal_values": terminal_values.tolist(),
        }
    # ------------------------------------------------------------------
    # Trade-shuffle Monte Carlo  (inspired by Jesse)
    # ------------------------------------------------------------------

    def analyze_trade_sequence(
        self,
        trades: list[dict],
        n_shuffles: int = 1000,
        cost_per_trade: float = 0.0,
    ) -> dict:
        """Shuffle the trade sequence and score order-dependent outcomes.

        A trade-shuffle test is only informative if the scored statistics
        depend on the *order* of the trades. The old implementation scored the
        compound annualized return and the standard deviation of per-trade
        returns -- both permutation invariant -- so every shuffle produced the
        same value (std_shuffled=0, stability_score=1.0 on every run) and it
        annualized trade-level returns with 252, inflating Sharpe by up to
        ~15.9x.

        Each permutation is now scored by:

        - **max drawdown** of the compounded equity path, with the peak scan
          starting at the initial wealth of 1.0 (so a loss on the first trade
          counts);
        - **terminal wealth**, after ``cost_per_trade`` (round-trip fees +
          slippage as a fraction of capital) is charged on every trade:
          ``r_net = (1 + r) * (1 - cost) - 1``;
        - the **per-trade Sharpe ratio**, ``mean(r_net)/std(r_net)``, with no
          252 annualization. It is reported for reference but is
          permutation-invariant by construction; the order-sensitive
          distribution is the max-drawdown one, which drives
          ``stability_score``.

        Args:
            trades: list of dicts with a ``return_pct`` per trade.
            n_shuffles: number of random permutations.
            cost_per_trade: round-trip cost per trade (0.002 = 20bp).

        Returns:
            dict with original statistics, shuffled max-drawdown /
            terminal-wealth / Sharpe distributions and a stability score.
        """
        if cost_per_trade < 0:
            raise ValueError(f"cost_per_trade must be >= 0, got {cost_per_trade!r}")

        returns = np.array(
            [t.get('return_pct', 0) for t in trades if 'return_pct' in t],
            dtype=float,
        )
        if len(returns) < 10:
            return {
                "error": f"Need >=10 trades, got {len(returns)}",
                "original_sharpe": None,
                "stability_score": None,
            }

        # Net trade returns after fees/slippage.
        net_returns = (1.0 + returns) * (1.0 - cost_per_trade) - 1.0
        n = len(net_returns)

        def _path_stats(path: np.ndarray) -> tuple[float, float, float]:
            """(max drawdown from wealth 1.0, terminal wealth, per-trade Sharpe)."""
            wealth = np.concatenate(([1.0], np.cumprod(1.0 + path)))
            peak = np.maximum.accumulate(wealth)
            max_dd = float(((wealth - peak) / peak).min())
            terminal = float(wealth[-1])
            vol = float(np.std(path, ddof=1)) if n > 1 else 0.0
            sharpe = float(np.mean(path) / vol) if vol > 1e-12 else 0.0
            return max_dd, terminal, sharpe

        orig_mdd, orig_terminal, orig_sharpe = _path_stats(net_returns)

        mdd_shuffled = np.zeros(n_shuffles)
        terminal_shuffled = np.zeros(n_shuffles)
        sharpe_shuffled = np.zeros(n_shuffles)
        for i in range(n_shuffles):
            perm = self.rng.permutation(net_returns)
            mdd_shuffled[i], terminal_shuffled[i], sharpe_shuffled[i] = (
                _path_stats(perm)
            )

        # Stability is dispersion of the path-dependent outcome (max
        # drawdown) across orderings: a strategy whose drawdown depends
        # heavily on the order of its trades is not stable.
        mdd_scale = max(abs(float(np.mean(mdd_shuffled))), 1e-6)
        cv = float(np.std(mdd_shuffled) / mdd_scale)
        stability_score = float(np.clip(1.0 / (1.0 + cv), 0, 1))

        return {
            "n_trades": n,
            "n_shuffles": n_shuffles,
            "cost_per_trade": float(cost_per_trade),
            "original_sharpe": round(orig_sharpe, 4),
            "original_max_drawdown": round(orig_mdd, 6),
            "original_terminal_wealth": round(orig_terminal, 6),
            "max_drawdown": {
                "mean": round(float(np.mean(mdd_shuffled)), 6),
                "std": round(float(np.std(mdd_shuffled)), 6),
                "worst": round(float(np.min(mdd_shuffled)), 6),
                "p5": round(float(np.percentile(mdd_shuffled, 5)), 6),
                "p95": round(float(np.percentile(mdd_shuffled, 95)), 6),
                "original": round(orig_mdd, 6),
            },
            "terminal_wealth": {
                "mean": round(float(np.mean(terminal_shuffled)), 6),
                "std": round(float(np.std(terminal_shuffled)), 6),
                "p5": round(float(np.percentile(terminal_shuffled, 5)), 6),
                "p95": round(float(np.percentile(terminal_shuffled, 95)), 6),
                "original": round(orig_terminal, 6),
            },
            "sharpe": {
                # Unannualized per-trade Sharpe (permutation invariant).
                "mean": round(float(np.mean(sharpe_shuffled)), 4),
                "std": round(float(np.std(sharpe_shuffled)), 4),
                "p5": round(float(np.percentile(sharpe_shuffled, 5)), 4),
                "p95": round(float(np.percentile(sharpe_shuffled, 95)), 4),
                "original": round(orig_sharpe, 4),
            },
            "stability_score": round(stability_score, 4),
        }

