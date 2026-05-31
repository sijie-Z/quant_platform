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
        n_blocks = self.horizon_days // block_size + 1

        terminal_values = []
        max_drawdowns = []
        annual_returns = []
        sharpe_ratios = []
        all_paths = np.zeros((self.n_simulations, self.horizon_days))

        for sim in range(self.n_simulations):
            # Sample blocks with replacement
            blocks = []
            for _ in range(n_blocks):
                start = self.rng.randint(0, max(1, n - block_size))
                blocks.append(ret_values[start:start + block_size])

            path_returns = np.concatenate(blocks)[:self.horizon_days]

            # Compute cumulative returns
            cum_returns = np.cumprod(1 + path_returns)
            terminal_value = cum_returns[-1]
            terminal_values.append(terminal_value)

            # Max drawdown
            peak = np.maximum.accumulate(cum_returns)
            dd = (cum_returns - peak) / peak
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
        df, loc, scale = sp_stats.t.fit(returns.dropna().values)
        df = np.clip(df, df_range[0], df_range[1])

        logger.info("Fitted Student-t: df=%.1f, loc=%.4f, scale=%.4f", df, loc, scale)

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

            peak = np.maximum.accumulate(cum_returns)
            dd = (cum_returns - peak) / peak
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
    ) -> dict:
        """Shuffle trade sequence to test if trade timing drives results."""
        from quant_platform.backtest.metrics import (
            annualized_return,
            annualized_volatility,
        )

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

        orig_arr = pd.Series(returns)
        orig_ret = float(annualized_return(orig_arr))
        orig_vol = float(annualized_volatility(orig_arr))
        orig_sharpe = orig_ret / orig_vol if orig_vol > 0 else 0.0

        shuffled_sharpes = np.zeros(n_shuffles)
        for i in range(n_shuffles):
            s = self.rng.permutation(returns)
            s_arr = pd.Series(s)
            s_ret = float(annualized_return(s_arr))
            s_vol = float(annualized_volatility(s_arr))
            shuffled_sharpes[i] = s_ret / s_vol if s_vol > 0 else 0.0

        cv = float(np.std(shuffled_sharpes) / max(abs(np.mean(shuffled_sharpes)), 1e-6))
        stability_score = float(np.clip(1.0 / (1.0 + cv), 0, 1))
        return {
            "n_trades": len(returns),
            "n_shuffles": n_shuffles,
            "original_sharpe": round(orig_sharpe, 4),
            "mean_shuffled": round(float(np.mean(shuffled_sharpes)), 4),
            "std_shuffled": round(float(np.std(shuffled_sharpes)), 4),
            "sharpe_5pct": round(float(np.percentile(shuffled_sharpes, 5)), 4),
            "sharpe_95pct": round(float(np.percentile(shuffled_sharpes, 95)), 4),
            "stability_score": round(stability_score, 4),
        }

