"""Tests for risk.factor_risk — Euler factor risk decomposition."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.risk.factor_risk import (
    estimate_factor_betas,
    factor_contribution_summary,
    factor_risk_decomposition,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    assets = [f"stock_{i}" for i in range(50)]

    # Create correlated factor-returns structure
    factor_momentum = pd.DataFrame(
        np.random.normal(0, 1, (300, 50)), index=dates, columns=assets,
    )
    factor_value = pd.DataFrame(
        np.random.normal(0, 1, (300, 50)), index=dates, columns=assets,
    )

    # Returns driven by factors: r = 0.3*momentum + 0.2*value + noise
    noise = np.random.normal(0, 0.01, (300, 50))
    returns = pd.DataFrame(
        0.3 * factor_momentum.values + 0.2 * factor_value.values + noise,
        index=dates, columns=assets,
    )

    factors = {"momentum": factor_momentum, "value": factor_value}
    portfolio_returns = returns.mean(axis=1)
    return returns, factors, portfolio_returns


def _orthogonal_factor_data(n=200, rho=0.9, factor_scale=0.01, idio_scale=0.005):
    """Factor returns whose sample moments are known in closed form.

    ``z1 = cos(2*pi*k/n)``, ``z2 = sin(2*pi*k/n)`` and ``z3 = cos(4*pi*k/n)``
    have zero mean, squared norm ``n/2`` and are mutually orthogonal. Hence
    every scaled basis vector has sample variance ``scale**2 * n / (2*(n-1))``
    (ddof=1), ``Cov(f1, f2) = rho * Var(f1)`` and all covariances with the
    idiosyncratic series are zero.
    """
    k = np.arange(n)
    z1 = np.cos(2 * np.pi * k / n)
    z2 = np.sin(2 * np.pi * k / n)
    z3 = np.cos(4 * np.pi * k / n)
    f1 = factor_scale * z1
    f2 = factor_scale * (rho * z1 + np.sqrt(1.0 - rho**2) * z2)
    alpha = idio_scale * z3
    index = pd.date_range("2021-01-01", periods=n, freq="B")
    return index, f1, f2, alpha


def _numpy_euler_reference(portfolio_returns, factor_betas, factor_returns):
    """Independent numpy computation of the Euler risk shares.

    Applies the documented alignment (forward-fill betas, missing factor
    returns as zero) but computes every variance/covariance with numpy on the
    sample where betas exist, following
    ``RC_j = Cov(c_j, TF) / Var(P)`` and
    ``idio = (Var(alpha) + 2*Cov(TF, alpha)) / Var(P)``.
    """
    dates = portfolio_returns.index
    contributions = {}
    for name, beta in factor_betas.items():
        beta_aligned = beta.reindex(dates).ffill()
        factor_aligned = factor_returns[name].reindex(dates).fillna(0.0)
        contributions[name] = beta_aligned * factor_aligned

    total_factor = sum(contributions.values())
    alpha = portfolio_returns - total_factor

    valid = portfolio_returns.notna() & total_factor.notna() & alpha.notna()
    p = portfolio_returns[valid].to_numpy(dtype=float)
    tf = total_factor[valid].to_numpy(dtype=float)
    a = alpha[valid].to_numpy(dtype=float)

    total_var = np.var(p, ddof=1)
    shares = {}
    for name, c in contributions.items():
        c_arr = c[valid].to_numpy(dtype=float)
        shares[name] = float(np.cov(c_arr, tf, ddof=1)[0, 1] / total_var)
    shares["idio"] = float(
        (np.var(a, ddof=1) + 2.0 * np.cov(tf, a, ddof=1)[0, 1]) / total_var
    )
    return shares


class TestEstimateFactorBetas:
    def test_returns_betas(self, sample_data):
        returns, factors, _ = sample_data
        betas = estimate_factor_betas(returns, factors, min_history=100)
        assert set(betas) == set(factors)
        for name in factors:
            assert isinstance(betas[name], pd.Series)
            # One daily cross-sectional regression per date after the warm-up.
            assert len(betas[name]) == len(returns) - 100
            assert betas[name].index[0] == returns.index[100]
            assert betas[name].index.is_monotonic_increasing

    def test_beta_values_reasonable(self, sample_data):
        returns, factors, _ = sample_data
        betas = estimate_factor_betas(returns, factors, min_history=100)
        # The fixture was generated with exposures 0.3 (momentum) and
        # 0.2 (value); a correctly specified daily OLS recovers them.
        assert betas["momentum"].mean() == pytest.approx(0.3, abs=0.01)
        assert betas["value"].mean() == pytest.approx(0.2, abs=0.01)
        for beta_series in betas.values():
            assert beta_series.abs().max() < 10

    def test_window_parameter_was_renamed(self, sample_data):
        # M25: ``window`` promised a rolling estimation window but the
        # implementation is a daily cross-sectional regression with a warm-up.
        returns, factors, _ = sample_data
        with pytest.raises(TypeError):
            estimate_factor_betas(returns, factors, window=100)


class TestFactorRiskDecomposition:
    def test_decomposition(self, sample_data):
        returns, factors, portfolio_returns = sample_data
        betas = estimate_factor_betas(returns, factors, min_history=100)
        factor_returns = {name: f.mean(axis=1) for name, f in factors.items()}
        result = factor_risk_decomposition(portfolio_returns, betas, factor_returns)

        for key in (
            "factor_contributions",
            "alpha_series",
            "total_factor_return",
            "factor_risk_share",
            "idiosyncratic_risk_share",
            "factor_annual_return",
            "alpha_annual_return",
            "alpha_volatility",
            "r_squared",
            "total_risk",
            "factor_risk",
            "idiosyncratic_risk",
        ):
            assert key in result
        assert set(result["factor_risk_share"]) == set(factors)
        if not np.isnan(result["r_squared"]):
            assert 0 <= result["r_squared"] <= 1

        # Euler decomposition: factor shares + idiosyncratic bucket == 1.
        shares = list(result["factor_risk_share"].values())
        shares.append(result["idiosyncratic_risk_share"])
        assert abs(sum(shares) - 1.0) < 1e-9

        # Independent numpy reference on the same estimated-beta sample.
        expected = _numpy_euler_reference(portfolio_returns, betas, factor_returns)
        for name in factors:
            assert result["factor_risk_share"][name] == pytest.approx(
                expected[name], abs=1e-12,
            )
        assert result["idiosyncratic_risk_share"] == pytest.approx(
            expected["idio"], abs=1e-12,
        )

        # Risk levels cover the estimated-beta sample used by the shares.
        beta_dates = betas["momentum"].index
        assert result["total_risk"] == pytest.approx(
            portfolio_returns.reindex(beta_dates).std() * np.sqrt(252),
            rel=1e-12,
        )

    def test_summary(self, sample_data):
        returns, factors, portfolio_returns = sample_data
        betas = estimate_factor_betas(returns, factors, min_history=100)
        factor_returns = {name: f.mean(axis=1) for name, f in factors.items()}
        result = factor_risk_decomposition(portfolio_returns, betas, factor_returns)
        summary = factor_contribution_summary(result)

        assert isinstance(summary, list)
        assert len(summary) == len(factors) + 1
        for row in summary:
            assert {"factor", "risk_share_pct", "annual_return_bps", "contribution"} <= set(row)

        pcts = [row["risk_share_pct"] for row in summary]
        assert pcts == sorted(pcts, reverse=True)
        # Rounded percentages reproduce the Euler total (100%).
        assert sum(pcts) == pytest.approx(100.0, abs=0.05)

        by_name = {row["factor"]: row for row in summary}
        for name in factors:
            assert by_name[name]["contribution"] == "systematic"
        alpha_row = by_name["Alpha (idiosyncratic)"]
        assert alpha_row["contribution"] == "alpha"
        assert alpha_row["risk_share_pct"] == pytest.approx(
            result["idiosyncratic_risk_share"] * 100, abs=0.01,
        )


class TestEulerRiskDecomposition:
    @pytest.mark.parametrize("rho", [0.9, -0.9])
    def test_shares_match_hand_computed_closed_form(self, rho):
        n, factor_scale, idio_scale = 200, 0.01, 0.005
        index, f1, f2, alpha = _orthogonal_factor_data(
            n, rho, factor_scale, idio_scale,
        )
        portfolio_returns = pd.Series(f1 + f2 + alpha, index=index)
        betas = {
            "f1": pd.Series(1.0, index=index),
            "f2": pd.Series(1.0, index=index),
        }
        factor_returns = {
            "f1": pd.Series(f1, index=index),
            "f2": pd.Series(f2, index=index),
        }

        result = factor_risk_decomposition(portfolio_returns, betas, factor_returns)

        # Hand-derived: with V = s^2 * n / (2(n-1)) and Va = a^2 * n / (2(n-1)),
        # Var(f1) = Var(f2) = V, Cov(f1, f2) = rho*V, Cov(f_j, alpha) = 0, so
        #     Var(P) = 2(1+rho)V + Va
        #     RC_1 = RC_2 = (1+rho)V / Var(P)
        #     idio = Va / Var(P)
        v = factor_scale**2 * n / (2 * (n - 1))
        v_idio = idio_scale**2 * n / (2 * (n - 1))
        total_var = 2 * (1 + rho) * v + v_idio
        expected_factor_share = (1 + rho) * v / total_var
        expected_idio_share = v_idio / total_var

        assert result["factor_risk_share"]["f1"] == pytest.approx(
            expected_factor_share, abs=1e-12,
        )
        assert result["factor_risk_share"]["f2"] == pytest.approx(
            expected_factor_share, abs=1e-12,
        )
        assert result["idiosyncratic_risk_share"] == pytest.approx(
            expected_idio_share, abs=1e-12,
        )

        shares = list(result["factor_risk_share"].values())
        shares.append(result["idiosyncratic_risk_share"])
        assert abs(sum(shares) - 1.0) < 1e-9
        assert max(shares) <= 1.0

        # Regression guard: the old marginal-variance formula Var(c_j)/Var(P)
        # ignored covariance; for correlated factors it differs materially
        # (and for rho=-0.9 it would report a share of 2.22).
        marginal_share = v / total_var
        assert abs(result["factor_risk_share"]["f1"] - marginal_share) > 1e-6

    def test_negative_correlation_does_not_inflate_shares(self):
        n, factor_scale, idio_scale = 200, 0.01, 0.005
        index, f1, f2, alpha = _orthogonal_factor_data(
            n, rho=-0.9, factor_scale=factor_scale, idio_scale=idio_scale,
        )
        portfolio_returns = pd.Series(f1 + f2 + alpha, index=index)
        betas = {
            "f1": pd.Series(1.0, index=index),
            "f2": pd.Series(1.0, index=index),
        }
        factor_returns = {
            "f1": pd.Series(f1, index=index),
            "f2": pd.Series(f2, index=index),
        }

        result = factor_risk_decomposition(portfolio_returns, betas, factor_returns)

        shares = list(result["factor_risk_share"].values())
        shares.append(result["idiosyncratic_risk_share"])
        assert all(0.0 <= share <= 1.0 for share in shares)
        assert abs(sum(shares) - 1.0) < 1e-9
