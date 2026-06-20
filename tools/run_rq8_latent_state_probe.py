#!/usr/bin/env python3
"""
RQ8 v2: Latent-State Control Probe

问题: 80d reversal alpha 在低维控制策略下不可恢复.
      更高级的隐状态模型 (HMM, Wavelet, Kalman) 能否捕捉?

实验:
  - Baseline: 80d fixed-grid (Sharpe +0.45)
  - Model A: HMM regime states → condition trade
  - Model B: Wavelet phase → condition trade
  - Model C: Kalman filter latent MR state → condition trade

不是 performance hunting。是 control hypothesis test。
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import pandas as pd
import numpy as np
from hmmlearn import hmm
import pywt

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("rq8")

REAL_START = "2018-01-01"
REAL_END = "2025-12-31"
SIGNAL_H = 40
HOLD_H = 80
SELECT_PCT = 0.20


def load():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date=REAL_START, end_date=REAL_END,
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    return pipeline.returns, pipeline.get_close(), pipeline.benchmark


def compound(x):
    return np.prod(1 + x) - 1 if len(x) > 0 else 0


def compute_signal(returns, signal_h=SIGNAL_H):
    """原始反转信号 + 过去收益."""
    past_ret = returns.rolling(signal_h, min_periods=signal_h).apply(compound, raw=True)
    return past_ret


# ── Helpers ──

def backtest_80d(past_ret, returns, trade_mask=None):
    """80d 反转回测。可选的 trade_mask: bool Series, True=允许交易."""
    n_stocks = len(returns.columns)
    n_select = max(1, int(n_stocks * SELECT_PCT))
    dates = returns.index
    indices = list(range(HOLD_H, len(dates) - SIGNAL_H, HOLD_H))

    rets = []
    skipped = 0
    for i in indices:
        if trade_mask is not None:
            rdate = dates[i]
            if rdate in trade_mask.index and not trade_mask[rdate]:
                skipped += 1
                continue

        pr = past_ret.iloc[i]
        valid = pr.dropna().sort_values()
        if len(valid) < n_select:
            continue
        selected = valid.head(n_select)
        end = i + HOLD_H
        if end >= len(dates):
            break
        hr = returns.iloc[i + 1:end + 1]
        if len(hr) == 0:
            continue
        ret = (hr[selected.index].mean(axis=1) + 1).prod() - 1
        rets.append(ret)

    ps = pd.Series(rets)
    if len(ps) < 3:
        return {"sharpe": np.nan, "ann_ret": 0, "mdd": 0, "n": len(ps), "skipped": skipped}
    af = np.sqrt(252 / HOLD_H)
    sharpe = ps.mean() / ps.std() * af if ps.std() > 1e-10 else 0
    ann = (1 + ps.mean()) ** (252 / HOLD_H) - 1
    cum = (1 + ps).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"sharpe": sharpe, "ann_ret": ann, "mdd": mdd if not np.isnan(mdd) else 0,
            "n": len(ps), "skipped": skipped}


# ── Model A: HMM Regime ──

def hmm_trade_mask(returns, n_states=2, lookback=20):
    """HMM regime model. 学习市场状态的隐 Markov 链.

    训练: 等权市场日收益序列
    输出: 当最可能状态为"均值回归态"(根据状态均值符号判定)时允许交易.
    """
    market = returns.mean(axis=1).dropna().values.reshape(-1, 1)

    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                            random_state=42, n_iter=100)
    model.fit(market)
    states = model.predict(market)

    # 识别均值回归态: 状态均值接近0 (震荡) 且自相关低
    state_means = np.array([market[states == s].mean() for s in range(n_states)])
    state_stds = np.array([market[states == s].std() for s in range(n_states)])

    # MR state = state with mean closest to 0 and high std (oscillatory)
    mr_score = np.abs(state_means) / (state_stds + 1e-8)
    mr_state = np.argmin(mr_score)

    dates = returns.mean(axis=1).index
    trade_mask = pd.Series(states == mr_state, index=dates[:len(states)])
    return trade_mask


# ── Model B: Wavelet Phase ──

def wavelet_trade_mask(returns, wavelet="db4", level=4):
    """Wavelet-based phase proxy.

    小波分解等权市场收益序列.
    提取低频近似分量 (对应 ~40-80d 周期).
    当近似分量处于上升趋势(相位有利)时允许交易.
    """
    market = returns.mean(axis=1).dropna().values
    dates = returns.mean(axis=1).dropna().index

    # 小波分解
    coeffs = pywt.wavedec(market, wavelet, level=level)
    cA = coeffs[0]  # 低频近似

    # 重采样回原始长度
    cA_up = pywt.upcoef("a", cA, wavelet, take=len(market), level=level)

    # 相位 = 近似分量的方向变化
    phase = np.sign(np.gradient(cA_up))

    # 映射到日期
    trade_mask = pd.Series(phase > 0, index=dates[:len(phase)])
    return trade_mask


# ── Model C: Kalman Filter ──

def kalman_trade_mask(returns, delta=0.01, vt=0.5, wt=0.1):
    """Kalman filter on mean-reversion state.

    Simple local level model:
      x_t = x_{t-1} + w_t    (state: latent MR level)
      y_t = x_t + v_t         (obs: market return)

    当 x_t < 0 (市场低于均衡, 未来可能反弹) → 允许交易.
    """
    market = returns.mean(axis=1).dropna()
    dates = market.index

    # Kalman filter
    n = len(market)
    x = np.zeros(n)
    P = np.zeros(n)
    # Initialize
    x[0] = market.iloc[0]
    P[0] = 1.0

    for t in range(1, n):
        # Predict
        x_pred = x[t - 1]
        P_pred = P[t - 1] + wt

        # Update
        K = P_pred / (P_pred + vt)
        x[t] = x_pred + K * (market.iloc[t] - x_pred)
        P[t] = (1 - K) * P_pred

    # 当隐含 MR 状态低于阈值时交易
    trade_mask = pd.Series(x < -delta, index=dates)
    return trade_mask


# ── Main ──

def main():
    print("=" * 80)
    print("  RQ8 v2: Latent-State Control Probe")
    print("  验证隐状态模型能否恢复 80d reversal alpha")
    print("=" * 80)

    returns, prices, benchmark = load()
    past_ret = compute_signal(returns)
    logger.info("Data: %d days, %d assets", len(returns), len(returns.columns))

    # ── Baseline ──
    base = backtest_80d(past_ret, returns)
    print(f"\nBaseline (no mask, 80d fixed-grid):")
    print(f"  Sharpe={base['sharpe']:.4f}  AnnRet={base['ann_ret']*100:.2f}%  "
          f"MDD={base['mdd']*100:.2f}%  trades={base['n']}")

    # ── HMM ──
    mask_a = hmm_trade_mask(returns, n_states=2)
    r_a = backtest_80d(past_ret, returns, mask_a)
    print(f"\n[A] HMM regime:")
    if not np.isnan(r_a["sharpe"]):
        print(f"  Sharpe={r_a['sharpe']:.4f}  AnnRet={r_a['ann_ret']*100:.2f}%  "
              f"MDD={r_a['mdd']*100:.2f}%  trades={r_a['n']}  skipped={r_a['skipped']}")
    else:
        print(f"  insufficient trades")

    # ── Wavelet ──
    mask_b = wavelet_trade_mask(returns)
    r_b = backtest_80d(past_ret, returns, mask_b)
    print(f"\n[B] Wavelet phase:")
    if not np.isnan(r_b["sharpe"]):
        print(f"  Sharpe={r_b['sharpe']:.4f}  AnnRet={r_b['ann_ret']*100:.2f}%  "
              f"MDD={r_b['mdd']*100:.2f}%  trades={r_b['n']}  skipped={r_b['skipped']}")
    else:
        print(f"  insufficient trades")

    # ── Kalman ──
    mask_c = kalman_trade_mask(returns)
    r_c = backtest_80d(past_ret, returns, mask_c)
    print(f"\n[C] Kalman filter:")
    if not np.isnan(r_c["sharpe"]):
        print(f"  Sharpe={r_c['sharpe']:.4f}  AnnRet={r_c['ann_ret']*100:.2f}%  "
              f"MDD={r_c['mdd']*100:.2f}%  trades={r_c['n']}  skipped={r_c['skipped']}")
    else:
        print(f"  insufficient trades")

    # ── 结论 ──
    print()
    print("=" * 80)
    all_results = [("Baseline", base), ("HMM", r_a), ("Wavelet", r_b), ("Kalman", r_c)]
    improved = any(not np.isnan(r["sharpe"]) and r["sharpe"] > base["sharpe"] + 0.05
                   for _, r in all_results[1:])
    if improved:
        best = max([(n, r) for n, r in all_results if not np.isnan(r["sharpe"])],
                   key=lambda x: x[1]["sharpe"])
        print(f"  Latent-state improves signal: YES")
        print(f"  Best model: {best[0]} (Sharpe {best[1]['sharpe']:.4f})")
        print(f"  vs baseline: {base['sharpe']:.4f}")
        print(f"  => NO-GO ZONE is model-limited, not fundamental")
    else:
        print(f"  Latent-state improves signal: NO")
        print(f"  All models <= baseline")
        print(f"  => NO-GO ZONE confirmed within tested model class")
    print("=" * 80)

    # 保存
    rows = [{"model": n, "sharpe": r["sharpe"], "ann_ret": r["ann_ret"],
             "mdd": r["mdd"], "n_trades": r["n"], "skipped": r["skipped"]}
            for n, r in all_results]
    pd.DataFrame(rows).to_csv("results/rq8_latent_state_probe.csv", index=False)


if __name__ == "__main__":
    main()
