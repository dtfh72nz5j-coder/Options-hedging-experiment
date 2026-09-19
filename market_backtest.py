"""Research question: does the GBM-simulated hedging error distribution
actually describe what happens if you delta-hedge against real historical
stock prices?

We take the exact same hedging engine (hedging.py) used throughout this
project and run it on real historical SPY windows instead of simulated GBM
paths, then compare the resulting P&L distribution to the matched GBM
simulation. Any difference is model risk made visible: GBM has no jumps,
constant volatility, and thin tails - the real world has none of those
guarantees.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from market_data import load_cached
from hedging import run_delta_hedge
from simulation import simulate_gbm_paths

S0, K, T, r, sigma = 100, 100, 1.0, 0.04, 0.20
N_STEPS = 252
STEP_DAYS = 5  # stride between overlapping historical windows


def build_rolling_windows(prices, n_steps=N_STEPS, step=STEP_DAYS, s0=S0):
    """Overlapping n_steps-day windows of real prices, each rescaled to start
    at `s0` so every window is directly comparable to a K=100 ATM call."""
    values = prices["close"].to_numpy()
    n = len(values)
    windows = [
        values[start:start + n_steps + 1] * (s0 / values[start])
        for start in range(0, n - n_steps, step)
    ]
    return np.array(windows)


def summarise(pnl):
    return {
        "Mean P&L": pnl.mean(),
        "Std P&L": pnl.std(),
        "RMSE": np.sqrt(np.mean(pnl**2)),
        "5th pct": np.percentile(pnl, 5),
        "1st pct": np.percentile(pnl, 1),
        "Skew": pd.Series(pnl).skew(),
        "Excess kurtosis": pd.Series(pnl).kurt(),
    }


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    prices = load_cached("SPY")
    real_paths = build_rolling_windows(prices)
    times = np.linspace(0, T, N_STEPS + 1)
    n_windows = real_paths.shape[0]
    n_independent = len(prices) // N_STEPS

    print(f"Built {n_windows} overlapping 1-year windows from "
          f"{prices.index.min().date()} to {prices.index.max().date()} "
          f"(~{n_independent} statistically independent years of data)")

    log_rets = np.diff(np.log(real_paths), axis=1)
    realised_vols = log_rets.std(axis=1) * np.sqrt(252)
    print(f"Realised vol across windows: mean={realised_vols.mean():.2%}, "
          f"std={realised_vols.std():.2%}, min={realised_vols.min():.2%}, "
          f"max={realised_vols.max():.2%} (hedged assuming a flat {sigma:.0%})")

    rows = {}
    pnl_by_source = {}
    for cost_rate, label in [(0.0, "0bps"), (0.001, "10bps")]:
        res = run_delta_hedge(real_paths, times, K, r, sigma, hedge_every=1, cost_rate=cost_rate)
        rows[f"real data, {label}"] = summarise(res["pnl"])
        if cost_rate == 0.0:
            pnl_by_source["Real SPY history"] = res["pnl"]

    # matched GBM simulation: same path count, same daily hedge, same 0bps
    sim_paths, sim_times = simulate_gbm_paths(S0, r, sigma, T, N_STEPS, n_windows, seed=7)
    res_sim = run_delta_hedge(sim_paths, sim_times, K, r, sigma, hedge_every=1, cost_rate=0.0)
    rows["GBM simulation, 0bps"] = summarise(res_sim["pnl"])
    pnl_by_source["GBM simulation"] = res_sim["pnl"]

    df = pd.DataFrame(rows).T
    print("\n" + df.to_string())
    df.to_csv("figures/market_backtest_summary.csv")

    # --- plot: real vs simulated P&L histograms, and the realised-vol spread ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"Real SPY history": "#DD8452", "GBM simulation": "#4C72B0"}
    xlim = (
        min(p.min() for p in pnl_by_source.values()),
        max(p.max() for p in pnl_by_source.values()),
    )
    for label, pnl in pnl_by_source.items():
        ax1.hist(pnl, bins=60, range=xlim, alpha=0.55, density=True,
                  label=f"{label} (std={pnl.std():.2f})", color=colors[label])
    ax1.axvline(0, color="black", lw=1, ls="--")
    ax1.set_xlabel("Terminal hedging P&L (daily hedge, zero costs)")
    ax1.set_ylabel("Density")
    ax1.set_title("Real historical SPY vs simulated GBM")
    ax1.legend()

    ax2.hist(realised_vols, bins=40, color="#55A868", alpha=0.8)
    ax2.axvline(sigma, color="black", lw=1.5, ls="--", label=f"hedged at {sigma:.0%}")
    ax2.set_xlabel("Realised volatility per 1-year window")
    ax2.set_ylabel("Count")
    ax2.set_title("Realised vol varies a lot; we hedge at a single flat number")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("figures/market_backtest.png", dpi=150)
    plt.close(fig)
    print("Saved figures/market_backtest.png")
