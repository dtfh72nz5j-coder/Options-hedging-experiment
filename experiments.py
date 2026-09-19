"""Run the delta-hedging frequency experiments and produce the project's figures."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation import simulate_gbm_paths
from hedging import run_delta_hedge

# ---------------------------------------------------------------------------
# Baseline parameters
# ---------------------------------------------------------------------------
S0, K, T, r, sigma = 100, 100, 1.0, 0.04, 0.20
N_STEPS = 252          # daily grid; other frequencies subsample this grid
N_PATHS = 10_000
SEED = 7

FREQUENCIES = {"Monthly": 12, "Weekly": 52, "Daily": 252}


def hedge_every_from_freq(hedges_per_year, n_steps=N_STEPS):
    """Convert a target 'hedges per year' into a step count on the daily grid."""
    return max(1, round(n_steps / hedges_per_year))


def summarise(pnl, total_cost):
    return {
        "Mean P&L": pnl.mean(),
        "Std P&L": pnl.std(),
        "RMSE": np.sqrt(np.mean(pnl**2)),
        "5th pct P&L": np.percentile(pnl, 5),
        "1st pct P&L": np.percentile(pnl, 1),
        "Mean cost": total_cost.mean(),
    }


def run_frequency_sweep(paths, times, cost_rate):
    rows = {}
    pnl_by_freq = {}
    for name, hpy in FREQUENCIES.items():
        step = hedge_every_from_freq(hpy)
        res = run_delta_hedge(paths, times, K, r, sigma, hedge_every=step, cost_rate=cost_rate)
        rows[name] = summarise(res["pnl"], res["total_cost"])
        rows[name]["Hedges/yr (actual)"] = res["n_hedges"] - 1  # exclude t=0
        pnl_by_freq[name] = res["pnl"]
    return pd.DataFrame(rows).T, pnl_by_freq


def plot_pnl_histograms(pnl_by_freq, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    all_pnl = np.concatenate(list(pnl_by_freq.values()))
    xlim = (np.percentile(all_pnl, 0.5), np.percentile(all_pnl, 99.5))
    for ax, (name, pnl) in zip(axes, pnl_by_freq.items()):
        ax.hist(pnl, bins=60, range=xlim, color="#4C72B0", alpha=0.85)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_title(f"{name} (std={pnl.std():.2f})")
        ax.set_xlabel("Terminal P&L")
    axes[0].set_ylabel("Frequency")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"figures/{filename}", dpi=150)
    plt.close(fig)
    print(f"Saved figures/{filename}")


def plot_tradeoff(results_by_cost):
    """Graph 1: RMSE vs hedge frequency. Graph 2: mean cost vs hedge frequency."""
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.5))
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

    for c, df in results_by_cost.items():
        x = df["Hedges/yr (actual)"]
        ax1.plot(x, df["RMSE"], marker="o", label=f"c = {c*1e4:.0f} bps")
        ax2.plot(x, df["Mean cost"], marker="o", label=f"c = {c*1e4:.0f} bps")

    ax1.set_xscale("log")
    ax1.set_xlabel("Hedges per year (log scale)")
    ax1.set_ylabel("RMSE of terminal P&L")
    ax1.set_title("Hedging error vs hedge frequency")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("figures/hedging_error_vs_frequency.png", dpi=150)
    plt.close(fig1)
    print("Saved figures/hedging_error_vs_frequency.png")

    ax2.set_xscale("log")
    ax2.set_xlabel("Hedges per year (log scale)")
    ax2.set_ylabel("Mean transaction cost")
    ax2.set_title("Transaction cost vs hedge frequency")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("figures/transaction_cost_vs_frequency.png", dpi=150)
    plt.close(fig2)
    print("Saved figures/transaction_cost_vs_frequency.png")


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    print(f"Simulating {N_PATHS:,} GBM paths, {N_STEPS} steps, seed={SEED} ...")
    paths, times = simulate_gbm_paths(S0, r, sigma, T, N_STEPS, N_PATHS, seed=SEED)

    # --- Steps 9-11: zero-cost frequency comparison ---
    print("\n=== Zero transaction costs ===")
    df_zero, pnl_zero = run_frequency_sweep(paths, times, cost_rate=0.0)
    print(df_zero)
    plot_pnl_histograms(pnl_zero, "Terminal P&L by hedge frequency (no transaction costs)",
                         "pnl_histograms_zero_cost.png")

    # --- Steps 12-14: transaction costs ---
    cost_rates = [0.0, 0.0005, 0.001]
    results_by_cost = {}
    for c in cost_rates:
        print(f"\n=== Transaction cost c = {c*1e4:.0f} bps ===")
        df_c, pnl_c = run_frequency_sweep(paths, times, cost_rate=c)
        print(df_c)
        results_by_cost[c] = df_c

    plot_tradeoff(results_by_cost)

    # save the with-cost summary table for the README/report
    combined = pd.concat(
        {f"{c*1e4:.0f}bps": df for c, df in results_by_cost.items()}, names=["cost", "frequency"]
    )
    combined.to_csv("figures/summary_results.csv")
    print("\nSaved figures/summary_results.csv")
