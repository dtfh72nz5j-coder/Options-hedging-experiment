"""Research question: for the same average transaction cost, does move-triggered
(band) hedging achieve lower replication error than clock-triggered (fixed
frequency) hedging?

Both policies are run on the *same* simulated paths, so any difference in
the RMSE-vs-cost frontier is attributable to the hedging policy itself, not
to Monte Carlo noise.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation import simulate_gbm_paths
from hedging import run_delta_hedge
from band_hedging import run_band_hedge

S0, K, T, r, sigma = 100, 100, 1.0, 0.04, 0.20
N_STEPS = 252
N_PATHS = 10_000
SEED = 7

HEDGES_PER_YEAR = [6, 12, 21, 42, 63, 84, 126, 252]  # exact divisors of 252 - no rounding collisions
BAND_WIDTHS = [0.20, 0.15, 0.12, 0.08, 0.05, 0.035, 0.02, 0.01, 0.005]
COST_RATES = [0.0005, 0.001]


def hedge_every_from_freq(hedges_per_year, n_steps=N_STEPS):
    return max(1, round(n_steps / hedges_per_year))


def run_time_frontier(paths, times, cost_rate):
    rows = []
    for hpy in HEDGES_PER_YEAR:
        step = hedge_every_from_freq(hpy)
        res = run_delta_hedge(paths, times, K, r, sigma, hedge_every=step, cost_rate=cost_rate)
        rows.append({
            "policy": "time-triggered",
            "param": hpy,
            "mean_trades": res["n_hedges"] - 1,
            "mean_cost": res["total_cost"].mean(),
            "rmse": np.sqrt(np.mean(res["pnl"]**2)),
        })
    return rows


def run_band_frontier(paths, times, cost_rate):
    rows = []
    for bw in BAND_WIDTHS:
        res = run_band_hedge(paths, times, K, r, sigma, band_width=bw, cost_rate=cost_rate)
        rows.append({
            "policy": "band-triggered",
            "param": bw,
            "mean_trades": res["n_trades"].mean(),
            "mean_cost": res["total_cost"].mean(),
            "rmse": np.sqrt(np.mean(res["pnl"]**2)),
        })
    return rows


def plot_frontiers(df):
    fig, axes = plt.subplots(1, len(COST_RATES), figsize=(6.5 * len(COST_RATES), 5), sharey=True)
    if len(COST_RATES) == 1:
        axes = [axes]

    for ax, c in zip(axes, COST_RATES):
        sub = df[df["cost_rate"] == c]
        for policy, marker, color in [("time-triggered", "o", "#4C72B0"), ("band-triggered", "s", "#DD8452")]:
            grp = sub[sub["policy"] == policy].sort_values("mean_cost")
            ax.plot(grp["mean_cost"], grp["rmse"], marker=marker, color=color, label=policy)
        ax.set_xlabel("Mean transaction cost")
        ax.set_title(f"c = {c*1e4:.0f} bps")
        ax.legend()
    axes[0].set_ylabel("RMSE of terminal P&L")
    fig.suptitle("Hedging-error / cost frontier: clock-triggered vs move-triggered rebalancing")
    fig.tight_layout()
    fig.savefig("figures/frontier_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved figures/frontier_comparison.png")


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    paths, times = simulate_gbm_paths(S0, r, sigma, T, N_STEPS, N_PATHS, seed=SEED)

    all_rows = []
    for c in COST_RATES:
        print(f"\n=== cost = {c*1e4:.0f} bps ===")
        time_rows = run_time_frontier(paths, times, c)
        band_rows = run_band_frontier(paths, times, c)
        for row in time_rows + band_rows:
            row["cost_rate"] = c
        all_rows.extend(time_rows + band_rows)

        df_c = pd.DataFrame(time_rows + band_rows)
        print(df_c.sort_values(["policy", "mean_cost"]).to_string(index=False))

    df = pd.DataFrame(all_rows)
    df.to_csv("figures/frontier_comparison.csv", index=False)
    plot_frontiers(df)

    # headline number: at matched mean cost, how much lower is band-hedging's RMSE?
    print("\n=== Matched-cost comparison (10bps) ===")
    sub = df[df.cost_rate == 0.001]
    time_sub = sub[sub.policy == "time-triggered"].sort_values("mean_cost")
    band_sub = sub[sub.policy == "band-triggered"].sort_values("mean_cost")
    target_cost = time_sub["mean_cost"].median()
    t_rmse = np.interp(target_cost, time_sub["mean_cost"], time_sub["rmse"])
    b_rmse = np.interp(target_cost, band_sub["mean_cost"], band_sub["rmse"])
    print(f"At mean cost ~= {target_cost:.3f}: time-triggered RMSE = {t_rmse:.4f}, "
          f"band-triggered RMSE = {b_rmse:.4f} ({(1 - b_rmse/t_rmse)*100:+.1f}%)")
