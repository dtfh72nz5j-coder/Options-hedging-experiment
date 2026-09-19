"""Optional extensions: a finer hedge-frequency sweep, and volatility misspecification.

Step 15 - does RMSE keep falling smoothly as we add more hedge dates, and where
does the marginal benefit start to flatten out relative to the rising cost?

Step 16 - what happens to the SELLER's hedged P&L when the volatility used to
price/hedge the option (implied vol) differs from the volatility that actually
realises in the market? This is the classic "short gamma, wrong vol" problem.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation import simulate_gbm_paths
from hedging import run_delta_hedge

S0, K, T, r, sigma = 100, 100, 1.0, 0.04, 0.20
N_STEPS = 252
N_STEPS_FINE = 1260  # 5x/trading-day resolution, so we can test hedge frequencies above 252/yr
N_PATHS = 10_000
SEED = 7

# from twice a year up to 5x/day - almost three orders of magnitude
EXTENDED_HEDGES_PER_YEAR = [2, 4, 6, 12, 28, 52, 84, 126, 180, 252, 315, 420, 630, 1260]


def hedge_every_from_freq(hedges_per_year, n_steps=N_STEPS_FINE):
    return max(1, round(n_steps / hedges_per_year))


# ---------------------------------------------------------------------------
# Step 15: finer frequency sweep
# ---------------------------------------------------------------------------
def run_extended_sweep(paths, times, cost_rates):
    n_steps = paths.shape[1] - 1
    rows = []
    for c in cost_rates:
        for hpy in EXTENDED_HEDGES_PER_YEAR:
            step = hedge_every_from_freq(hpy, n_steps)
            res = run_delta_hedge(paths, times, K, r, sigma, hedge_every=step, cost_rate=c)
            pnl, cost = res["pnl"], res["total_cost"]
            rows.append({
                "cost_bps": c * 1e4,
                "hedges_per_year": res["n_hedges"] - 1,
                "rmse": np.sqrt(np.mean(pnl**2)),
                "std_pnl": pnl.std(),
                "mean_cost": cost.mean(),
            })
    return pd.DataFrame(rows)


def plot_extended_sweep(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # theoretical 1/sqrt(n) reference curve, calibrated off the zero-cost daily point
    zero_cost = df[df["cost_bps"] == 0.0].sort_values("hedges_per_year")
    ref_n, ref_rmse = zero_cost["hedges_per_year"].iloc[-1], zero_cost["rmse"].iloc[-1]
    n_grid = np.array(EXTENDED_HEDGES_PER_YEAR, dtype=float)
    theory = ref_rmse * np.sqrt(ref_n / n_grid)

    for c, grp in df.groupby("cost_bps"):
        grp = grp.sort_values("hedges_per_year")
        ax1.plot(grp["hedges_per_year"], grp["rmse"], marker="o", label=f"c = {c:.0f} bps")
        ax2.plot(grp["hedges_per_year"], grp["mean_cost"], marker="o", label=f"c = {c:.0f} bps")

    ax1.plot(n_grid, theory, ls="--", color="black", lw=1, label=r"theory: $\propto 1/\sqrt{n}$")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Hedges per year (log scale)")
    ax1.set_ylabel("RMSE of terminal P&L (log scale)")
    ax1.set_title("Hedging error vs frequency, with theoretical scaling")
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_xlabel("Hedges per year (log scale)")
    ax2.set_ylabel("Mean transaction cost")
    ax2.set_title("Transaction cost: finer frequency grid")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("figures/extended_frequency_sweep.png", dpi=150)
    plt.close(fig)
    print("Saved figures/extended_frequency_sweep.png")


# ---------------------------------------------------------------------------
# Step 16: volatility misspecification
# ---------------------------------------------------------------------------
def run_vol_misspecification(implied_vol=0.20, realised_vols=(0.15, 0.20, 0.25)):
    """Hedge assuming `implied_vol` throughout, but simulate the stock at each
    of `realised_vols`. Daily hedging, zero transaction costs, to isolate the
    pure model-risk effect from hedging-frequency/cost effects."""
    rows = {}
    pnl_by_vol = {}
    for i, rv in enumerate(realised_vols):
        paths, times = simulate_gbm_paths(S0, r, rv, T, N_STEPS, N_PATHS, seed=SEED + i)
        res = run_delta_hedge(paths, times, K, r, implied_vol, hedge_every=1, cost_rate=0.0)
        pnl = res["pnl"]
        label = f"realised {rv:.0%}"
        rows[label] = {
            "Mean P&L": pnl.mean(),
            "Std P&L": pnl.std(),
            "RMSE": np.sqrt(np.mean(pnl**2)),
            "5th pct": np.percentile(pnl, 5),
            "1st pct": np.percentile(pnl, 1),
        }
        pnl_by_vol[label] = pnl
    return pd.DataFrame(rows).T, pnl_by_vol


def plot_vol_misspecification(pnl_by_vol, implied_vol):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#DD8452", "#4C72B0", "#55A868"]
    for (label, pnl), color in zip(pnl_by_vol.items(), colors):
        ax.hist(pnl, bins=60, alpha=0.55, label=f"{label} (mean={pnl.mean():.2f})",
                 color=color, density=True)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Terminal hedging P&L")
    ax.set_ylabel("Density")
    ax.set_title(f"Hedged at implied vol = {implied_vol:.0%}, daily rebalancing, zero costs")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/vol_misspecification_pnl.png", dpi=150)
    plt.close(fig)
    print("Saved figures/vol_misspecification_pnl.png")


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    print(f"Simulating {N_PATHS:,} GBM paths at {N_STEPS_FINE}-step resolution for the extended sweep ...")
    fine_paths, fine_times = simulate_gbm_paths(S0, r, sigma, T, N_STEPS_FINE, N_PATHS, seed=SEED)

    print("\n=== Step 15: extended hedge-frequency sweep ===")
    df_ext = run_extended_sweep(fine_paths, fine_times, cost_rates=[0.0, 0.0005, 0.001])
    print(df_ext.to_string(index=False))
    df_ext.to_csv("figures/extended_frequency_sweep.csv", index=False)
    plot_extended_sweep(df_ext)

    print("\n=== Step 16: volatility misspecification (implied 20%, daily hedge) ===")
    df_vol, pnl_by_vol = run_vol_misspecification()
    print(df_vol)
    df_vol.to_csv("figures/vol_misspecification_summary.csv")
    plot_vol_misspecification(pnl_by_vol, implied_vol=0.20)
