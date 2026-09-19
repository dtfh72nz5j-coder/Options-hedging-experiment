"""Move-triggered ("no-transaction band") delta hedging.

Standard delta hedging rebalances on a clock (daily, weekly, ...) regardless
of whether the hedge has actually drifted. Whalley & Wilmott (1997) and the
earlier Leland (1985) line of work ask a different question: under
proportional transaction costs, it's asymptotically better to leave the
hedge alone while it's "close enough", and only trade once the held delta
has drifted more than some band width away from the option's true delta.

This module implements that policy so it can be compared directly against
the fixed-frequency hedge in hedging.py on the same cost/RMSE frontier.
"""

import numpy as np

from black_scholes import black_scholes_call, call_delta


def run_band_hedge(paths, times, K, r, sigma, band_width, cost_rate=0.0):
    """Delta-hedge a short call, trading only when the hedge drifts too far.

    Checked at every step of the `times` grid (so use a daily grid if you
    want daily monitoring) - but a trade is only actually placed when
    |true delta - held delta| > band_width. A band_width of 0 reduces this
    to the fixed-frequency daily hedge in hedging.py.

    Returns
    -------
    dict with keys: pnl, total_cost, n_trades (per path, excluding the
    forced settle at expiry), C0.
    """
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    T = times[-1]

    S0 = paths[:, 0]
    tau0 = T - times[0]
    C0 = black_scholes_call(S0[0], K, tau0, r, sigma)
    held_delta = call_delta(S0, K, tau0, r, sigma)

    cash = C0 - held_delta * S0
    total_cost = np.zeros(n_paths)
    n_trades = np.zeros(n_paths)

    for t_idx in range(1, n_steps + 1):
        dt = times[t_idx] - times[t_idx - 1]
        cash *= np.exp(r * dt)

        S_t = paths[:, t_idx]
        tau = T - times[t_idx]

        if tau > 1e-12:
            true_delta = call_delta(S_t, K, tau, r, sigma)
        else:
            true_delta = (S_t > K).astype(float)

        drift = true_delta - held_delta
        if t_idx == n_steps:
            trigger = np.ones(n_paths, dtype=bool)  # always settle fully at expiry
        else:
            trigger = np.abs(drift) > band_width

        trade = np.where(trigger, drift, 0.0)
        cash -= trade * S_t

        cost = cost_rate * S_t * np.abs(trade)
        cash -= cost
        total_cost += cost
        n_trades += trigger  # bool array adds as 0/1

        held_delta = np.where(trigger, true_delta, held_delta)

    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0)
    pnl = cash + held_delta * S_T - payoff

    return {
        "pnl": pnl,
        "total_cost": total_cost,
        "n_trades": n_trades,
        "C0": C0,
    }


if __name__ == "__main__":
    from simulation import simulate_gbm_paths
    from hedging import run_delta_hedge

    S0, K, T, r, sigma = 100, 100, 1.0, 0.04, 0.20
    n_steps, n_paths = 252, 10_000

    paths, times = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=7)

    # band_width=0 should reproduce the fixed daily hedge almost exactly
    band0 = run_band_hedge(paths, times, K, r, sigma, band_width=0.0, cost_rate=0.0)
    daily = run_delta_hedge(paths, times, K, r, sigma, hedge_every=1, cost_rate=0.0)

    print(f"band_width=0   RMSE: {np.sqrt(np.mean(band0['pnl']**2)):.4f}")
    print(f"daily (clock)  RMSE: {np.sqrt(np.mean(daily['pnl']**2)):.4f}")
    assert np.allclose(band0["pnl"], daily["pnl"], atol=1e-8), \
        "band_width=0 should exactly reproduce daily clock-hedging"
    print("Sanity check passed: band_width=0 reduces to standard daily hedging.")

    # a real band: how much trading does it save, at what RMSE cost?
    band = run_band_hedge(paths, times, K, r, sigma, band_width=0.05, cost_rate=0.001)
    print(f"\nband_width=0.05, 10bps costs:")
    print(f"  Mean trades/year: {band['n_trades'].mean():.1f} (vs 252 for daily clock)")
    print(f"  RMSE: {np.sqrt(np.mean(band['pnl']**2)):.4f}")
    print(f"  Mean cost: {band['total_cost'].mean():.4f}")
