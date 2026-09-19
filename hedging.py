"""Discrete delta-hedging simulation for a short European call position."""

import numpy as np

from black_scholes import black_scholes_call, call_delta


def run_delta_hedge(paths, times, K, r, sigma, hedge_every=1, cost_rate=0.0):
    """Simulate a discretely-rebalanced delta hedge for a short call, across many paths.

    We sell one call at t=0, receive the premium, and dynamically buy/sell
    stock to replicate it. Rebalancing happens every `hedge_every` steps of
    the `times` grid (e.g. hedge_every=21 on a 252-step/1y grid ~ monthly).

    Parameters
    ----------
    paths : ndarray (n_paths, n_steps + 1)
        Simulated stock price paths (from simulate_gbm_paths).
    times : ndarray (n_steps + 1,)
        Time grid matching `paths`, from 0 to T.
    K, r, sigma : option strike, risk-free rate, and the volatility used for
        PRICING/HEDGING (may differ from the volatility used to simulate paths).
    hedge_every : int
        Rebalance every this many steps of the grid.
    cost_rate : float
        Proportional transaction cost, charged as cost_rate * S * |shares traded|.

    Returns
    -------
    dict with keys:
        pnl              : ndarray (n_paths,) terminal hedging P&L
        total_cost       : ndarray (n_paths,) total transaction costs paid per path
        n_hedges         : int, number of rebalances performed (including t=0)
        C0               : float, initial option premium received
    """
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    T = times[-1]
    dt_grid = times[1] - times[0]

    hedge_idx = list(range(0, n_steps + 1, hedge_every))
    if hedge_idx[-1] != n_steps:
        hedge_idx.append(n_steps)  # always rebalance/settle at expiry

    S0 = paths[:, 0]
    tau0 = T - times[0]
    C0 = black_scholes_call(S0[0], K, tau0, r, sigma)  # same for every path
    delta_prev = call_delta(S0, K, tau0, r, sigma)

    cash = C0 - delta_prev * S0
    total_cost = np.zeros(n_paths)

    prev_t_idx = 0
    for t_idx in hedge_idx[1:]:
        # grow cash at the risk-free rate over the elapsed time since last hedge
        elapsed = times[t_idx] - times[prev_t_idx]
        cash *= np.exp(r * elapsed)

        S_t = paths[:, t_idx]
        tau = T - times[t_idx]

        if tau > 1e-12:
            delta_t = call_delta(S_t, K, tau, r, sigma)
        else:
            # at expiry, "delta" collapses to the indicator of finishing ITM
            delta_t = (S_t > K).astype(float)

        trade = delta_t - delta_prev
        cash -= trade * S_t

        cost = cost_rate * S_t * np.abs(trade)
        cash -= cost
        total_cost += cost

        delta_prev = delta_t
        prev_t_idx = t_idx

    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0)
    pnl = cash + delta_prev * S_T - payoff

    return {
        "pnl": pnl,
        "total_cost": total_cost,
        "n_hedges": len(hedge_idx),
        "C0": C0,
    }


if __name__ == "__main__":
    from simulation import simulate_gbm_paths

    S0, K, T, r, sigma = 100, 100, 1.0, 0.04, 0.20
    n_steps, n_paths = 252, 10_000

    paths, times = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=1)

    # Daily hedging (hedge every step), zero transaction costs
    result = run_delta_hedge(paths, times, K, r, sigma, hedge_every=1, cost_rate=0.0)
    pnl = result["pnl"]

    print(f"C0 (premium received): {result['C0']:.4f}")
    print(f"Number of hedge dates: {result['n_hedges']}")
    print(f"Mean P&L:   {pnl.mean():.4f}")
    print(f"Std P&L:    {pnl.std():.4f}")
    print(f"RMSE:       {np.sqrt(np.mean(pnl**2)):.4f}")
    print(f"5th pct:    {np.percentile(pnl, 5):.4f}")
    print(f"1st pct:    {np.percentile(pnl, 1):.4f}")

    # With daily hedging and no costs, P&L should be small and roughly centred near 0
    assert abs(pnl.mean()) < 0.5, "daily-hedged mean P&L should be close to 0"
    print("Sanity check passed: daily hedge P&L is small and roughly centred at 0.")
