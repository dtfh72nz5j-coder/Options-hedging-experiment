"""Monte Carlo simulation of stock price paths under geometric Brownian motion."""

import numpy as np


def simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=None):
    """Simulate GBM paths using the exact lognormal discretisation.

    S(t+dt) = S(t) * exp[(mu - sigma^2/2) dt + sigma sqrt(dt) Z],  Z ~ N(0,1)

    Returns
    -------
    paths : ndarray, shape (n_paths, n_steps + 1)
        paths[:, 0] == S0 for every path.
    times : ndarray, shape (n_steps + 1,)
        Time grid from 0 to T.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    paths = S0 * np.exp(log_paths)
    times = np.linspace(0, T, n_steps + 1)
    return paths, times


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    ### --- parameters ---
    S0, mu, sigma, T = 100, 0.04, 0.20, 1.0
    n_steps, n_paths = 252, 10_000
    ###

    paths, times = simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=42)

    # --- sanity checks ---
    assert paths.shape == (n_paths, n_steps + 1)
    assert np.allclose(paths[:, 0], S0), "all paths should start at S0"
    assert np.all(paths > 0), "GBM paths should stay strictly positive"

    std_early = paths[:, n_steps // 4].std()
    std_late = paths[:, -1].std()
    assert std_late > std_early, "dispersion should grow through time"
    print(f"Std dev at t=T/4: {std_early:.2f}, at t=T: {std_late:.2f}")
    print(f"Mean terminal price: {paths[:, -1].mean():.2f} (E[S_T] under mu = {S0 * np.exp(mu * T):.2f})")
    print("All checks passed.")

    # --- plot 20 sample paths ---
    plt.figure(figsize=(8, 5))
    for i in range(20):
        plt.plot(times, paths[i], lw=0.8, alpha=0.8)
    plt.axhline(S0, color="black", lw=0.8, ls="--", label="S0")
    plt.xlabel("Time (years)")
    plt.ylabel("Stock price")
    plt.title("20 simulated GBM paths")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/gbm_sample_paths.png", dpi=150)
    print("Saved figures/gbm_sample_paths.png")
