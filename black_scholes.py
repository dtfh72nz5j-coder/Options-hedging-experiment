"""Black-Scholes pricing and delta for a European call option."""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, tau, r, sigma):
    """Compute d1 and d2. tau is time-to-expiry (T - t), in years."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return d1, d2


def black_scholes_call(S, K, tau, r, sigma):
    """European call price under Black-Scholes.

    S     : spot price
    K     : strike price
    tau   : time to expiry in years (T - t)
    r     : risk-free rate (annualised, continuously compounded)
    sigma : volatility (annualised)
    """
    d1, d2 = _d1_d2(S, K, tau, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def call_delta(S, K, tau, r, sigma):
    """Black-Scholes delta of a European call: dC/dS = Phi(d1)."""
    d1, _ = _d1_d2(S, K, tau, r, sigma)
    return norm.cdf(d1)


if __name__ == "__main__":
    # Baseline parameters from the project spec
    S0, K, T, r, sigma = 100, 100, 1.0, 0.04, 0.20

    price = black_scholes_call(S0, K, T, r, sigma)
    delta = call_delta(S0, K, T, r, sigma)
    print(f"ATM 1y call price: {price:.4f}")
    print(f"ATM 1y call delta: {delta:.4f}")

    # --- sanity checks ---
    assert price > 0, "price should be positive"
    assert black_scholes_call(110, K, T, r, sigma) > price, "higher S -> higher price"
    assert black_scholes_call(S0, 110, T, r, sigma) < price, "higher K -> lower price"
    assert 0 <= delta <= 1, "delta should be in [0, 1]"
    assert 0.5 <= delta <= 0.65, "ATM delta should be roughly 0.5-0.6"

    deep_itm_delta = call_delta(200, K, T, r, sigma)
    deep_otm_delta = call_delta(20, K, T, r, sigma)
    assert deep_itm_delta > 0.95, "deep ITM delta should be near 1"
    assert deep_otm_delta < 0.05, "deep OTM delta should be near 0"

    print("All checks passed.")
