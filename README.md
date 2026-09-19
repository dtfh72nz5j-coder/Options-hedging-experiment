# Discrete Delta Hedging Under Transaction Costs

Monte Carlo study of how hedge rebalancing frequency trades off replication
error against transaction costs, for a delta-hedged European call.

## Research Question

How does discrete hedging frequency affect the replication error and
transaction costs of a delta-hedged European call option?

## Model

- **Geometric Brownian motion** for the underlying: `dS = μS dt + σS dW`,
  simulated with the exact lognormal discretisation.
- **Black-Scholes** pricing and delta for a European call, `Δ = Φ(d1)`.
- **Discrete delta hedging**: sell one call at `t=0`, hold `Δ(t)` shares,
  rebalance at fixed intervals, finance everything through a cash account
  earning the risk-free rate `r`. Terminal hedging P&L is
  `B(T) + Δ(T)S(T) - max(S(T)-K, 0)` — zero if hedging were continuous and
  frictionless, a random variable at any finite hedge frequency.
- **Proportional transaction costs**: `cost = c · S(t) · |Δ(t) - Δ(t-1)|`,
  charged against the cash account at every rebalance.

## Experiment

| Parameter | Value |
|---|---|
| `S0`, `K` | 100, 100 (at-the-money) |
| `T` | 1 year |
| `r` | 4% |
| `σ` (pricing & simulation) | 20% |
| Monte Carlo paths | 10,000 |
| Hedge frequencies | 12 (monthly), 52 (weekly), 252 (daily), plus an extended sweep from 2 to 1,260/year |
| Transaction costs | 0, 5bps, 10bps of notional traded |

Extensions: a finer frequency sweep to locate a practical optimum, and a
volatility-misspecification test (hedge at 20% implied vol while the market
realises 15%/20%/25%).

## Results

**Zero transaction costs — frequency alone:**

| Frequency | Mean P&L | Std P&L | RMSE | 5th pct | 1st pct |
|---|---|---|---|---|---|
| Monthly (12/yr) | -0.01 | 1.92 | 1.92 | -3.23 | -5.31 |
| Weekly (51/yr) | 0.01 | 0.95 | 0.95 | -1.55 | -2.65 |
| Daily (252/yr) | 0.00 | 0.44 | 0.44 | -0.74 | -1.15 |

Hedging error shrinks roughly as `1/√n` as hedge frequency `n` increases —
confirmed against the theoretical scaling law across nearly three orders of
magnitude of hedge frequency (see `figures/extended_frequency_sweep.png`).

**With transaction costs — the trade-off appears:**

| Frequency | Cost | Mean P&L | Std P&L | RMSE | Mean cost |
|---|---|---|---|---|---|
| Monthly | 10bps | -0.12 | 1.93 | 1.93 | 0.11 |
| Weekly | 10bps | -0.22 | 0.97 | 0.99 | 0.22 |
| Daily | 10bps | -0.50 | 0.49 | 0.70 | 0.49 |

Daily hedging still has the lowest *variance*, but its RMSE is dragged up by
a mean P&L that's turned meaningfully negative — fees eat into the hedge.

**Practical optimal hedge frequency.** Sweeping frequency from 2 to
1,260 hedges/year reveals a genuine minimum in RMSE once costs are switched
on, because mean transaction cost grows like `√n` while the discretisation
variance shrinks like `1/n` — two competing power laws whose sum has a
minimum at `n* ∝ 1/c`:

| Cost | Optimal frequency | Minimum RMSE |
|---|---|---|
| 5bps | ~420/year | 0.49 |
| 10bps | ~180/year | 0.70 |

Doubling the cost rate roughly halves the optimal hedge frequency, matching
the `n* ∝ 1/c` prediction.

**Volatility misspecification.** Hedging at a fixed 20% implied vol while
the market realises a different volatility produces a systematic (not just
noisier) P&L shift — the seller profits when realised vol undershoots
implied vol, and loses when it overshoots:

| Realised vol | Mean P&L | RMSE |
|---|---|---|
| 15% | +1.97 | 2.10 |
| 20% | +0.01 | 0.44 |
| 25% | -1.99 | 2.23 |

**Figures** (`figures/`):

| File | Shows |
|---|---|
| `gbm_sample_paths.png` | 20 simulated stock paths under GBM |
| `pnl_histograms_zero_cost.png` | P&L distribution narrowing with hedge frequency |
| `hedging_error_vs_frequency.png` | RMSE vs frequency, at 0/5/10bps |
| `transaction_cost_vs_frequency.png` | Mean transaction cost vs frequency |
| `extended_frequency_sweep.png` | Fine-grained sweep with `1/√n` theory overlay; the cost curves' U-shape |
| `vol_misspecification_pnl.png` | P&L distributions under vol misspecification |
| `frontier_comparison.png` | Clock-triggered vs move-triggered rehedging, RMSE vs cost |
| `market_backtest.png` | Real SPY-history hedging P&L vs matched GBM simulation |

## Extensions

Two further research questions go beyond the core frequency/cost trade-off
above.

**1. Does *how* you decide when to rehedge matter, independent of *how
often*?** Standard practice (and everything above) rebalances on a clock —
daily, weekly, monthly. `band_hedging.py` implements the alternative from
Whalley & Wilmott (1997): only trade when the option's true delta has
drifted more than a band width away from the delta you're currently
holding, letting small drifts net out instead of paying to chase them.
`frontier_comparison.py` runs both policies on the *same* simulated paths
and traces out an RMSE-vs-mean-cost frontier for each:

| Policy | RMSE at matched mean cost (~$0.23, 10bps) |
|---|---|
| Clock-triggered (fixed frequency) | 0.9976 |
| Move-triggered (delta band) | 0.8697 (**-12.8%**) |

Move-triggered rehedging Pareto-dominates clock-triggered rehedging across
almost the entire frontier — for the same amount paid in transaction costs,
watching delta rather than the clock gets a materially lower replication
error.

**2. Does the GBM-simulated hedging-error distribution actually describe
real markets?** `market_backtest.py` runs the *exact same* hedging engine
used throughout this project on 1,292 overlapping 1-year windows of real
SPY daily closes (2000–2026, via LSEG Workspace, cached in `data/`) instead
of simulated GBM paths, hedging every window at a flat 20% vol assumption,
and compares the result to a matched GBM simulation:

| Source | Mean P&L | Std P&L | RMSE | Skew | Excess kurtosis |
|---|---|---|---|---|---|
| Real SPY history | +0.79 | 2.45 | 2.58 | -1.42 | 2.85 |
| GBM simulation | +0.00 | 0.44 | 0.44 | -0.16 | 1.93 |

Realised hedging-P&L standard deviation is **~5.6x larger** in real data
than GBM predicts — and the shape is qualitatively different, not just
wider: strong negative skew and a fat left tail (losses beyond -$12,
something GBM structurally cannot produce) driven by crisis-period
volatility spikes (realised vol across windows ranges 7%–46%, against the
flat 20% assumption used to hedge). This is model risk beyond the simple
vol-misspecification test above: it isn't just that the *average* volatility
can be wrong, it's that volatility clusters and jumps in ways a constant-vol
lognormal process cannot represent. (Caveat: overlapping windows share most
of their days, so the effective independent sample size is closer to the
~26 calendar years in the data than the 1,292 nominal windows — the mean/std
gap is still real, but treat exact percentiles with that in mind.)

## Conclusion

Discrete delta hedging leaves a genuine replication error whose standard
deviation falls as `1/√n` in the hedge frequency `n` — a direct consequence
of the option's gamma and the way volatility (not drift) drives the
quadratic variation of the hedging error between rebalances. But every
rebalance costs money: expected transaction cost grows as `√n`. The two
effects combine into a real trade-off with a locatable optimum — in this
setup, roughly 420 hedges/year at 5bps and 180/year at 10bps — beyond which
hedging *more* actually makes total P&L variability *worse*. Getting the
hedging volatility wrong is far more damaging than any hedge frequency
choice: a 5-point vol misspecification shifted mean P&L by roughly ±$2,
dwarfing the frequency effects at any practical hedge cadence. Choosing
*when* to rehedge is not just about frequency either — a delta-band policy
beats a clock at the same cost by ~13% RMSE. And the starkest finding is
that the entire GBM framework understates real-world hedging risk by
roughly 5-6x, because real markets cluster and jump in ways a constant-vol
lognormal process cannot represent — the single biggest source of error in
this whole exercise is the model itself, not the discretisation of it.

## Limitations

- Constant volatility and GBM dynamics — no jumps, no stochastic volatility
  (the market backtest quantifies exactly how much this costs you).
- Transaction costs are a simplified proportional model, not a realistic
  order book (no bid-ask spread, market impact, or size-dependent slippage).
- Black-Scholes assumptions throughout (frictionless continuous trading in
  the limit, no shorting constraints, borrowing/lending at a single rate).
- Only tested one strike/maturity (at-the-money, 1 year); results may differ
  for deep ITM/OTM options or very short-dated ones where gamma is sharper.

## How to Run

```bash
cd options-hedging
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib

python black_scholes.py       # pricing/delta sanity checks
python simulation.py          # GBM path simulation + sample plot
python hedging.py             # single delta-hedge run, sanity check
python experiments.py         # Steps 9-14: frequency + cost sweep, main figures
python extensions.py          # Steps 15-16: extended sweep + vol misspecification
python band_hedging.py        # rehedging-band engine, sanity check
python frontier_comparison.py # clock- vs move-triggered RMSE/cost frontier
python market_backtest.py     # real SPY history vs GBM (needs data/SPY_daily.csv)
```

`market_backtest.py` reads from a local cache (`data/SPY_daily.csv`) so it
runs standalone. To refresh or extend that cache, open LSEG Workspace
(logged in) and run `python market_data.py`, which pulls fresh daily prices
via `lseg.data` and re-caches them — `pip install lseg-data` first.

All figures and result tables are written to `figures/`.
