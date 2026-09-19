"""Fetch and cache real historical daily prices via LSEG Workspace.

Fetching requires the LSEG Workspace desktop app running and logged in
locally (it proxies API requests through localhost). Once fetched, prices
are cached to data/<instrument>_daily.csv so later runs (and the actual
analysis in market_backtest.py) don't need the app open at all.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def _cache_path(instrument):
    return DATA_DIR / f"{instrument.replace('.', '_')}_daily.csv"


def fetch_and_cache(instrument="SPY", start="2000-01-01", end=None, force=False):
    """Download daily close prices from LSEG and cache them locally."""
    cache_path = _cache_path(instrument)
    if cache_path.exists() and not force:
        return load_cached(instrument)

    import datetime as dt

    import lseg.data as ld

    end = end or dt.date.today().isoformat()

    ld.open_session()
    try:
        raw = ld.get_history(
            universe=instrument, fields=["TRDPRC_1"], interval="daily", start=start, end=end
        )
    finally:
        ld.close_session()

    df = raw.copy()
    df.columns = ["close"]
    df.index.name = "date"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"Cached {len(df)} rows to {cache_path}")
    return df


def load_cached(instrument="SPY"):
    cache_path = _cache_path(instrument)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached data at {cache_path}. Run fetch_and_cache() first "
            "(needs LSEG Workspace open and logged in)."
        )
    return pd.read_csv(cache_path, index_col="date", parse_dates=True)


if __name__ == "__main__":
    df = fetch_and_cache("SPY", start="2000-01-01")
    print(df.shape)
    print(df.head())
    print(df.tail())

    import numpy as np

    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    annualised_vol = log_returns.std() * np.sqrt(252)
    print(f"\nFull-sample annualised realised vol: {annualised_vol:.2%}")
