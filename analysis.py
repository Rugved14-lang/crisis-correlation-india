"""Core analysis functions: kept separate from the UI so they are easy to test."""
import numpy as np
import pandas as pd

# Sector map used to order the heatmap so clusters are visible
SECTORS = {
    "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT", "HCLTECH.NS": "IT",
    "HDFCBANK.NS": "Banks", "ICICIBANK.NS": "Banks", "SBIN.NS": "Banks", "KOTAKBANK.NS": "Banks",
    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma",
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "NTPC.NS": "Energy",
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG",
    "MARUTI.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
    "TATASTEEL.NS": "Metals",
}

# Market regimes to compare (editable in the app)
DEFAULT_PERIODS = {
    "Calm (2017)": ("2017-01-01", "2017-12-31"),
    "COVID crash": ("2020-02-20", "2020-03-31"),
    "2022 selloff": ("2022-01-01", "2022-06-30"),
}


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def avg_pairwise_corr(returns: pd.DataFrame) -> float:
    """Mean of all off-diagonal correlations (each pair counted once)."""
    c = returns.corr().values
    n = c.shape[0]
    if n < 2:
        return float("nan")
    return float((c.sum() - n) / (n * (n - 1)))


def rolling_avg_corr(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Average pairwise correlation over a rolling window: the 'headline' chart."""
    out = {}
    idx = returns.index
    for i in range(window, len(idx) + 1):
        out[idx[i - 1]] = avg_pairwise_corr(returns.iloc[i - window:i])
    return pd.Series(out, name=f"{window}d avg correlation")


def slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[start:end]


def order_by_sector(columns) -> list:
    return sorted(columns, key=lambda t: (SECTORS.get(t, "Z"), t))


def annualised_vol(returns: pd.DataFrame) -> pd.Series:
    return returns.std() * np.sqrt(252)


def rolling_vol(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    return returns.rolling(window).std() * np.sqrt(252)