"""
servers/market_data/tools/price_history.py

Implementation of the get_price_history tool.

Responsibilities:
    - Load historical OHLCV data from CSV fixtures (MARKET_DATA_SOURCE=fixture)
    - Clean prices: replace zeros with NaN, forward fill, back fill
    - Compute log-returns from closing prices (single source of truth)
    - Switch to live yFinance download when MARKET_DATA_SOURCE=live (v2)

This file contains pure computation logic only.
No MCP protocol code — that lives in server.py.

Path resolution:
    __file__ is .../servers/market_data/tools/price_history.py
    Project root is four parents up → .../mcp-investment-copilot/
    Fixtures directory → .../mcp-investment-copilot/data/fixtures/

Data quality:
    Missing prices are forward-filled (standard practice for liquid large-caps).
    A missing price on day t produces zero return for day t — far less harmful
    than an artificially large return or NaN propagating through the system.
    Zero prices are treated as invalid and replaced with NaN before filling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_PERIODS = {"1y", "2y", "3y", "5y"}
VALID_RETURN_TYPES = {"log", "simple"}


# ── Data quality ──────────────────────────────────────────────────────────────

def _clean_prices(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Clean closing prices before return computation.

    Steps:
        1. Replace zero prices with NaN — zero is not a valid closing price
        2. Forward fill NaN — standard practice for liquid large-cap stocks
           A missing price on day t gets the price from day t-1
           This produces a zero return for the gap day — acceptable distortion
        3. Back fill any remaining NaN at the start of the series
           Handles edge case where first rows are missing
        4. Validate — raise if unresolvable NaN remains after filling

    Args:
        df:     DataFrame with at least a "Close" column
        symbol: Symbol name — used in error messages only

    Returns:
        Cleaned DataFrame with no NaN or zero closing prices

    Raises:
        ValueError: if NaN remains after forward and back fill
    """
    df = df.copy()

    # Step 1 — zero prices are invalid, replace with NaN
    zero_count = (df["Close"] == 0).sum()
    if zero_count > 0:
        df.loc[df["Close"] == 0, "Close"] = np.nan

    # Step 2 — forward fill: missing day gets previous day's price
    df["Close"] = df["Close"].ffill()

    # Step 3 — back fill: handles NaN at the very start of the series
    df["Close"] = df["Close"].bfill()

    # Step 4 — validate nothing remains unresolvable
    if df["Close"].isna().any():
        raise ValueError(
            f"Symbol {symbol} has unresolvable NaN closing prices "
            f"after forward and back fill. Inspect the fixture file."
        )

    return df


# ── Fixture loader ────────────────────────────────────────────────────────────

def _load_fixture(symbol: str, period: str) -> pd.DataFrame:
    """
    Load a single symbol's OHLCV data from CSV fixture.

    Expected filename: {symbol}_{period}.csv
    Expected columns:  Date, Open, High, Low, Close, Volume

    Raises:
        FileNotFoundError: if fixture CSV does not exist
        ValueError: if CSV is missing required columns or is empty
    """
    filename = f"{symbol}_{period}.csv"
    filepath = FIXTURES_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Fixture not found: {filepath}\n"
            f"Run scripts/download_fixtures.py to generate fixtures."
        )

    df = pd.read_csv(filepath, parse_dates=["Date"])

    if df.empty:
        raise ValueError(f"Fixture file is empty: {filepath}")

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Fixture {filename} missing required columns: {missing}"
        )

    df = df.sort_values("Date").reset_index(drop=True)

    # Clean prices before returning
    df = _clean_prices(df, symbol)

    return df


# ── Return computation ────────────────────────────────────────────────────────

def _compute_log_returns(prices: list[float]) -> list[float]:
    """
    Compute daily log-returns from a closing price series.

    log_return_t = ln(P_t / P_{t-1})

    Returns a list of length len(prices) - 1.
    The first element has no return (no previous price) — it is dropped.

    This is the single source of truth for log-returns across the system.
    All downstream servers (Risk Engine, Optimiser, Simulator) consume
    log-returns computed here — never raw prices directly.

    Raises:
        ValueError: if prices contain zero or negative values
        ValueError: if computation produces NaN or infinite returns
    """
    arr = np.array(prices, dtype=float)

    # Guard against zero or negative prices
    if np.any(arr <= 0):
        raise ValueError(
            "Price series contains zero or negative values. "
            "Cannot compute log-returns. "
            "Check _clean_prices — this should have been caught earlier."
        )

    log_returns = np.diff(np.log(arr))

    # Guard against NaN or infinite returns
    if not np.all(np.isfinite(log_returns)):
        raise ValueError(
            "Log-return computation produced NaN or infinite values. "
            "Inspect the price series for anomalies."
        )

    return log_returns.tolist()


def _compute_simple_returns(prices: list[float]) -> list[float]:
    """
    Compute daily simple returns from a closing price series.

    simple_return_t = (P_t - P_{t-1}) / P_{t-1}

    Returns a list of length len(prices) - 1.
    """
    arr = np.array(prices, dtype=float)

    if np.any(arr <= 0):
        raise ValueError(
            "Price series contains zero or negative values. "
            "Cannot compute simple returns."
        )

    simple_returns = np.diff(arr) / arr[:-1]

    if not np.all(np.isfinite(simple_returns)):
        raise ValueError(
            "Simple return computation produced NaN or infinite values."
        )

    return simple_returns.tolist()


# ── Live data loader (v2) ─────────────────────────────────────────────────────

def _load_live(symbol: str, period: str) -> pd.DataFrame:
    """
    Download live data from yFinance.
    Only active when MARKET_DATA_SOURCE=live.
    Not called in v1 — fixture path is the default.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for live data. "
            "It is installed but the import failed unexpectedly."
        )

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No live data returned for {symbol}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Clean prices before returning
    df = _clean_prices(df, symbol)

    return df


# ── Main tool function ────────────────────────────────────────────────────────

def get_price_history(
    symbols: list[str],
    period: str = "2y",
    return_type: Literal["log", "simple"] = "log",
) -> dict:
    """
    Retrieve historical daily closing prices and pre-computed returns
    for a list of NSE symbols.

    This is the implementation of the get_price_history MCP tool.
    Called by server.py — never called directly by other servers.

    Args:
        symbols:     List of NSE ticker symbols e.g. ["RELIANCE.NS", "TCS.NS"]
        period:      Historical period e.g. "1y", "2y"
        return_type: "log" (default) or "simple"

    Returns:
        dict matching the get_price_history tool output schema:
            prices:      dict[symbol, list[float]]  — daily closing prices
            log_returns: dict[symbol, list[float]]  — pre-computed returns
            dates:       list[str]                  — aligned date strings
            source:      str                        — "fixture" | "live"
            period:      str                        — echoed for audit

    Raises:
        ValueError: if period or return_type is invalid
        FileNotFoundError: if a fixture file is missing
        ValueError: if symbols list is empty
    """
    # ── Input validation ──────────────────────────────────────────
    if not symbols:
        raise ValueError("symbols list must not be empty")

    if period not in VALID_PERIODS:
        raise ValueError(
            f"Invalid period '{period}'. Must be one of: {VALID_PERIODS}"
        )

    if return_type not in VALID_RETURN_TYPES:
        raise ValueError(
            f"Invalid return_type '{return_type}'. "
            f"Must be one of: {VALID_RETURN_TYPES}"
        )

    # ── Determine data source ─────────────────────────────────────
    source = os.environ.get("MARKET_DATA_SOURCE", "fixture").lower()
    if source not in {"fixture", "live"}:
        raise ValueError(
            f"Invalid MARKET_DATA_SOURCE='{source}'. "
            f"Must be 'fixture' or 'live'."
        )

    # ── Load data for each symbol ─────────────────────────────────
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if source == "fixture":
            frames[symbol] = _load_fixture(symbol, period)
        else:
            frames[symbol] = _load_live(symbol, period)

    # ── Align dates across all symbols ───────────────────────────
    # Use intersection of dates — ensures all symbols have the same
    # date index, handling holidays and missing trading days cleanly
    date_sets = [set(df["Date"].astype(str)) for df in frames.values()]
    common_dates = sorted(set.intersection(*date_sets))

    if not common_dates:
        raise ValueError(
            f"No common trading dates found across symbols: {symbols}"
        )

    # ── Build output ──────────────────────────────────────────────
    prices: dict[str, list[float]] = {}
    returns: dict[str, list[float]] = {}

    for symbol, df in frames.items():
        # Filter to common dates only
        mask = df["Date"].astype(str).isin(set(common_dates))
        aligned = df[mask].sort_values("Date")
        close = aligned["Close"].tolist()
        prices[symbol] = close

        # Compute returns — length is len(close) - 1
        if return_type == "log":
            returns[symbol] = _compute_log_returns(close)
        else:
            returns[symbol] = _compute_simple_returns(close)

    # Dates aligned to returns length — drop first date (no return for t=0)
    return_dates = common_dates[1:]

    return {
        "prices":      prices,
        "log_returns": returns,
        "dates":       return_dates,
        "source":      source,
        "period":      period,
    }