"""
scripts/download_fixtures.py

One-time utility script to download historical price data and fundamentals
for all NSE symbols used in the MCP Investment Copilot.

Run once from the project root:
    python scripts/download_fixtures.py

Output:
    data/fixtures/{SYMBOL}_2y.csv          — daily OHLCV price history
    data/fixtures/{SYMBOL}_fundamentals.csv — P/E, market cap, sector, etc.

After running, commit the fixtures to the repo.
Tests and all MCP servers read from these files — yFinance is never
called during normal operation (MARKET_DATA_SOURCE=fixture).

External dependencies (one-time install):
    python -m pip install yfinance pandas
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Path setup ────────────────────────────────────────────────────────────────
# Resolve project root regardless of where the script is invoked from
PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
SYMBOLS: list[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "ADANIENT.NS",
    "BAJFINANCE.NS",
    "BHARTIARTL.NS",
    "SBIN.NS",
    "LT.NS",
]

PERIOD = "2y"

# SEBI market cap tier classification (approximate INR crore thresholds)
# Large cap:  rank 1–100 by market cap
# Mid cap:    rank 101–250
# Small cap:  rank 251+
# For fixtures we assign tiers manually based on known classification
MARKET_CAP_TIER: dict[str, str] = {
    "RELIANCE.NS":   "large",
    "TCS.NS":        "large",
    "INFY.NS":       "large",
    "HDFCBANK.NS":   "large",
    "ICICIBANK.NS":  "large",
    "ADANIENT.NS":   "large",
    "BAJFINANCE.NS": "large",
    "BHARTIARTL.NS": "large",
    "SBIN.NS":       "large",
    "LT.NS":         "large",
}

# Throttle between yFinance calls to avoid rate limiting
SLEEP_BETWEEN_CALLS = 2.0  # seconds


# ── Price History ─────────────────────────────────────────────────────────────

def download_price_history(symbol: str, period: str) -> pd.DataFrame:
    """
    Download daily OHLCV data for a symbol and return a clean DataFrame.
    Columns: Date, Open, High, Low, Close, Volume
    Index: reset (Date as column, not index)
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No price data returned for {symbol}")

    # Keep only standard OHLCV columns — drop Dividends, Stock Splits
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)  # strip timezone
    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    return df


def save_price_history(symbol: str, df: pd.DataFrame, period: str) -> Path:
    """Save price history DataFrame to fixtures directory."""
    filename = f"{symbol}_{period}.csv"
    filepath = FIXTURES_DIR / filename
    df.to_csv(filepath, index=False)
    return filepath


def summarise_price_history(symbol: str, df: pd.DataFrame) -> None:
    """Print a human-readable summary of the downloaded price data."""
    missing = df.isnull().sum().sum()
    date_min = df["Date"].min()
    date_max = df["Date"].max()
    rows = len(df)
    close_min = df["Close"].min()
    close_max = df["Close"].max()
    close_last = df["Close"].iloc[-1]

    print(f"  Rows         : {rows} trading days")
    print(f"  Date range   : {date_min} → {date_max}")
    print(f"  Close range  : {close_min:,.2f} → {close_max:,.2f} INR")
    print(f"  Last close   : {close_last:,.2f} INR")
    print(f"  Missing vals : {missing}")
    if missing > 0:
        print(f"  ⚠️  WARNING: {missing} missing values detected — inspect before use")


# ── Fundamentals ──────────────────────────────────────────────────────────────

def download_fundamentals(symbol: str) -> dict:
    """
    Download fundamental data for a symbol via yFinance.
    Returns a flat dict with fields matching the get_fundamentals tool schema.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info

    # Extract with safe fallbacks — yFinance info keys vary by symbol
    market_cap_inr = info.get("marketCap", None)
    market_cap_cr = round(market_cap_inr / 1e7, 2) if market_cap_inr else None

    return {
        "symbol":           symbol,
        "pe_ratio":         info.get("trailingPE", None),
        "market_cap_cr":    market_cap_cr,
        "sector":           info.get("sector", "Unknown"),
        "industry":         info.get("industry", "Unknown"),
        "dividend_yield":   info.get("dividendYield", 0.0),
        "market_cap_tier":  MARKET_CAP_TIER.get(symbol, "unknown"),
        "currency":         info.get("currency", "INR"),
        "exchange":         info.get("exchange", "NSI"),
        "long_name":        info.get("longName", symbol),
    }


def save_fundamentals(records: list[dict]) -> Path:
    """Save all fundamentals records to a single CSV fixture."""
    filepath = FIXTURES_DIR / "fundamentals.csv"
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    return filepath


def summarise_fundamentals(records: list[dict]) -> None:
    """Print a human-readable summary of the downloaded fundamentals."""
    print("\n── Fundamentals Summary ──────────────────────────────────────────")
    print(f"  {'Symbol':<20} {'Sector':<25} {'Tier':<10} {'P/E':>8} {'Mkt Cap (Cr)':>15}")
    print(f"  {'-'*20} {'-'*25} {'-'*10} {'-'*8} {'-'*15}")
    for r in records:
        pe = f"{r['pe_ratio']:.1f}" if r["pe_ratio"] else "N/A"
        mc = f"{r['market_cap_cr']:,.0f}" if r["market_cap_cr"] else "N/A"
        print(f"  {r['symbol']:<20} {r['sector']:<25} {r['market_cap_tier']:<10} {pe:>8} {mc:>15}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    warnings.filterwarnings("ignore")  # suppress yFinance deprecation warnings

    print("=" * 60)
    print("MCP Investment Copilot — Fixture Downloader")
    print(f"Symbols : {len(SYMBOLS)}")
    print(f"Period  : {PERIOD}")
    print(f"Output  : {FIXTURES_DIR}")
    print("=" * 60)

    # ── Price History ──────────────────────────────────────────────
    price_results: dict[str, str] = {}  # symbol → "ok" | "failed"
    fund_records: list[dict] = []

    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"\n[{i}/{len(SYMBOLS)}] {symbol}")
        print("  ── Price History")

        try:
            df = download_price_history(symbol, PERIOD)
            filepath = save_price_history(symbol, df, PERIOD)
            summarise_price_history(symbol, df)
            print(f"  ✓ Saved → {filepath.relative_to(PROJECT_ROOT)}")
            price_results[symbol] = "ok"
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            price_results[symbol] = f"failed: {e}"

        # ── Fundamentals ───────────────────────────────────────────
        print("  ── Fundamentals")
        try:
            record = download_fundamentals(symbol)
            fund_records.append(record)
            print(f"  ✓ Sector: {record['sector']} | "
                  f"P/E: {record['pe_ratio']} | "
                  f"Mkt Cap: {record['market_cap_cr']:,.0f} Cr"
                  if record["market_cap_cr"] else
                  f"  ✓ Sector: {record['sector']} | P/E: {record['pe_ratio']} | Mkt Cap: N/A")
        except Exception as e:
            print(f"  ✗ Fundamentals FAILED: {e}")

        # One sleep per symbol — throttle between symbols, not between calls
        if i < len(SYMBOLS):
            time.sleep(SLEEP_BETWEEN_CALLS)

    # ── Save Fundamentals ──────────────────────────────────────────
    if fund_records:
        fund_path = save_fundamentals(fund_records)
        summarise_fundamentals(fund_records)
        print(f"\n  ✓ Fundamentals saved → {fund_path.relative_to(PROJECT_ROOT)}")

    # ── Final Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Download Complete")
    print("=" * 60)
    ok = [s for s, r in price_results.items() if r == "ok"]
    failed = [s for s, r in price_results.items() if r != "ok"]

    print(f"  Succeeded : {len(ok)}/{len(SYMBOLS)}")
    if failed:
        print(f"  Failed    : {len(failed)}")
        for s in failed:
            print(f"    ✗ {s}: {price_results[s]}")
        sys.exit(1)
    else:
        print("  All symbols downloaded successfully.")
        print(f"\n  Next step: commit data/fixtures/ to the repo.")
        print("  Tests and MCP servers will read from these files.")
        print("  Never run this script again unless fixtures need refresh.")


if __name__ == "__main__":
    main()