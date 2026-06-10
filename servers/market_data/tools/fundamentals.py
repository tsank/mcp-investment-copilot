"""
servers/market_data/tools/fundamentals.py

Implementation of the get_fundamentals tool.

Responsibilities:
    - Load fundamental data from CSV fixture (fundamentals.csv)
    - Filter to requested symbols
    - Return structured fundamental data per symbol

This file contains pure computation logic only.
No MCP protocol code — that lives in server.py.

Path resolution:
    __file__ is .../servers/market_data/tools/fundamentals.py
    Project root is four parents up → .../mcp-investment-copilot/
    Fixtures directory → .../mcp-investment-copilot/data/fixtures/

Fixture file:
    data/fixtures/fundamentals.csv
    All symbols in one file — filtered by symbol at query time.
    Columns: symbol, pe_ratio, market_cap_cr, sector, industry,
             dividend_yield, market_cap_tier, currency, exchange, long_name

Why industry is included:
    industry provides finer-grained classification than sector alone.
    Example: sector="Financial Services", industry="Private Sector Bank"
    The Compliance server can use industry for finer concentration checks
    in future versions without requiring a schema change.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"
FUNDAMENTALS_FILE = FIXTURES_DIR / "fundamentals.csv"

# ── Required columns in fundamentals.csv ──────────────────────────────────────
REQUIRED_COLS = {
    "symbol",
    "pe_ratio",
    "market_cap_cr",
    "sector",
    "industry",
    "dividend_yield",
    "market_cap_tier",
    "currency",
    "exchange",
    "long_name",
}


# ── Fixture loader ────────────────────────────────────────────────────────────

def _load_fundamentals_fixture() -> pd.DataFrame:
    """
    Load the fundamentals fixture CSV into a DataFrame.

    All symbols are stored in a single file — one row per symbol.
    Filtering to requested symbols happens in get_fundamentals.

    Raises:
        FileNotFoundError: if fundamentals.csv does not exist
        ValueError: if CSV is missing required columns or is empty
    """
    if not FUNDAMENTALS_FILE.exists():
        raise FileNotFoundError(
            f"Fundamentals fixture not found: {FUNDAMENTALS_FILE}\n"
            f"Run scripts/download_fixtures.py to generate fixtures."
        )

    df = pd.read_csv(FUNDAMENTALS_FILE)

    if df.empty:
        raise ValueError(
            f"Fundamentals fixture is empty: {FUNDAMENTALS_FILE}"
        )

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Fundamentals fixture missing required columns: {missing}"
        )

    return df


def _row_to_dict(row: pd.Series) -> dict:
    """
    Convert a single DataFrame row to the fundamental data dict schema.

    Handles NaN values gracefully — converts to None for JSON serialisation.
    pe_ratio and market_cap_cr may be NaN if yFinance did not return them.

    Args:
        row: A single row from the fundamentals DataFrame

    Returns:
        dict matching the FundamentalData schema in orchestrator/state.py
    """
    def _safe(val):
        """Convert NaN to None, keep all other values as-is."""
        if pd.isna(val):
            return None
        return val

    return {
        "pe_ratio":        _safe(row.get("pe_ratio")),
        "market_cap_cr":   _safe(row.get("market_cap_cr")),
        "sector":          str(row.get("sector", "Unknown")),
        "industry":        str(row.get("industry", "Unknown")),
        "dividend_yield":  float(row.get("dividend_yield", 0.0) or 0.0),
        "market_cap_tier": str(row.get("market_cap_tier", "unknown")),
        "currency":        str(row.get("currency", "INR")),
        "exchange":        str(row.get("exchange", "NSI")),
        "long_name":       _safe(row.get("long_name")),
    }


# ── Main tool function ────────────────────────────────────────────────────────

def get_fundamentals(symbols: list[str]) -> dict:
    """
    Retrieve fundamental data for a list of NSE symbols.

    This is the implementation of the get_fundamentals MCP tool.
    Called by server.py — never called directly by other servers.

    The sector field is critical for the Compliance server's sector
    concentration checks. The industry field provides finer-grained
    classification for future compliance rules.

    Args:
        symbols: List of NSE ticker symbols e.g. ["RELIANCE.NS", "TCS.NS"]

    Returns:
        dict matching the get_fundamentals tool output schema:
            fundamentals: dict[symbol, FundamentalData]
            source:       str — always "fixture" in v1
            missing:      list[str] — symbols not found in fixture

    Raises:
        ValueError: if symbols list is empty
        FileNotFoundError: if fundamentals fixture does not exist
    """
    # ── Input validation ──────────────────────────────────────────
    if not symbols:
        raise ValueError("symbols list must not be empty")

    # ── Load fixture ──────────────────────────────────────────────
    df = _load_fundamentals_fixture()

    # ── Build symbol index for fast lookup ────────────────────────
    # Index by symbol column — one row per symbol
    df_indexed = df.set_index("symbol")

    # ── Build output ──────────────────────────────────────────────
    fundamentals: dict[str, dict] = {}
    missing: list[str] = []

    for symbol in symbols:
        if symbol in df_indexed.index:
            row = df_indexed.loc[symbol]
            fundamentals[symbol] = _row_to_dict(row)
        else:
            # Symbol not in fixture — record as missing, do not raise
            # Allows partial results when some symbols lack fundamentals
            missing.append(symbol)

    # Warn if any symbols were not found — not a hard failure
    # The orchestrator can proceed with partial fundamentals
    # Compliance server will flag missing sector_map entries
    if missing:
        import warnings
        warnings.warn(
            f"Fundamentals not found for symbols: {missing}. "
            f"These symbols will be absent from the fundamentals output. "
            f"Run scripts/download_fixtures.py to refresh fixtures.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "fundamentals": fundamentals,
        "source":       "fixture",
        "missing":      missing,
    }