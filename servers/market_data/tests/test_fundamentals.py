"""
servers/market_data/tests/test_fundamentals.py

Unit tests for the get_fundamentals tool.

Test strategy:
    - All tests read from data/fixtures/fundamentals.csv
    - Validates correct structure, field types, and known values
    - Validates NaN handling — pe_ratio and market_cap_cr may be None
    - Validates missing symbol handling — recorded in missing list
    - Validates sector field integrity — critical for Compliance server

Run from project root:
    cd ~/genaiprojects/mcp-investment-copilot
    pytest servers/market_data/tests/test_fundamentals.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
SERVER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVER_ROOT))

from tools.fundamentals import (
    _load_fundamentals_fixture,
    _row_to_dict,
    get_fundamentals,
)


# ── Known fixture values ──────────────────────────────────────────────────────
# These values are sourced from the downloaded fundamentals.csv fixture.
# If fixtures are refreshed, these may need updating.

KNOWN_SECTORS = {
    "RELIANCE.NS":   "Energy",
    "TCS.NS":        "Technology",
    "INFY.NS":       "Technology",
    "HDFCBANK.NS":   "Financial Services",
    "ICICIBANK.NS":  "Financial Services",
    "ADANIENT.NS":   "Energy",
    "BAJFINANCE.NS": "Financial Services",
    "BHARTIARTL.NS": "Communication Services",
    "SBIN.NS":       "Financial Services",
    "LT.NS":         "Industrials",
}

ALL_SYMBOLS = list(KNOWN_SECTORS.keys())


# ── Tests: _load_fundamentals_fixture ─────────────────────────────────────────

class TestLoadFundamentalsFixture:

    def test_fixture_loads_successfully(self):
        """fundamentals.csv loads without error."""
        df = _load_fundamentals_fixture()
        assert not df.empty

    def test_fixture_has_required_columns(self):
        """All required columns are present in the fixture."""
        df = _load_fundamentals_fixture()
        required = {
            "symbol", "pe_ratio", "market_cap_cr", "sector",
            "industry", "dividend_yield", "market_cap_tier",
            "currency", "exchange", "long_name",
        }
        assert required.issubset(set(df.columns))

    def test_fixture_has_all_ten_symbols(self):
        """All 10 downloaded symbols are present in the fixture."""
        df = _load_fundamentals_fixture()
        symbols_in_fixture = set(df["symbol"].tolist())
        for symbol in ALL_SYMBOLS:
            assert symbol in symbols_in_fixture, \
                f"{symbol} missing from fundamentals fixture"

    def test_fixture_row_count(self):
        """Fixture has exactly 10 rows — one per symbol."""
        df = _load_fundamentals_fixture()
        assert len(df) == 10


# ── Tests: _row_to_dict ───────────────────────────────────────────────────────

class TestRowToDict:

    def _get_row(self, symbol: str):
        """Helper — get a single row from the fixture for a known symbol."""
        import pandas as pd
        df = _load_fundamentals_fixture()
        df_indexed = df.set_index("symbol")
        return df_indexed.loc[symbol]

    def test_output_has_all_required_keys(self):
        """_row_to_dict output has all required keys."""
        row = self._get_row("RELIANCE.NS")
        result = _row_to_dict(row)

        required_keys = {
            "pe_ratio", "market_cap_cr", "sector", "industry",
            "dividend_yield", "market_cap_tier", "currency",
            "exchange", "long_name",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_nan_pe_ratio_returns_none(self):
        """
        pe_ratio may be NaN in yFinance data.
        _row_to_dict must convert NaN to None — not leave as float NaN
        which would fail JSON serialisation.
        """
        import pandas as pd
        import numpy as np

        # Create a synthetic row with NaN pe_ratio
        row = pd.Series({
            "pe_ratio":        float("nan"),
            "market_cap_cr":   500000.0,
            "sector":          "Technology",
            "industry":        "IT Services",
            "dividend_yield":  0.01,
            "market_cap_tier": "large",
            "currency":        "INR",
            "exchange":        "NSI",
            "long_name":       "Test Corp",
        })
        result = _row_to_dict(row)
        assert result["pe_ratio"] is None

    def test_nan_market_cap_returns_none(self):
        """market_cap_cr NaN converts to None."""
        import pandas as pd

        row = pd.Series({
            "pe_ratio":        25.0,
            "market_cap_cr":   float("nan"),
            "sector":          "Technology",
            "industry":        "IT Services",
            "dividend_yield":  0.01,
            "market_cap_tier": "large",
            "currency":        "INR",
            "exchange":        "NSI",
            "long_name":       "Test Corp",
        })
        result = _row_to_dict(row)
        assert result["market_cap_cr"] is None

    def test_sector_is_string(self):
        """Sector field is always a string — never NaN or None."""
        row = self._get_row("TCS.NS")
        result = _row_to_dict(row)
        assert isinstance(result["sector"], str)
        assert len(result["sector"]) > 0

    def test_dividend_yield_defaults_to_zero(self):
        """dividend_yield defaults to 0.0 if missing or None."""
        import pandas as pd

        row = pd.Series({
            "pe_ratio":        25.0,
            "market_cap_cr":   500000.0,
            "sector":          "Technology",
            "industry":        "IT Services",
            "dividend_yield":  None,
            "market_cap_tier": "large",
            "currency":        "INR",
            "exchange":        "NSI",
            "long_name":       "Test Corp",
        })
        result = _row_to_dict(row)
        assert result["dividend_yield"] == 0.0


# ── Tests: get_fundamentals ───────────────────────────────────────────────────

class TestGetFundamentals:

    def test_single_symbol_returns_correct_structure(self):
        """Output has fundamentals, source, and missing keys."""
        result = get_fundamentals(["RELIANCE.NS"])

        assert "fundamentals" in result
        assert "source" in result
        assert "missing" in result

    def test_source_is_fixture(self):
        """Source field is always 'fixture' in v1."""
        result = get_fundamentals(["RELIANCE.NS"])
        assert result["source"] == "fixture"

    def test_known_sectors_are_correct(self):
        """
        Sector values for all 10 symbols match known classifications.
        This is the most critical field — consumed by Compliance server
        for sector concentration checks.
        """
        result = get_fundamentals(ALL_SYMBOLS)

        for symbol, expected_sector in KNOWN_SECTORS.items():
            actual_sector = result["fundamentals"][symbol]["sector"]
            assert actual_sector == expected_sector, \
                f"{symbol}: expected sector '{expected_sector}', got '{actual_sector}'"

    def test_all_symbols_present_in_output(self):
        """Every requested symbol appears in the fundamentals dict."""
        result = get_fundamentals(ALL_SYMBOLS)

        for symbol in ALL_SYMBOLS:
            assert symbol in result["fundamentals"], \
                f"{symbol} missing from fundamentals output"

    def test_missing_symbol_in_missing_list(self):
        """Symbol not in fixture appears in missing list, not in fundamentals."""
        result = get_fundamentals(["RELIANCE.NS", "INVALID.NS"])

        assert "INVALID.NS" in result["missing"]
        assert "INVALID.NS" not in result["fundamentals"]
        assert "RELIANCE.NS" in result["fundamentals"]

    def test_missing_list_empty_when_all_found(self):
        """missing list is empty when all symbols are found."""
        result = get_fundamentals(ALL_SYMBOLS)
        assert result["missing"] == []

    def test_market_cap_tier_is_large_for_all_symbols(self):
        """All 10 symbols are classified as large cap per SEBI definition."""
        result = get_fundamentals(ALL_SYMBOLS)

        for symbol in ALL_SYMBOLS:
            tier = result["fundamentals"][symbol]["market_cap_tier"]
            assert tier == "large", \
                f"{symbol}: expected tier 'large', got '{tier}'"

    def test_pe_ratio_is_float_or_none(self):
        """pe_ratio is either a float or None — never NaN."""
        import math
        result = get_fundamentals(ALL_SYMBOLS)

        for symbol in ALL_SYMBOLS:
            pe = result["fundamentals"][symbol]["pe_ratio"]
            if pe is not None:
                assert isinstance(pe, float)
                assert math.isfinite(pe), \
                    f"{symbol}: pe_ratio is not finite: {pe}"

    def test_market_cap_is_positive_or_none(self):
        """market_cap_cr is either a positive float or None."""
        result = get_fundamentals(ALL_SYMBOLS)

        for symbol in ALL_SYMBOLS:
            mc = result["fundamentals"][symbol]["market_cap_cr"]
            if mc is not None:
                assert mc > 0, \
                    f"{symbol}: market_cap_cr is not positive: {mc}"

    def test_empty_symbols_raises_value_error(self):
        """Empty symbols list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            get_fundamentals([])

    def test_partial_result_when_one_symbol_missing(self):
        """
        When one symbol is missing, valid symbols still return results.
        System does not fail entirely — partial results are usable.
        """
        result = get_fundamentals(["RELIANCE.NS", "TCS.NS", "INVALID.NS"])

        assert "RELIANCE.NS" in result["fundamentals"]
        assert "TCS.NS" in result["fundamentals"]
        assert "INVALID.NS" not in result["fundamentals"]
        assert "INVALID.NS" in result["missing"]

    def test_sector_field_is_never_empty_string(self):
        """Sector field is never an empty string for valid symbols."""
        result = get_fundamentals(ALL_SYMBOLS)

        for symbol in ALL_SYMBOLS:
            sector = result["fundamentals"][symbol]["sector"]
            assert sector != "", \
                f"{symbol}: sector is empty string"
            assert sector != "Unknown", \
                f"{symbol}: sector is 'Unknown' — fixture may be stale"