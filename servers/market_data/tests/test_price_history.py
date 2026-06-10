"""
servers/market_data/tests/test_price_history.py

Unit tests for the get_price_history tool.

Test strategy:
    - All tests use MARKET_DATA_SOURCE=fixture — no yFinance calls
    - Fixture data is read from data/fixtures/ — committed to repo
    - Mathematical correctness validated against hand-computed values
    - Edge cases: missing fixture, empty symbols, invalid period,
      zero price handling, date alignment across symbols

Run from project root:
    cd ~/genaiprojects/mcp-investment-copilot
    pytest servers/market_data/tests/test_price_history.py -v
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
# Add server root to sys.path so tools/ is importable
SERVER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVER_ROOT))

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"

from tools.price_history import (
    _clean_prices,
    _compute_log_returns,
    _compute_simple_returns,
    _load_fixture,
    get_price_history,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def set_fixture_source(monkeypatch):
    """
    Ensure all tests use fixture data — never live yFinance.
    autouse=True means this applies to every test in this file automatically.
    """
    monkeypatch.setenv("MARKET_DATA_SOURCE", "fixture")


# ── Tests: _compute_log_returns ───────────────────────────────────────────────

class TestComputeLogReturns:

    def test_known_prices_produce_known_returns(self):
        """
        Hand-computed log-returns for a known price series.
        ln(102/100) = 0.019803
        ln(101/102) = -0.009852
        ln(105/101) = 0.038840
        """
        prices = [100.0, 102.0, 101.0, 105.0]
        returns = _compute_log_returns(prices)

        assert len(returns) == 3
        assert math.isclose(returns[0],  0.019803, rel_tol=1e-4)
        assert math.isclose(returns[1], -0.009852, rel_tol=1e-4)
        assert math.isclose(returns[2],  0.038840, rel_tol=1e-4)

    def test_returns_length_is_n_minus_one(self):
        """Log-returns series is always one shorter than price series."""
        prices = [100.0, 102.0, 104.0, 103.0, 107.0]
        returns = _compute_log_returns(prices)
        assert len(returns) == len(prices) - 1

    def test_constant_prices_produce_zero_returns(self):
        """Constant price series produces all-zero log-returns."""
        prices = [100.0] * 5
        returns = _compute_log_returns(prices)
        assert all(math.isclose(r, 0.0, abs_tol=1e-10) for r in returns)

    def test_zero_price_raises_value_error(self):
        """Zero price in series raises ValueError — cannot compute log(0)."""
        prices = [100.0, 0.0, 102.0]
        with pytest.raises(ValueError, match="zero or negative"):
            _compute_log_returns(prices)

    def test_negative_price_raises_value_error(self):
        """Negative price raises ValueError — not valid for equities."""
        prices = [100.0, -50.0, 102.0]
        with pytest.raises(ValueError, match="zero or negative"):
            _compute_log_returns(prices)

    def test_all_returns_are_finite(self):
        """All returns must be finite — no NaN or inf."""
        prices = [100.0, 102.0, 98.0, 105.0, 103.0]
        returns = _compute_log_returns(prices)
        assert all(math.isfinite(r) for r in returns)


# ── Tests: _compute_simple_returns ───────────────────────────────────────────

class TestComputeSimpleReturns:

    def test_known_prices_produce_known_returns(self):
        """
        Hand-computed simple returns for a known price series.
        (102-100)/100 = 0.02
        (101-102)/102 = -0.009804
        """
        prices = [100.0, 102.0, 101.0]
        returns = _compute_simple_returns(prices)

        assert len(returns) == 2
        assert math.isclose(returns[0],  0.02,     rel_tol=1e-4)
        assert math.isclose(returns[1], -0.009804, rel_tol=1e-4)

    def test_returns_length_is_n_minus_one(self):
        prices = [100.0, 102.0, 104.0, 103.0]
        returns = _compute_simple_returns(prices)
        assert len(returns) == len(prices) - 1


# ── Tests: _clean_prices ──────────────────────────────────────────────────────

class TestCleanPrices:

    def _make_df(self, closes: list) -> pd.DataFrame:
        """Helper — create a minimal DataFrame with a Close column."""
        return pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
            "Close": closes,
        })

    def test_zero_price_is_forward_filled(self):
        """Zero price on day t gets replaced by price from day t-1."""
        df = self._make_df([100.0, 0.0, 102.0])
        cleaned = _clean_prices(df, "TEST.NS")
        assert cleaned["Close"].iloc[1] == 100.0

    def test_nan_price_is_forward_filled(self):
        """NaN price on day t gets replaced by price from day t-1."""
        df = self._make_df([100.0, float("nan"), 102.0])
        cleaned = _clean_prices(df, "TEST.NS")
        assert cleaned["Close"].iloc[1] == 100.0

    def test_nan_at_start_is_back_filled(self):
        """NaN at start of series is back-filled from the first valid price."""
        df = self._make_df([float("nan"), 100.0, 102.0])
        cleaned = _clean_prices(df, "TEST.NS")
        assert cleaned["Close"].iloc[0] == 100.0

    def test_clean_prices_unchanged(self):
        """Valid prices pass through unchanged."""
        prices = [100.0, 102.0, 101.0, 105.0]
        df = self._make_df(prices)
        cleaned = _clean_prices(df, "TEST.NS")
        assert cleaned["Close"].tolist() == prices

    def test_all_nan_raises_value_error(self):
        """All-NaN series cannot be filled — raises ValueError."""
        df = self._make_df([float("nan"), float("nan"), float("nan")])
        with pytest.raises(ValueError, match="unresolvable NaN"):
            _clean_prices(df, "TEST.NS")


# ── Tests: _load_fixture ──────────────────────────────────────────────────────

class TestLoadFixture:

    def test_loads_known_symbol(self):
        """RELIANCE.NS fixture loads successfully."""
        df = _load_fixture("RELIANCE.NS", "2y")
        assert not df.empty
        assert "Close" in df.columns
        assert "Date" in df.columns

    def test_row_count_is_reasonable(self):
        """2y fixture should have approximately 490-510 trading days."""
        df = _load_fixture("RELIANCE.NS", "2y")
        assert 480 <= len(df) <= 520

    def test_missing_fixture_raises_file_not_found(self):
        """Non-existent symbol raises FileNotFoundError with helpful message."""
        with pytest.raises(FileNotFoundError, match="download_fixtures.py"):
            _load_fixture("INVALID.NS", "2y")

    def test_no_zero_prices_after_loading(self):
        """After loading and cleaning, no zero closing prices remain."""
        df = _load_fixture("RELIANCE.NS", "2y")
        assert (df["Close"] > 0).all()

    def test_no_nan_prices_after_loading(self):
        """After loading and cleaning, no NaN closing prices remain."""
        df = _load_fixture("RELIANCE.NS", "2y")
        assert not df["Close"].isna().any()

    def test_dates_are_sorted(self):
        """Dates are in ascending chronological order."""
        df = _load_fixture("RELIANCE.NS", "2y")
        dates = pd.to_datetime(df["Date"])
        assert dates.is_monotonic_increasing


# ── Tests: get_price_history ──────────────────────────────────────────────────

class TestGetPriceHistory:

    def test_single_symbol_returns_correct_structure(self):
        """Output dict has all required keys for a single symbol."""
        result = get_price_history(["RELIANCE.NS"], period="2y")

        assert "prices" in result
        assert "log_returns" in result
        assert "dates" in result
        assert "source" in result
        assert "period" in result

    def test_source_is_fixture(self):
        """Source field confirms fixture data was used."""
        result = get_price_history(["RELIANCE.NS"], period="2y")
        assert result["source"] == "fixture"

    def test_period_is_echoed(self):
        """Period field echoes the requested period."""
        result = get_price_history(["RELIANCE.NS"], period="2y")
        assert result["period"] == "2y"

    def test_prices_and_returns_length_relationship(self):
        """
        Returns series is one shorter than prices series.
        prices:  [P0, P1, P2, ..., Pn]   length = n+1
        returns: [r1, r2, ..., rn]        length = n
        dates:   aligned to returns       length = n
        """
        result = get_price_history(["RELIANCE.NS"], period="2y")
        prices = result["prices"]["RELIANCE.NS"]
        returns = result["log_returns"]["RELIANCE.NS"]
        dates = result["dates"]

        assert len(returns) == len(prices) - 1
        assert len(dates) == len(returns)

    def test_multiple_symbols_date_alignment(self):
        """
        All symbols have the same number of dates after alignment.
        ICICIBANK.NS had 497 rows vs 498 for others — intersection handles this.
        """
        symbols = ["RELIANCE.NS", "TCS.NS", "ICICIBANK.NS", "SBIN.NS"]
        result = get_price_history(symbols, period="2y")

        lengths = [len(result["log_returns"][s]) for s in symbols]
        assert len(set(lengths)) == 1  # all lengths identical

    def test_all_symbols_present_in_output(self):
        """Every requested symbol appears in prices and log_returns."""
        symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        result = get_price_history(symbols, period="2y")

        for s in symbols:
            assert s in result["prices"]
            assert s in result["log_returns"]

    def test_all_returns_are_finite(self):
        """No NaN or infinite values in log-returns for any symbol."""
        symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
        result = get_price_history(symbols, period="2y")

        for s in symbols:
            returns = result["log_returns"][s]
            assert all(math.isfinite(r) for r in returns), \
                f"Non-finite return found for {s}"

    def test_simple_return_type(self):
        """return_type='simple' produces simple returns, not log-returns."""
        result_log    = get_price_history(["RELIANCE.NS"], return_type="log")
        result_simple = get_price_history(["RELIANCE.NS"], return_type="simple")

        log_r    = result_log["log_returns"]["RELIANCE.NS"][0]
        simple_r = result_simple["log_returns"]["RELIANCE.NS"][0]

        # Simple and log returns are close but not identical for non-zero returns
        assert not math.isclose(log_r, simple_r, rel_tol=1e-6)

    def test_empty_symbols_raises_value_error(self):
        """Empty symbols list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            get_price_history([])

    def test_invalid_period_raises_value_error(self):
        """Invalid period string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid period"):
            get_price_history(["RELIANCE.NS"], period="10y")

    def test_invalid_return_type_raises_value_error(self):
        """Invalid return_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid return_type"):
            get_price_history(["RELIANCE.NS"], return_type="arithmetic")

    def test_invalid_symbol_raises_file_not_found(self):
        """Symbol with no fixture raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_price_history(["INVALID.NS"])

    def test_prices_are_all_positive(self):
        """All closing prices are positive — no zero or negative values."""
        result = get_price_history(["RELIANCE.NS", "TCS.NS"])
        for symbol, prices in result["prices"].items():
            assert all(p > 0 for p in prices), \
                f"Non-positive price found for {symbol}"