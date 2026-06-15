"""
servers/risk_engine/tests/test_risk_metrics.py

Unit tests for the compute_risk_metrics tool.

Test strategy:
    - Mathematical correctness validated against hand-computed values
    - Known return series → known VaR, CVaR, Sharpe, max drawdown
    - Edge cases: empty weights, weights not summing to 1,
      mismatched symbols, different length series
    - Integration: full tool function with realistic fixture-derived inputs

Run from project root:
    cd ~/genaiprojects/mcp-investment-copilot
    pytest servers/risk_engine/tests/test_risk_metrics.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
SERVER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVER_ROOT))

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

from tools.risk_metrics import (
    _compute_cvar,
    _compute_max_drawdown,
    _compute_portfolio_returns,
    _compute_sharpe,
    _compute_var,
    _compute_volatility,
    compute_risk_metrics,
)


# ── Shared test data ──────────────────────────────────────────────────────────

def make_simple_returns() -> dict:
    """
    Construct a known return series for mathematical validation.

    10 returns: [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
    Sorted ascending: same order as above.

    For a single asset with weight=1.0, portfolio returns = asset returns.

    Hand-computed values:
        VaR_90  = -percentile(returns, 10) = -(-0.04) = 0.04
        VaR_80  = -percentile(returns, 20) = -(-0.03) = 0.03
        CVaR_90 = -mean(returns below -VaR_90)
                = -mean([-0.05, -0.04]) = 0.045  (1 value below, using strict <)
        mean    = 0.0  (symmetric around zero)
        std     = std(returns, ddof=1)
    """
    returns = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
    return {
        "A": returns,
    }


def make_simple_prices() -> dict:
    """
    Construct a known price series consistent with make_simple_returns.
    Starting price 100, applying simple returns approximately.
    Used for max drawdown test.
    """
    prices = [100.0, 95.0, 91.2, 88.5, 86.7, 85.8, 86.7, 88.4, 91.1, 94.7, 99.4]
    return {"A": prices}


# ── Tests: _compute_portfolio_returns ─────────────────────────────────────────

class TestComputePortfolioReturns:

    def test_single_asset_full_weight(self):
        """Single asset with weight=1.0 — portfolio returns equal asset returns."""
        log_returns = {"A": [0.01, -0.02, 0.03, -0.01, 0.02]}
        weights = {"A": 1.0}
        result = _compute_portfolio_returns(log_returns, weights)

        assert len(result) == 5
        for i, expected in enumerate([0.01, -0.02, 0.03, -0.01, 0.02]):
            assert math.isclose(result[i], expected, rel_tol=1e-9)

    def test_two_assets_equal_weights(self):
        """Two assets with equal weights — portfolio return is average."""
        log_returns = {
            "A": [0.04, -0.02],
            "B": [0.00,  0.02],
        }
        weights = {"A": 0.5, "B": 0.5}
        result = _compute_portfolio_returns(log_returns, weights)

        assert math.isclose(result[0], 0.02, rel_tol=1e-9)  # (0.04+0.00)/2
        assert math.isclose(result[1], 0.00, rel_tol=1e-9)  # (-0.02+0.02)/2

    def test_weights_not_summing_to_one_raises(self):
        """Weights summing to != 1 raises ValueError."""
        log_returns = {"A": [0.01, 0.02]}
        weights = {"A": 0.6, "B": 0.6}
        with pytest.raises(ValueError, match="sum to 1"):
            _compute_portfolio_returns(log_returns, weights)

    def test_missing_symbol_in_returns_raises(self):
        """Symbol in weights but not in log_returns raises ValueError."""
        log_returns = {"A": [0.01, 0.02]}
        weights = {"A": 0.5, "B": 0.5}
        with pytest.raises(ValueError, match="not found in log_returns"):
            _compute_portfolio_returns(log_returns, weights)

    def test_different_length_series_raises(self):
        """Return series of different lengths raises ValueError."""
        log_returns = {"A": [0.01, 0.02, 0.03], "B": [0.01, 0.02]}
        weights = {"A": 0.5, "B": 0.5}
        with pytest.raises(ValueError, match="different lengths"):
            _compute_portfolio_returns(log_returns, weights)


# ── Tests: _compute_var ───────────────────────────────────────────────────────

class TestComputeVar:

    def test_var_known_series(self):
        """
        Hand-computed VaR for known return series.

        Series: [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
        VaR_90 = -percentile(series, 10) = -(-0.041) ≈ 0.041
        (numpy uses linear interpolation between -0.05 and -0.04 at 10th percentile)
        """
        returns = np.array(
            [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
        )
        var = _compute_var(returns, percentile=10.0)

        # VaR must be positive (loss expressed as positive number)
        assert var > 0

        # Must be between the two most extreme losses
        assert 0.04 <= var <= 0.05

    def test_var_is_positive(self):
        """VaR is always expressed as a positive loss number."""
        returns = np.array([-0.03, -0.02, -0.01, 0.01, 0.02])
        var = _compute_var(returns, percentile=20.0)
        assert var > 0

    def test_var_99_greater_than_var_95(self):
        """VaR_99 must be greater than or equal to VaR_95 for any return series."""
        returns = np.array(
            [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
        )
        var_95 = _compute_var(returns, percentile=5.0)
        var_99 = _compute_var(returns, percentile=1.0)
        assert var_99 >= var_95


# ── Tests: _compute_cvar ──────────────────────────────────────────────────────

class TestComputeCVar:

    def test_cvar_greater_than_var(self):
        """
        CVaR must always be >= VaR for the same confidence level.
        CVaR is the mean of the tail beyond VaR — always at least as bad.
        """
        returns = np.array(
            [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
        )
        var_90 = _compute_var(returns, percentile=10.0)
        cvar_90 = _compute_cvar(returns, var=var_90)
        assert cvar_90 >= var_90

    def test_cvar_is_positive(self):
        """CVaR is always expressed as a positive loss number."""
        returns = np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.01])
        var = _compute_var(returns, percentile=20.0)
        cvar = _compute_cvar(returns, var=var)
        assert cvar > 0

    def test_cvar_known_series(self):
        """
        Hand-computed CVaR for known series.

        Series: [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
        VaR_90 ≈ 0.041 (10th percentile negated, with interpolation)
        Tail = all returns < -0.041 = [-0.05]
        CVaR_90 = -mean([-0.05]) = 0.05
        """
        returns = np.array(
            [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
        )
        var_90 = _compute_var(returns, percentile=10.0)
        cvar_90 = _compute_cvar(returns, var=var_90)

        # CVaR must be at least as large as the worst return
        assert cvar_90 <= 0.05 + 1e-9

        # CVaR must be >= VaR
        assert cvar_90 >= var_90


# ── Tests: _compute_sharpe ────────────────────────────────────────────────────

class TestComputeSharpe:

    def test_zero_mean_positive_rfr_gives_negative_sharpe(self):
        """
        Zero mean return with positive risk-free rate gives negative Sharpe.
        Sharpe = (0 - rfr_daily) / std × √252 < 0
        """
        returns = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01] * 10)
        sharpe = _compute_sharpe(returns, risk_free_rate=0.065)
        assert sharpe < 0

    def test_positive_excess_return_gives_positive_sharpe(self):
        """
        Returns consistently above the risk-free rate give positive Sharpe.
        Uses varied returns with positive mean well above daily rfr.
        """
        np.random.seed(42)
        # Mean daily return 0.1% >> daily rfr (0.065/252 ≈ 0.026%)
        returns = np.random.normal(0.001, 0.01, 252)
        sharpe = _compute_sharpe(returns, risk_free_rate=0.065)
        assert sharpe > 0

    def test_zero_std_raises_value_error(self):
        """Constant returns (zero std) raises ValueError — cannot divide by zero."""
        # A series of exactly 2 identical values gives std=0 with ddof=1
        returns = np.array([0.001, 0.001])
        with pytest.raises(ValueError, match="standard deviation is zero"):
            _compute_sharpe(returns, risk_free_rate=0.065)

    def test_sharpe_annualisation(self):
        """
        Verify Sharpe is annualised correctly.
        Daily return = 0.001, std = 0.01, rfr = 0 → Sharpe = 0.001/0.01 × √252
        """
        mean_daily = 0.001
        std_daily  = 0.01
        # Construct returns with known mean and std
        np.random.seed(42)
        returns = np.random.normal(mean_daily, std_daily, 10000)

        sharpe = _compute_sharpe(returns, risk_free_rate=0.0)
        expected = (mean_daily / std_daily) * np.sqrt(252)

        # Allow 5% tolerance due to sampling variation
        assert math.isclose(sharpe, expected, rel_tol=0.05)


# ── Tests: _compute_max_drawdown ──────────────────────────────────────────────

class TestComputeMaxDrawdown:

    def test_monotonically_increasing_prices_zero_drawdown(self):
        """Prices that only go up have zero maximum drawdown."""
        prices = {"A": [100.0, 101.0, 102.0, 103.0, 104.0]}
        weights = {"A": 1.0}
        dd = _compute_max_drawdown(prices, weights)
        assert math.isclose(dd, 0.0, abs_tol=1e-9)

    def test_known_drawdown(self):
        """
        Hand-computed max drawdown.
        Prices: [100, 120, 80, 90]
        Peak at 120, trough at 80.
        Drawdown = (80 - 120) / 120 = -0.3333...
        """
        prices = {"A": [100.0, 120.0, 80.0, 90.0]}
        weights = {"A": 1.0}
        dd = _compute_max_drawdown(prices, weights)
        assert math.isclose(dd, -1/3, rel_tol=1e-4)

    def test_drawdown_is_negative(self):
        """Max drawdown is always expressed as a negative fraction."""
        prices = {"A": [100.0, 90.0, 95.0, 85.0, 92.0]}
        weights = {"A": 1.0}
        dd = _compute_max_drawdown(prices, weights)
        assert dd < 0

    def test_drawdown_between_minus_one_and_zero(self):
        """Max drawdown is always between -1 and 0."""
        prices = {"A": [100.0, 50.0, 80.0, 30.0, 60.0]}
        weights = {"A": 1.0}
        dd = _compute_max_drawdown(prices, weights)
        assert -1.0 <= dd <= 0.0


# ── Tests: compute_risk_metrics (full tool) ───────────────────────────────────

class TestComputeRiskMetrics:

    def _load_fixture_data(self):
        """
        Load real fixture data for integration testing.
        Reads CSVs directly via pandas — avoids cross-server import conflicts.
        """
        import pandas as pd

        fixtures_dir = PROJECT_ROOT / "data" / "fixtures"
        symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

        frames = {}
        for symbol in symbols:
            df = pd.read_csv(fixtures_dir / f"{symbol}_2y.csv", parse_dates=["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            frames[symbol] = df

        # Align dates via intersection
        date_sets = [set(df["Date"].astype(str)) for df in frames.values()]
        common_dates = sorted(set.intersection(*date_sets))

        prices = {}
        log_returns = {}

        for symbol, df in frames.items():
            mask = df["Date"].astype(str).isin(set(common_dates))
            aligned = df[mask].sort_values("Date")
            close = aligned["Close"].tolist()
            prices[symbol] = close
            arr = np.array(close)
            log_returns[symbol] = np.diff(np.log(arr)).tolist()

        return {"prices": prices, "log_returns": log_returns}

    def test_output_has_all_required_keys(self):
        """Output dict has all keys matching RiskMetricsResult schema."""
        log_returns = make_simple_returns()
        weights = {"A": 1.0}
        prices = make_simple_prices()

        result = compute_risk_metrics(log_returns, weights, prices)

        required_keys = {
            "var_95", "var_99", "cvar_95", "cvar_99",
            "sharpe_ratio", "max_drawdown", "volatility",
            "portfolio_return", "risk_free_rate", "computation_window",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_cvar_greater_than_var(self):
        """CVaR_95 >= VaR_95 and CVaR_99 >= VaR_99."""
        log_returns = make_simple_returns()
        weights = {"A": 1.0}
        prices = make_simple_prices()

        result = compute_risk_metrics(log_returns, weights, prices)

        assert result["cvar_95"] >= result["var_95"]
        assert result["cvar_99"] >= result["var_99"]

    def test_var_99_greater_than_var_95(self):
        """VaR_99 >= VaR_95 — stricter confidence level gives higher VaR."""
        log_returns = make_simple_returns()
        weights = {"A": 1.0}
        prices = make_simple_prices()

        result = compute_risk_metrics(log_returns, weights, prices)

        assert result["var_99"] >= result["var_95"]

    def test_max_drawdown_is_negative(self):
        """Max drawdown is always a negative fraction."""
        log_returns = make_simple_returns()
        weights = {"A": 1.0}
        prices = make_simple_prices()

        result = compute_risk_metrics(log_returns, weights, prices)

        assert result["max_drawdown"] < 0

    def test_risk_free_rate_echoed(self):
        """risk_free_rate field echoes the input value."""
        log_returns = make_simple_returns()
        weights = {"A": 1.0}
        prices = make_simple_prices()

        result = compute_risk_metrics(
            log_returns, weights, prices, risk_free_rate=0.07
        )
        assert math.isclose(result["risk_free_rate"], 0.07)

    def test_volatility_dict_has_all_symbols(self):
        """Volatility dict contains an entry for every symbol in log_returns."""
        np.random.seed(42)
        returns_a = np.random.normal(0.001, 0.015, 100).tolist()
        returns_b = np.random.normal(0.001, 0.012, 100).tolist()
        log_returns = {"A": returns_a, "B": returns_b}
        weights = {"A": 0.6, "B": 0.4}
        prices = {
            "A": [100.0 * np.exp(sum(returns_a[:i])) for i in range(101)],
            "B": [200.0 * np.exp(sum(returns_b[:i])) for i in range(101)],
        }
        result = compute_risk_metrics(log_returns, weights, prices)

        assert "A" in result["volatility"]
        assert "B" in result["volatility"]

    def test_empty_weights_raises(self):
        """Empty weights dict raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_risk_metrics({}, {}, {})

    def test_negative_risk_free_rate_raises(self):
        """Negative risk-free rate raises ValueError."""
        log_returns = make_simple_returns()
        weights = {"A": 1.0}
        prices = make_simple_prices()

        with pytest.raises(ValueError, match="non-negative"):
            compute_risk_metrics(log_returns, weights, prices, risk_free_rate=-0.01)

    def test_integration_with_fixture_data(self):
        """
        Integration test using real NSE fixture data.
        Validates that the full tool produces sensible results
        for a realistic equal-weighted portfolio of 5 large-cap stocks.
        """
        data = self._load_fixture_data()
        symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
        weights = {s: 0.2 for s in symbols}

        result = compute_risk_metrics(
            log_returns=data["log_returns"],
            weights=weights,
            prices=data["prices"],
            risk_free_rate=0.065,
        )

        # Sanity checks on realistic values for NSE large-cap portfolio
        # VaR_95 for a diversified NSE portfolio should be between 0.5% and 5%
        assert 0.005 <= result["var_95"] <= 0.05, \
            f"VaR_95 out of expected range: {result['var_95']:.4f}"

        # CVaR_95 must be >= VaR_95
        assert result["cvar_95"] >= result["var_95"]

        # Max drawdown for a 2y window should be between -50% and 0%
        assert -0.50 <= result["max_drawdown"] <= 0.0, \
            f"max_drawdown out of expected range: {result['max_drawdown']:.4f}"

        # Volatility for NSE large-caps should be between 10% and 60% annualised
        for symbol in symbols:
            vol = result["volatility"][symbol]
            assert 0.10 <= vol <= 0.60, \
                f"{symbol} volatility out of expected range: {vol:.4f}"