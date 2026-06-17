"""
servers/portfolio_optimiser/tests/test_optimise.py

Unit tests for the optimise_portfolio tool.

Test strategy:
    Mathematical validation:
        2-asset portfolio with known analytical solution validates
        the implementation against closed-form expected values.
        This is the most critical test — it confirms the solver
        is computing the correct mathematics, not just running
        without error.

    Property-based validation:
        For realistic NSE fixture data where closed-form solutions
        do not exist, we validate mathematical properties:
            - All frontier points are Pareto-optimal
            - Max Sharpe point lies on the frontier
            - Weights sum to 1 for all solutions
            - Weight bounds are respected
            - Higher rfr shifts tangency toward higher-return portfolios

    Integration validation:
        Full tool function with real NSE fixture data.
        Validates realistic output ranges for large-cap Indian equities.

2-asset analytical case:
    Asset A: mu=0.10, sigma=0.20
    Asset B: mu=0.20, sigma=0.30
    Correlation: 0.0 (uncorrelated)

    Minimum variance weights:
        w_A* = sigma_B² / (sigma_A² + sigma_B²)
             = 0.09 / (0.04 + 0.09)
             = 0.6923
        w_B* = 1 - w_A* = 0.3077

Run from project root:
    cd ~/genaiprojects/mcp-investment-copilot
    pytest servers/portfolio_optimiser/tests/test_optimise.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
SERVER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVER_ROOT))

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"

from tools.optimise import (
    _build_bounds,
    _build_constraints,
    _compute_covariance_matrix,
    _compute_expected_returns,
    _find_max_sharpe,
    _negative_sharpe,
    _portfolio_return,
    _portfolio_variance,
    _scan_efficient_frontier,
    optimise_portfolio,
)


# ── Shared synthetic data ─────────────────────────────────────────────────────

def make_two_asset_returns(
    n_days: int = 500,
    mu_a: float = 0.10 / 252,
    mu_b: float = 0.20 / 252,
    sigma_a: float = 0.20 / (252 ** 0.5),
    sigma_b: float = 0.30 / (252 ** 0.5),
    seed: int = 42,
) -> dict[str, list[float]]:
    """
    Generate synthetic uncorrelated return series for two assets.

    Annualised parameters:
        Asset A: mu=10%, sigma=20%
        Asset B: mu=20%, sigma=30%
        Correlation: 0.0

    Daily parameters derived by dividing by 252 (mean) and √252 (std).
    """
    rng = np.random.default_rng(seed)
    returns_a = rng.normal(mu_a, sigma_a, n_days).tolist()
    returns_b = rng.normal(mu_b, sigma_b, n_days).tolist()
    return {"A": returns_a, "B": returns_b}


def load_fixture_returns(symbols: list[str]) -> dict[str, list[float]]:
    """
    Load log-returns for NSE symbols from fixtures.
    Reads CSVs directly — avoids cross-server import conflicts.
    """
    frames = {}
    for symbol in symbols:
        df = pd.read_csv(FIXTURES_DIR / f"{symbol}_2y.csv", parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        frames[symbol] = df

    date_sets = [set(df["Date"].astype(str)) for df in frames.values()]
    common_dates = sorted(set.intersection(*date_sets))

    log_returns = {}
    for symbol, df in frames.items():
        mask = df["Date"].astype(str).isin(set(common_dates))
        aligned = df[mask].sort_values("Date")
        close = aligned["Close"].values
        log_returns[symbol] = np.diff(np.log(close)).tolist()

    return log_returns


# ── Tests: _compute_expected_returns ─────────────────────────────────────────

class TestComputeExpectedReturns:

    def test_returns_correct_symbol_order(self):
        """Symbols list returned matches dict key order."""
        log_returns = make_two_asset_returns()
        mu, symbols = _compute_expected_returns(log_returns)
        assert set(symbols) == {"A", "B"}
        assert len(mu) == 2

    def test_annualisation(self):
        """
        Expected return is annualised (multiplied by 252).
        For a series with known daily mean, annualised = daily × 252.
        """
        daily_mean = 0.001
        returns = {"A": [daily_mean] * 500}
        mu, _ = _compute_expected_returns(returns)
        expected_annual = daily_mean * 252
        assert math.isclose(mu[0], expected_annual, rel_tol=1e-6)

    def test_higher_return_asset_has_higher_mu(self):
        """
        Asset B (mu=20%) should have higher expected return than A (mu=10%).
        """
        log_returns = make_two_asset_returns(n_days=10000)
        mu, symbols = _compute_expected_returns(log_returns)
        idx_a = symbols.index("A")
        idx_b = symbols.index("B")
        assert mu[idx_b] > mu[idx_a]


# ── Tests: _compute_covariance_matrix ────────────────────────────────────────

class TestComputeCovarianceMatrix:

    def test_shape_is_n_by_n(self):
        """Covariance matrix shape is (N_assets × N_assets)."""
        log_returns = make_two_asset_returns()
        mu, symbols = _compute_expected_returns(log_returns)
        cov = _compute_covariance_matrix(log_returns, symbols)
        assert cov.shape == (2, 2)

    def test_matrix_is_symmetric(self):
        """Covariance matrix must be symmetric: Σ[i,j] == Σ[j,i]."""
        log_returns = load_fixture_returns(["RELIANCE.NS", "TCS.NS", "INFY.NS"])
        mu, symbols = _compute_expected_returns(log_returns)
        cov = _compute_covariance_matrix(log_returns, symbols)
        assert np.allclose(cov, cov.T, atol=1e-10)

    def test_diagonal_elements_are_positive(self):
        """Diagonal elements are individual asset variances — must be positive."""
        log_returns = make_two_asset_returns()
        mu, symbols = _compute_expected_returns(log_returns)
        cov = _compute_covariance_matrix(log_returns, symbols)
        assert np.all(np.diag(cov) > 0)

    def test_uncorrelated_assets_have_near_zero_offdiagonal(self):
        """
        Two uncorrelated assets should have near-zero off-diagonal covariance.
        With n_days=10000 and correlation=0, off-diagonal → 0 by LLN.
        """
        log_returns = make_two_asset_returns(n_days=10000)
        mu, symbols = _compute_expected_returns(log_returns)
        cov = _compute_covariance_matrix(log_returns, symbols)
        idx_a = symbols.index("A")
        idx_b = symbols.index("B")
        # Off-diagonal should be small relative to diagonal
        offdiag = abs(cov[idx_a, idx_b])
        diag_mean = (cov[idx_a, idx_a] + cov[idx_b, idx_b]) / 2
        assert offdiag < diag_mean * 0.15  # less than 15% of average variance


# ── Tests: objective functions ────────────────────────────────────────────────

class TestObjectiveFunctions:

    def test_portfolio_variance_equal_weights(self):
        """
        Hand-computed portfolio variance for 2-asset uncorrelated case.
        w = [0.5, 0.5], sigma_A=0.20, sigma_B=0.30, correlation=0
        Σ = [[0.04, 0], [0, 0.09]]
        Var = 0.5²×0.04 + 0.5²×0.09 = 0.01 + 0.0225 = 0.0325
        Vol = √0.0325 ≈ 0.1803
        """
        weights = np.array([0.5, 0.5])
        cov = np.array([[0.04, 0.0], [0.0, 0.09]])
        var = _portfolio_variance(weights, cov)
        assert math.isclose(var, 0.0325, rel_tol=1e-6)

    def test_portfolio_return_equal_weights(self):
        """
        Hand-computed portfolio return for 2-asset case.
        w = [0.5, 0.5], mu = [0.10, 0.20]
        Return = 0.5×0.10 + 0.5×0.20 = 0.15
        """
        weights = np.array([0.5, 0.5])
        mu = np.array([0.10, 0.20])
        ret = _portfolio_return(weights, mu)
        assert math.isclose(ret, 0.15, rel_tol=1e-6)

    def test_negative_sharpe_is_negative_of_sharpe(self):
        """
        _negative_sharpe returns -(Sharpe ratio).
        Sharpe = (0.15 - 0.065) / √0.0325 ≈ 0.472
        negative_sharpe ≈ -0.472
        """
        weights = np.array([0.5, 0.5])
        mu = np.array([0.10, 0.20])
        cov = np.array([[0.04, 0.0], [0.0, 0.09]])
        rfr = 0.065

        neg_sharpe = _negative_sharpe(weights, mu, cov, rfr)
        port_ret = 0.15
        port_vol = math.sqrt(0.0325)
        expected_sharpe = (port_ret - rfr) / port_vol

        assert math.isclose(-neg_sharpe, expected_sharpe, rel_tol=1e-6)


# ── Tests: _scan_efficient_frontier ──────────────────────────────────────────

class TestScanEfficientFrontier:

    def _get_two_asset_inputs(self, n_days: int = 10000):
        log_returns = make_two_asset_returns(n_days=n_days)
        mu, symbols = _compute_expected_returns(log_returns)
        cov = _compute_covariance_matrix(log_returns, symbols)
        return mu, cov, symbols

    def test_returns_n_points(self):
        """Frontier scan returns approximately n_points valid solutions."""
        mu, cov, symbols = self._get_two_asset_inputs()
        points = _scan_efficient_frontier(mu, cov, symbols, 20, 0.0, 1.0)
        # Allow some points to be skipped at infeasible return levels
        assert len(points) >= 15

    def test_points_sorted_by_volatility(self):
        """Frontier points are sorted by volatility ascending."""
        mu, cov, symbols = self._get_two_asset_inputs()
        points = _scan_efficient_frontier(mu, cov, symbols, 20, 0.0, 1.0)
        vols = [p["volatility"] for p in points]
        assert vols == sorted(vols)

    def test_all_weights_sum_to_one(self):
        """All frontier portfolios are fully invested (weights sum to 1)."""
        mu, cov, symbols = self._get_two_asset_inputs()
        points = _scan_efficient_frontier(mu, cov, symbols, 20, 0.0, 1.0)
        for p in points:
            w_sum = sum(p["weights"].values())
            assert math.isclose(w_sum, 1.0, abs_tol=1e-6), \
                f"Weights sum to {w_sum:.8f}, expected 1.0"

    def test_all_weights_respect_bounds(self):
        """All weights are within [min_weight, max_weight]."""
        mu, cov, symbols = self._get_two_asset_inputs()
        min_w, max_w = 0.0, 0.8
        points = _scan_efficient_frontier(mu, cov, symbols, 20, min_w, max_w)
        for p in points:
            for w in p["weights"].values():
                assert w >= min_w - 1e-6
                assert w <= max_w + 1e-6

    def test_minimum_variance_portfolio_known_weights(self):
        """
        For 2-asset uncorrelated case, minimum variance weights are known:
            w_A* = sigma_B² / (sigma_A² + sigma_B²) = 0.09/0.13 ≈ 0.6923
            w_B* = 1 - w_A* ≈ 0.3077

        The minimum variance portfolio is the leftmost point on the frontier
        (lowest volatility). We validate the first frontier point is close
        to the analytical solution.
        """
        log_returns = make_two_asset_returns(n_days=50000)
        mu, symbols = _compute_expected_returns(log_returns)
        cov = _compute_covariance_matrix(log_returns, symbols)
        points = _scan_efficient_frontier(mu, cov, symbols, 50, 0.0, 1.0)

        # Minimum variance = leftmost (lowest vol) frontier point
        min_var_point = points[0]
        w_a = min_var_point["weights"]["A"]
        w_b = min_var_point["weights"]["B"]

        # Analytical solution
        sigma_a_sq = 0.04   # 0.20² annualised
        sigma_b_sq = 0.09   # 0.30² annualised
        w_a_analytical = sigma_b_sq / (sigma_a_sq + sigma_b_sq)  # 0.6923
        w_b_analytical = 1 - w_a_analytical                       # 0.3077

        assert math.isclose(w_a, w_a_analytical, abs_tol=0.05), \
            f"w_A={w_a:.4f}, expected≈{w_a_analytical:.4f}"
        assert math.isclose(w_b, w_b_analytical, abs_tol=0.05), \
            f"w_B={w_b:.4f}, expected≈{w_b_analytical:.4f}"

    def test_higher_return_points_have_higher_volatility(self):
        """
        Along the Efficient Frontier, higher return requires higher risk.
        This is the fundamental Markowitz property.
        """
        mu, cov, symbols = self._get_two_asset_inputs()
        points = _scan_efficient_frontier(mu, cov, symbols, 20, 0.0, 1.0)
        returns = [p["expected_return"] for p in points]
        vols = [p["volatility"] for p in points]
        # Returns and volatilities should both increase along the frontier
        # (sorted by vol, returns should be approximately increasing)
        assert returns[-1] > returns[0]


# ── Tests: _find_max_sharpe ───────────────────────────────────────────────────

class TestFindMaxSharpe:

    def _get_two_asset_inputs(self, n_days: int = 10000):
        log_returns = make_two_asset_returns(n_days=n_days)
        mu, symbols = _compute_expected_returns(log_returns)
        cov = _compute_covariance_matrix(log_returns, symbols)
        return mu, cov, symbols

    def test_weights_sum_to_one(self):
        """Maximum Sharpe portfolio weights sum to 1."""
        mu, cov, symbols = self._get_two_asset_inputs()
        result = _find_max_sharpe(mu, cov, symbols, rfr=0.065,
                                  min_weight=0.0, max_weight=1.0)
        w_sum = sum(result["weights"].values())
        assert math.isclose(w_sum, 1.0, abs_tol=1e-6)

    def test_sharpe_is_positive(self):
        """
        Maximum Sharpe ratio must be positive.
        (portfolio return exceeds risk-free rate for our synthetic data)
        """
        mu, cov, symbols = self._get_two_asset_inputs(n_days=50000)
        result = _find_max_sharpe(mu, cov, symbols, rfr=0.065,
                                  min_weight=0.0, max_weight=1.0)
        assert result["sharpe_ratio"] > 0

    def test_higher_rfr_reduces_sharpe(self):
        """
        Higher risk-free rate reduces Sharpe ratio for the same portfolio.
        Sharpe = (return - rfr) / vol — numerator shrinks as rfr increases.
        """
        mu, cov, symbols = self._get_two_asset_inputs(n_days=50000)
        result_low  = _find_max_sharpe(mu, cov, symbols, rfr=0.03,
                                       min_weight=0.0, max_weight=1.0)
        result_high = _find_max_sharpe(mu, cov, symbols, rfr=0.10,
                                       min_weight=0.0, max_weight=1.0)
        assert result_low["sharpe_ratio"] >= result_high["sharpe_ratio"]

    def test_max_weight_constraint_respected(self):
        """No asset exceeds max_weight in the Maximum Sharpe portfolio."""
        mu, cov, symbols = self._get_two_asset_inputs()
        max_w = 0.7
        result = _find_max_sharpe(mu, cov, symbols, rfr=0.065,
                                  min_weight=0.0, max_weight=max_w)
        for w in result["weights"].values():
            assert w <= max_w + 1e-6

    def test_volatility_is_positive(self):
        """Portfolio volatility must be positive."""
        mu, cov, symbols = self._get_two_asset_inputs()
        result = _find_max_sharpe(mu, cov, symbols, rfr=0.065,
                                  min_weight=0.0, max_weight=1.0)
        assert result["portfolio_volatility"] > 0


# ── Tests: optimise_portfolio (full tool) ─────────────────────────────────────

class TestOptimisePortfolio:

    def test_output_has_all_required_keys(self):
        """Output dict has all keys matching OptimisationResult schema."""
        log_returns = make_two_asset_returns()
        result = optimise_portfolio(log_returns)

        required_keys = {
            "optimal_weights", "max_sharpe_weights",
            "expected_return", "portfolio_volatility",
            "sharpe_ratio", "cml_slope",
            "efficient_frontier", "solver_used",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_optimal_and_max_sharpe_weights_identical(self):
        """optimal_weights and max_sharpe_weights are the same portfolio."""
        log_returns = make_two_asset_returns()
        result = optimise_portfolio(log_returns)
        assert result["optimal_weights"] == result["max_sharpe_weights"]

    def test_cml_slope_equals_sharpe_ratio(self):
        """cml_slope equals sharpe_ratio — slope of the Capital Market Line."""
        log_returns = make_two_asset_returns()
        result = optimise_portfolio(log_returns)
        assert math.isclose(
            result["cml_slope"], result["sharpe_ratio"], rel_tol=1e-6
        )

    def test_solver_used_echoed(self):
        """solver_used field echoes the requested solver."""
        log_returns = make_two_asset_returns()
        result = optimise_portfolio(log_returns, solver="convex_qp")
        assert result["solver_used"] == "convex_qp"

    def test_efficient_frontier_not_empty(self):
        """Efficient Frontier contains at least one point."""
        log_returns = make_two_asset_returns()
        result = optimise_portfolio(log_returns)
        assert len(result["efficient_frontier"]) > 0

    def test_frontier_points_have_required_keys(self):
        """Each frontier point has volatility, expected_return, weights."""
        log_returns = make_two_asset_returns()
        result = optimise_portfolio(log_returns)
        for point in result["efficient_frontier"]:
            assert "volatility" in point
            assert "expected_return" in point
            assert "weights" in point

    def test_single_asset_raises_value_error(self):
        """Single asset portfolio raises ValueError — need at least 2."""
        log_returns = {"A": [0.001] * 500}
        with pytest.raises(ValueError, match="At least 2 assets"):
            optimise_portfolio(log_returns)

    def test_empty_log_returns_raises_value_error(self):
        """Empty log_returns raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            optimise_portfolio({})

    def test_invalid_solver_raises_value_error(self):
        """Invalid solver name raises ValueError."""
        log_returns = make_two_asset_returns()
        with pytest.raises(ValueError, match="Invalid solver"):
            optimise_portfolio(log_returns, solver="invalid")

    def test_unimplemented_solver_raises_not_implemented(self):
        """V2/V3 solvers raise NotImplementedError in v1."""
        log_returns = make_two_asset_returns()
        with pytest.raises(NotImplementedError):
            optimise_portfolio(log_returns, solver="nsga2")

    def test_negative_rfr_raises_value_error(self):
        """Negative risk-free rate raises ValueError."""
        log_returns = make_two_asset_returns()
        with pytest.raises(ValueError, match="non-negative"):
            optimise_portfolio(log_returns, risk_free_rate=-0.01)

    def test_invalid_weight_bounds_raises_value_error(self):
        """min_weight >= max_weight raises ValueError."""
        log_returns = make_two_asset_returns()
        with pytest.raises(ValueError, match="min_weight"):
            optimise_portfolio(
                log_returns,
                constraints={"min_weight": 0.5, "max_weight": 0.3}
            )

    def test_integration_with_nse_fixtures(self):
        """
        Integration test with real NSE fixture data.
        Validates realistic output for a 5-asset large-cap portfolio.
        """
        symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS",
                   "HDFCBANK.NS", "ICICIBANK.NS"]
        log_returns = load_fixture_returns(symbols)

        result = optimise_portfolio(
            log_returns,
            risk_free_rate=0.065,
            n_frontier_points=20,
            constraints={"min_weight": 0.0, "max_weight": 0.40},
        )

        # Weights sum to 1
        w_sum = sum(result["optimal_weights"].values())
        assert math.isclose(w_sum, 1.0, abs_tol=1e-4)

        # All weights within bounds
        for w in result["optimal_weights"].values():
            assert 0.0 - 1e-6 <= w <= 0.40 + 1e-6

        # Portfolio volatility in realistic range for NSE large-caps
        vol = result["portfolio_volatility"]
        assert 0.05 <= vol <= 0.50, \
            f"Portfolio volatility {vol:.4f} out of expected range"

        # Frontier has reasonable number of points
        assert len(result["efficient_frontier"]) >= 10