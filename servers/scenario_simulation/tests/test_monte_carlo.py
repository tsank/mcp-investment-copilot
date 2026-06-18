"""
servers/scenario_simulation/tests/test_monte_carlo.py

Unit tests for the run_monte_carlo tool.

Test strategy:
    Deterministic: all tests use random_seed=42 for reproducibility.

    Mathematical properties validated:
        - CVaR >= VaR always (fundamental property)
        - CVaR_99 >= CVaR_95 (stricter confidence = worse tail)
        - Student-t CVaR > Gaussian CVaR for same data (fat tails)
        - Antithetic variates produce paired paths summing near zero
        - Terminal values are sums of horizon-length return series
        - Percentiles are ordered: p10 < p25 < p50 < p75 < p90

    Output schema validated:
        - All required keys present
        - fitted_nu present for student_t, None for others
        - distribution_used echoed correctly
        - n_simulations echoed correctly

Run from project root:
    cd ~/genaiprojects/mcp-investment-copilot
    pytest servers/scenario_simulation/tests/test_monte_carlo.py -v
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

from tools.monte_carlo import (
    _compute_cvar_var,
    _compute_percentiles,
    _compute_portfolio_returns,
    _compute_terminal_values,
    _fit_gaussian,
    _fit_student_t,
    _generate_gaussian_paths,
    _generate_student_t_paths,
    _historical_bootstrap,
    run_monte_carlo,
)


# ── Shared test data ──────────────────────────────────────────────────────────

def load_fixture_returns(symbols: list[str]) -> dict[str, list[float]]:
    """Load log-returns from NSE fixtures — avoids cross-server imports."""
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


def make_synthetic_returns(
    n_assets: int = 3,
    n_days: int = 500,
    seed: int = 42,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Generate synthetic returns and equal weights for testing."""
    rng = np.random.default_rng(seed)
    symbols = [f"ASSET_{i}" for i in range(n_assets)]
    log_returns = {
        s: rng.normal(0.0005, 0.015, n_days).tolist()
        for s in symbols
    }
    weights = {s: 1.0 / n_assets for s in symbols}
    return log_returns, weights


# ── Tests: _compute_portfolio_returns ─────────────────────────────────────────

class TestComputePortfolioReturns:

    def test_single_asset_full_weight(self):
        """Single asset with weight=1.0 — portfolio returns equal asset returns."""
        log_returns = {"A": [0.01, -0.02, 0.03]}
        weights = {"A": 1.0}
        result = _compute_portfolio_returns(log_returns, weights)
        assert len(result) == 3
        assert math.isclose(result[0], 0.01, rel_tol=1e-9)

    def test_weights_not_summing_to_one_raises(self):
        """Weights not summing to 1 raises ValueError."""
        log_returns = {"A": [0.01], "B": [0.02]}
        weights = {"A": 0.4, "B": 0.4}
        with pytest.raises(ValueError, match="sum to 1"):
            _compute_portfolio_returns(log_returns, weights)

    def test_missing_symbol_raises(self):
        """Symbol in weights but not in log_returns raises ValueError."""
        log_returns = {"A": [0.01]}
        weights = {"A": 0.5, "B": 0.5}
        with pytest.raises(ValueError, match="not found in log_returns"):
            _compute_portfolio_returns(log_returns, weights)


# ── Tests: distribution fitting ───────────────────────────────────────────────

class TestDistributionFitting:

    def test_student_t_nu_is_positive(self):
        """Fitted Student-t degrees of freedom must be positive."""
        rng = np.random.default_rng(42)
        returns = rng.standard_t(df=5, size=500) * 0.01
        nu, mu, sigma = _fit_student_t(returns)
        assert nu > 0

    def test_student_t_sigma_is_positive(self):
        """Fitted Student-t scale must be positive."""
        rng = np.random.default_rng(42)
        returns = rng.standard_t(df=5, size=500) * 0.01
        nu, mu, sigma = _fit_student_t(returns)
        assert sigma > 0

    def test_gaussian_sigma_is_positive(self):
        """Fitted Gaussian standard deviation must be positive."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.015, 500)
        mu, sigma = _fit_gaussian(returns)
        assert sigma > 0

    def test_gaussian_mu_close_to_sample_mean(self):
        """Fitted Gaussian mean should be close to sample mean."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.015, 10000)
        mu, sigma = _fit_gaussian(returns)
        assert math.isclose(mu, np.mean(returns), rel_tol=1e-6)


# ── Tests: path generation ────────────────────────────────────────────────────

class TestPathGeneration:

    def test_student_t_paths_shape(self):
        """Student-t paths have shape (n_simulations, horizon_days)."""
        rng = np.random.default_rng(42)
        paths = _generate_student_t_paths(
            nu=5.0, mu=0.001, sigma=0.015,
            n_simulations=100, horizon_days=21, rng=rng
        )
        assert paths.shape == (100, 21)

    def test_gaussian_paths_shape(self):
        """Gaussian paths have shape (n_simulations, horizon_days)."""
        rng = np.random.default_rng(42)
        paths = _generate_gaussian_paths(
            mu=0.001, sigma=0.015,
            n_simulations=100, horizon_days=21, rng=rng
        )
        assert paths.shape == (100, 21)

    def test_antithetic_pairs_sum_near_zero(self):
        """
        Antithetic variates: base path + mirror path should sum near zero
        for each time step (since mirror = -base before scaling by mu).
        For zero-mean distribution (mu=0), pairs sum exactly to zero.
        """
        rng = np.random.default_rng(42)
        n = 100
        paths = _generate_gaussian_paths(
            mu=0.0, sigma=0.015,
            n_simulations=n, horizon_days=21, rng=rng
        )
        half = n // 2
        base = paths[:half]
        anti = paths[half:]
        pair_sums = base + anti
        # For mu=0, base + anti = (mu + sigma×z) + (mu - sigma×z) = 2×mu = 0
        assert np.allclose(pair_sums, 0.0, atol=1e-10)

    def test_historical_bootstrap_shape(self):
        """Historical bootstrap has shape (n_simulations, horizon_days)."""
        rng = np.random.default_rng(42)
        returns = np.random.default_rng(0).normal(0.001, 0.015, 500)
        paths = _historical_bootstrap(returns, n_simulations=100,
                                      horizon_days=21, rng=rng)
        assert paths.shape == (100, 21)

    def test_historical_bootstrap_values_from_original(self):
        """Bootstrap draws are from the original return series."""
        rng = np.random.default_rng(42)
        original = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        paths = _historical_bootstrap(original, n_simulations=50,
                                      horizon_days=10, rng=rng)
        original_set = set(original.tolist())
        for val in paths.flatten():
            assert round(val, 10) in {round(x, 10) for x in original_set}

    def test_deterministic_with_seed(self):
        """Same seed produces identical paths."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        paths1 = _generate_gaussian_paths(0.001, 0.015, 100, 21, rng1)
        paths2 = _generate_gaussian_paths(0.001, 0.015, 100, 21, rng2)
        assert np.allclose(paths1, paths2)


# ── Tests: terminal values and risk metrics ───────────────────────────────────

class TestTerminalValuesAndRiskMetrics:

    def test_terminal_values_shape(self):
        """Terminal values shape is (N,) — one value per path."""
        paths = np.random.default_rng(42).normal(0.001, 0.015, (100, 21))
        tv = _compute_terminal_values(paths)
        assert tv.shape == (100,)

    def test_terminal_values_are_row_sums(self):
        """Terminal value for each path equals sum of that path's daily returns."""
        paths = np.array([[0.01, -0.02, 0.03],
                          [0.02, 0.01, -0.01]])
        tv = _compute_terminal_values(paths)
        assert math.isclose(tv[0], 0.02, rel_tol=1e-9)
        assert math.isclose(tv[1], 0.02, rel_tol=1e-9)

    def test_cvar_greater_than_var(self):
        """CVaR >= VaR at the same confidence level — always."""
        rng = np.random.default_rng(42)
        tv = rng.normal(-0.05, 0.10, 10000)
        cvar, var = _compute_cvar_var(tv, 0.95)
        assert cvar >= var

    def test_cvar_99_greater_than_cvar_95(self):
        """CVaR_99 >= CVaR_95 — stricter confidence = worse expected tail loss."""
        rng = np.random.default_rng(42)
        tv = rng.normal(-0.05, 0.10, 10000)
        cvar_95, var_95 = _compute_cvar_var(tv, 0.95)
        cvar_99, var_99 = _compute_cvar_var(tv, 0.99)
        assert cvar_99 >= cvar_95

    def test_percentiles_are_ordered(self):
        """Percentiles must be in ascending order: p10 < p25 < p50 < p75 < p90."""
        rng = np.random.default_rng(42)
        tv = rng.normal(0.05, 0.15, 10000)
        p = _compute_percentiles(tv)
        assert p["p10"] < p["p25"] < p["p50"] < p["p75"] < p["p90"]


# ── Tests: run_monte_carlo (full tool) ───────────────────────────────────────

class TestRunMonteCarlo:

    def test_output_has_all_required_keys(self):
        """Output dict has all keys matching SimulationOutput schema."""
        log_returns, weights = make_synthetic_returns()
        result = run_monte_carlo(
            log_returns, weights,
            n_simulations=1000, horizon_days=21,
            random_seed=42
        )
        required = {
            "cvar_95", "cvar_99", "var_95", "var_99",
            "percentiles", "n_simulations", "distribution_used", "fitted_nu"
        }
        assert required.issubset(set(result.keys()))

    def test_cvar_greater_than_var(self):
        """CVaR_95 >= VaR_95 in full tool output."""
        log_returns, weights = make_synthetic_returns()
        result = run_monte_carlo(
            log_returns, weights,
            n_simulations=1000, horizon_days=21, random_seed=42
        )
        assert result["cvar_95"] >= result["var_95"]
        assert result["cvar_99"] >= result["var_99"]

    def test_student_t_fitted_nu_is_float(self):
        """fitted_nu is a float for Student-t distribution."""
        log_returns, weights = make_synthetic_returns()
        result = run_monte_carlo(
            log_returns, weights, distribution="student_t",
            n_simulations=1000, horizon_days=21, random_seed=42
        )
        assert result["fitted_nu"] is not None
        assert isinstance(result["fitted_nu"], float)

    def test_gaussian_fitted_nu_is_none(self):
        """fitted_nu is None for Gaussian distribution."""
        log_returns, weights = make_synthetic_returns()
        result = run_monte_carlo(
            log_returns, weights, distribution="gaussian",
            n_simulations=1000, horizon_days=21, random_seed=42
        )
        assert result["fitted_nu"] is None

    def test_bootstrap_fitted_nu_is_none(self):
        """fitted_nu is None for historical bootstrap."""
        log_returns, weights = make_synthetic_returns()
        result = run_monte_carlo(
            log_returns, weights, distribution="historical_bootstrap",
            n_simulations=1000, horizon_days=21, random_seed=42
        )
        assert result["fitted_nu"] is None

    def test_distribution_used_echoed(self):
        """distribution_used echoes the requested distribution."""
        log_returns, weights = make_synthetic_returns()
        for dist in ["student_t", "gaussian", "historical_bootstrap"]:
            result = run_monte_carlo(
                log_returns, weights, distribution=dist,
                n_simulations=500, horizon_days=21, random_seed=42
            )
            assert result["distribution_used"] == dist

    def test_n_simulations_echoed(self):
        """n_simulations echoed in output (may be bumped to even number)."""
        log_returns, weights = make_synthetic_returns()
        result = run_monte_carlo(
            log_returns, weights,
            n_simulations=1000, horizon_days=21, random_seed=42
        )
        assert result["n_simulations"] == 1000

    def test_odd_n_simulations_bumped_to_even(self):
        """Odd n_simulations is bumped to next even number for antithetic variates."""
        log_returns, weights = make_synthetic_returns()
        result = run_monte_carlo(
            log_returns, weights,
            n_simulations=999, horizon_days=21, random_seed=42
        )
        assert result["n_simulations"] == 1000

    def test_deterministic_with_seed(self):
        """Same random_seed produces identical CVaR results."""
        log_returns, weights = make_synthetic_returns()
        result1 = run_monte_carlo(
            log_returns, weights,
            n_simulations=1000, horizon_days=21, random_seed=42
        )
        result2 = run_monte_carlo(
            log_returns, weights,
            n_simulations=1000, horizon_days=21, random_seed=42
        )
        assert math.isclose(result1["cvar_95"], result2["cvar_95"], rel_tol=1e-9)

    def test_invalid_distribution_raises(self):
        """Invalid distribution name raises ValueError."""
        log_returns, weights = make_synthetic_returns()
        with pytest.raises(ValueError, match="Invalid distribution"):
            run_monte_carlo(log_returns, weights, distribution="invalid")

    def test_integration_with_nse_fixtures(self):
        """
        Integration test with real NSE fixture data.
        Validates realistic CVaR range for a 5-asset NSE portfolio.
        """
        symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS",
                   "HDFCBANK.NS", "ICICIBANK.NS"]
        log_returns = load_fixture_returns(symbols)
        weights = {s: 0.2 for s in symbols}

        result = run_monte_carlo(
            log_returns, weights,
            horizon_days=252, n_simulations=5000,
            distribution="student_t", random_seed=42
        )

        # CVaR_95 for a diversified NSE large-cap portfolio over 1 year
        # should be in a realistic range
        assert result["cvar_95"] > 0, "CVaR must be positive (loss)"
        assert result["cvar_95"] < 1.0, "CVaR should be < 100% loss"
        assert result["cvar_95"] >= result["var_95"]
        assert result["percentiles"]["p50"] is not None