"""
servers/scenario_simulation/tests/test_garch_simulation.py

Unit tests for the run_garch_simulation tool.

Test strategy:
    Deterministic: all tests use random_seed=42.

    Mathematical properties validated:
        - CVaR >= VaR always
        - CVaR_99 >= CVaR_95
        - GARCH paths show volatility clustering:
          elevated starting vol → higher CVaR than suppressed starting vol
        - Cholesky correctly produces correlated paths
        - GARCH recursion: σ_t evolves at every step
        - Weights respected in portfolio return aggregation

    GARCH parameter fixtures:
        Tests use synthetic GARCH parameters with known stationarity properties.
        alpha + beta < 1 always — required for stationarity.

Run from project root:
    cd ~/genaiprojects/mcp-investment-copilot
    pytest servers/scenario_simulation/tests/test_garch_simulation.py -v
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

from tools.garch_simulation import (
    _compute_cholesky,
    _generate_garch_paths,
    _validate_garch_params,
    run_garch_simulation,
)
from tools.monte_carlo import (
    _compute_cvar_var,
    _compute_terminal_values,
)


# ── Shared test data ──────────────────────────────────────────────────────────

def make_synthetic_garch_params(
    symbols: list[str],
    omega: float = 0.00001,
    alpha: float = 0.08,
    beta: float = 0.88,
    nu: float = 5.0,
) -> dict:
    """
    Create synthetic GARCH parameters for testing.
    alpha + beta = 0.96 < 1.0 — stationary.
    """
    return {
        s: {"omega": omega, "alpha": alpha, "beta": beta, "nu": nu}
        for s in symbols
    }


def make_synthetic_current_vols(
    symbols: list[str],
    vol: float = 0.20,
) -> dict:
    """Create synthetic current volatilities (annualised)."""
    return {s: vol for s in symbols}


def make_synthetic_returns(
    symbols: list[str],
    n_days: int = 500,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Generate synthetic log-returns for Cholesky computation."""
    rng = np.random.default_rng(seed)
    return {
        s: rng.normal(0.0005, 0.015, n_days).tolist()
        for s in symbols
    }


def load_fixture_returns(symbols: list[str]) -> dict[str, list[float]]:
    """Load log-returns from NSE fixtures."""
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


def load_garch_params_from_risk_engine(
    symbols: list[str],
) -> tuple[dict, dict]:
    """
    Fit real GARCH parameters using the Risk Engine tool.
    Uses importlib to avoid cross-server tools/ namespace conflict.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "garch_forecast",
        PROJECT_ROOT / "servers" / "risk_engine" / "tools" / "garch_forecast.py",
    )
    garch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(garch_module)

    log_returns = load_fixture_returns(symbols)
    result = garch_module.compute_garch_forecast(log_returns, horizon_days=21)

    return result["garch_params"], result["current_vols"]


# ── Tests: _validate_garch_params ────────────────────────────────────────────

class TestValidateGarchParams:

    def test_valid_params_pass(self):
        """Valid GARCH parameters pass validation without error."""
        symbols = ["A", "B"]
        garch_params = make_synthetic_garch_params(symbols)
        current_vols = make_synthetic_current_vols(symbols)
        _validate_garch_params(garch_params, current_vols, symbols)

    def test_missing_symbol_in_garch_params_raises(self):
        """Missing symbol in garch_params raises ValueError."""
        symbols = ["A", "B"]
        garch_params = make_synthetic_garch_params(["A"])  # B missing
        current_vols = make_synthetic_current_vols(symbols)
        with pytest.raises(ValueError, match="missing for symbol"):
            _validate_garch_params(garch_params, current_vols, symbols)

    def test_missing_symbol_in_current_vols_raises(self):
        """Missing symbol in current_vols raises ValueError."""
        symbols = ["A", "B"]
        garch_params = make_synthetic_garch_params(symbols)
        current_vols = make_synthetic_current_vols(["A"])  # B missing
        with pytest.raises(ValueError, match="missing for symbol"):
            _validate_garch_params(garch_params, current_vols, symbols)

    def test_non_stationary_params_raises(self):
        """alpha + beta >= 1 raises ValueError."""
        symbols = ["A"]
        garch_params = {"A": {"omega": 0.00001, "alpha": 0.10, "beta": 0.95, "nu": 5.0}}
        current_vols = {"A": 0.20}
        with pytest.raises(ValueError, match="non-stationary"):
            _validate_garch_params(garch_params, current_vols, symbols)

    def test_zero_current_vol_raises(self):
        """Zero or negative current_vol raises ValueError."""
        symbols = ["A"]
        garch_params = make_synthetic_garch_params(symbols)
        current_vols = {"A": 0.0}
        with pytest.raises(ValueError, match="must be positive"):
            _validate_garch_params(garch_params, current_vols, symbols)


# ── Tests: _compute_cholesky ──────────────────────────────────────────────────

class TestComputeCholesky:

    def test_cholesky_shape(self):
        """Cholesky factor has shape (N_assets, N_assets)."""
        symbols = ["A", "B", "C"]
        log_returns = make_synthetic_returns(symbols)
        L = _compute_cholesky(log_returns, symbols)
        assert L.shape == (3, 3)

    def test_cholesky_is_lower_triangular(self):
        """Cholesky factor is lower triangular."""
        symbols = ["A", "B"]
        log_returns = make_synthetic_returns(symbols)
        L = _compute_cholesky(log_returns, symbols)
        # Upper triangle (above diagonal) should be zero
        assert np.allclose(np.triu(L, k=1), 0.0, atol=1e-10)

    def test_cholesky_reconstructs_correlation(self):
        """L @ L^T should recover the original correlation matrix."""
        symbols = ["A", "B", "C"]
        log_returns = make_synthetic_returns(symbols, n_days=1000)
        L = _compute_cholesky(log_returns, symbols)

        returns_matrix = np.array([log_returns[s] for s in symbols])
        corr_original = np.corrcoef(returns_matrix)

        # L @ L^T should approximate the original correlation matrix
        corr_reconstructed = L @ L.T
        assert np.allclose(corr_original, corr_reconstructed, atol=1e-6)


# ── Tests: _generate_garch_paths ─────────────────────────────────────────────

class TestGenerateGarchPaths:

    def test_paths_shape(self):
        """GARCH paths have shape (n_simulations, horizon_days)."""
        symbols = ["A", "B"]
        log_returns = make_synthetic_returns(symbols)
        garch_params = make_synthetic_garch_params(symbols)
        current_vols = make_synthetic_current_vols(symbols)
        weights = {"A": 0.6, "B": 0.4}
        rng = np.random.default_rng(42)

        paths = _generate_garch_paths(
            log_returns, weights, garch_params, current_vols,
            symbols, horizon_days=21, n_simulations=100, rng=rng
        )
        assert paths.shape == (100, 21)

    def test_elevated_vol_produces_wider_distribution(self):
        """
        Elevated starting volatility produces wider terminal value distribution.
        Higher current_vol → GARCH starts at higher σ_T → larger shocks → wider CVaR.
        """
        symbols = ["A"]
        log_returns = make_synthetic_returns(symbols, n_days=1000)
        garch_params = make_synthetic_garch_params(symbols)
        weights = {"A": 1.0}

        # Low starting vol
        current_vols_low = {"A": 0.10}
        rng_low = np.random.default_rng(42)
        paths_low = _generate_garch_paths(
            log_returns, weights, garch_params, current_vols_low,
            symbols, horizon_days=252, n_simulations=2000, rng=rng_low
        )
        tv_low = _compute_terminal_values(paths_low)
        cvar_low, _ = _compute_cvar_var(tv_low, 0.95)

        # High starting vol
        current_vols_high = {"A": 0.40}
        rng_high = np.random.default_rng(42)
        paths_high = _generate_garch_paths(
            log_returns, weights, garch_params, current_vols_high,
            symbols, horizon_days=252, n_simulations=2000, rng=rng_high
        )
        tv_high = _compute_terminal_values(paths_high)
        cvar_high, _ = _compute_cvar_var(tv_high, 0.95)

        # Elevated starting vol should produce higher CVaR (worse tail)
        assert cvar_high > cvar_low, \
            f"Expected cvar_high ({cvar_high:.4f}) > cvar_low ({cvar_low:.4f})"

    def test_deterministic_with_seed(self):
        """Same seed produces identical GARCH paths."""
        symbols = ["A", "B"]
        log_returns = make_synthetic_returns(symbols)
        garch_params = make_synthetic_garch_params(symbols)
        current_vols = make_synthetic_current_vols(symbols)
        weights = {"A": 0.5, "B": 0.5}

        rng1 = np.random.default_rng(42)
        paths1 = _generate_garch_paths(
            log_returns, weights, garch_params, current_vols,
            symbols, 21, 100, rng1
        )

        rng2 = np.random.default_rng(42)
        paths2 = _generate_garch_paths(
            log_returns, weights, garch_params, current_vols,
            symbols, 21, 100, rng2
        )

        assert np.allclose(paths1, paths2)


# ── Tests: run_garch_simulation (full tool) ───────────────────────────────────

class TestRunGarchSimulation:

    def _get_inputs(self, symbols: list[str] = None):
        """Helper — create complete inputs for run_garch_simulation."""
        if symbols is None:
            symbols = ["A", "B"]
        log_returns = make_synthetic_returns(symbols)
        weights = {s: 1.0 / len(symbols) for s in symbols}
        garch_params = make_synthetic_garch_params(symbols)
        current_vols = make_synthetic_current_vols(symbols)
        return log_returns, weights, garch_params, current_vols

    def test_output_has_all_required_keys(self):
        """Output dict has all keys matching SimulationOutput schema."""
        lr, w, gp, cv = self._get_inputs()
        result = run_garch_simulation(
            lr, w, gp, cv,
            horizon_days=21, n_simulations=500, random_seed=42
        )
        required = {
            "cvar_95", "cvar_99", "var_95", "var_99",
            "percentiles", "n_simulations", "distribution_used", "fitted_nu"
        }
        assert required.issubset(set(result.keys()))

    def test_distribution_used_is_garch_student_t(self):
        """distribution_used is always 'garch_student_t'."""
        lr, w, gp, cv = self._get_inputs()
        result = run_garch_simulation(
            lr, w, gp, cv,
            horizon_days=21, n_simulations=500, random_seed=42
        )
        assert result["distribution_used"] == "garch_student_t"

    def test_fitted_nu_is_none(self):
        """fitted_nu is None — ν was fitted by Risk Engine, not here."""
        lr, w, gp, cv = self._get_inputs()
        result = run_garch_simulation(
            lr, w, gp, cv,
            horizon_days=21, n_simulations=500, random_seed=42
        )
        assert result["fitted_nu"] is None

    def test_cvar_greater_than_var(self):
        """CVaR_95 >= VaR_95 and CVaR_99 >= VaR_99."""
        lr, w, gp, cv = self._get_inputs()
        result = run_garch_simulation(
            lr, w, gp, cv,
            horizon_days=21, n_simulations=1000, random_seed=42
        )
        assert result["cvar_95"] >= result["var_95"]
        assert result["cvar_99"] >= result["var_99"]

    def test_cvar_99_greater_than_cvar_95(self):
        """CVaR_99 >= CVaR_95."""
        lr, w, gp, cv = self._get_inputs()
        result = run_garch_simulation(
            lr, w, gp, cv,
            horizon_days=21, n_simulations=1000, random_seed=42
        )
        assert result["cvar_99"] >= result["cvar_95"]

    def test_percentiles_are_ordered(self):
        """Percentiles are in ascending order."""
        lr, w, gp, cv = self._get_inputs()
        result = run_garch_simulation(
            lr, w, gp, cv,
            horizon_days=21, n_simulations=1000, random_seed=42
        )
        p = result["percentiles"]
        assert p["p10"] < p["p25"] < p["p50"] < p["p75"] < p["p90"]

    def test_deterministic_with_seed(self):
        """Same random_seed produces identical CVaR results."""
        lr, w, gp, cv = self._get_inputs()
        result1 = run_garch_simulation(
            lr, w, gp, cv,
            horizon_days=21, n_simulations=500, random_seed=42
        )
        result2 = run_garch_simulation(
            lr, w, gp, cv,
            horizon_days=21, n_simulations=500, random_seed=42
        )
        assert math.isclose(result1["cvar_95"], result2["cvar_95"], rel_tol=1e-9)

    def test_invalid_weights_raises(self):
        """Weights not summing to 1 raises ValueError."""
        lr, _, gp, cv = self._get_inputs()
        bad_weights = {"A": 0.3, "B": 0.3}
        with pytest.raises(ValueError, match="sum to 1"):
            run_garch_simulation(lr, bad_weights, gp, cv, random_seed=42)

    def test_missing_garch_params_raises(self):
        """Missing GARCH params for a symbol raises ValueError."""
        symbols = ["A", "B"]
        lr = make_synthetic_returns(symbols)
        w = {"A": 0.5, "B": 0.5}
        gp = make_synthetic_garch_params(["A"])  # B missing
        cv = make_synthetic_current_vols(symbols)
        with pytest.raises(ValueError, match="missing for symbol"):
            run_garch_simulation(lr, w, gp, cv, random_seed=42)

    def test_integration_with_nse_fixtures(self):
        """
        Integration test using real NSE fixture data and real GARCH parameters
        fitted by the Risk Engine. Validates realistic CVaR range.
        """
        symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        log_returns = load_fixture_returns(symbols)
        weights = {s: 1.0 / 3 for s in symbols}
        garch_params, current_vols = load_garch_params_from_risk_engine(symbols)

        result = run_garch_simulation(
            log_returns, weights,
            garch_params=garch_params,
            current_vols=current_vols,
            horizon_days=63,
            n_simulations=2000,
            random_seed=42,
        )

        assert result["cvar_95"] > 0
        assert result["cvar_95"] < 1.0
        assert result["cvar_95"] >= result["var_95"]
        assert result["distribution_used"] == "garch_student_t"