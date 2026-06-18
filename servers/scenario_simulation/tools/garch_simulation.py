"""
servers/scenario_simulation/tools/garch_simulation.py

Implementation of the run_garch_simulation tool.

Responsibilities:
    - Consume GARCH parameters fitted by the Risk Engine via AgentState
    - Generate N stochastic paths where σ_t evolves at every step via
      the GARCH recursion (not constant as in run_monte_carlo)
    - Preserve cross-asset correlation structure via Cholesky decomposition
    - Compute terminal values, CVaR, VaR, and percentile distribution

Key design decisions:
    No re-fitting:
        GARCH parameters (ω, α, β, ν) and current conditional volatilities
        (σ_T per asset) are passed in as inputs — pre-fitted by the Risk Engine.
        The Simulator does not re-fit. This ensures:
            (1) Parameter consistency: same parameters drive the regime signal
                in the Risk Engine and the paths here
            (2) No redundant MLE computation
            (3) Clean server boundary: Risk Engine estimates, Simulator generates

    Weights source:
        Weights are passed explicitly — orchestrator decides current vs optimal.
        Called twice in FULL analysis type:
            current weights  → garch_sim field in SimulationResult
            optimal weights  → garch_sim_optimal field in SimulationResult

    GARCH recursion per path per step:
        For each simulation path i = 1..N:
            Initialise σ²_T from current_vols (per asset)
            For each future step t = T+1..T+H:
                Draw correlated innovations ε_t via Cholesky
                Compute return: r_t = μ + ε_t
                Update variance: σ²_{t+1} = ω + α·ε²_t + β·σ²_t
            Terminal value = Σ_t w^T r_t

    Correlation structure:
        V1: static Cholesky decomposition of historical covariance matrix
            Uses the same Σ estimated from historical log-returns
        V3 planned: DCC-GARCH time-varying correlation

    Why paths matter more than in run_monte_carlo:
        GARCH introduces serial dependence — σ_t depends on σ_{t-1} and ε_{t-1}
        A large shock at step t raises σ at step t+1 (volatility clustering)
        This temporal structure cannot be collapsed to a terminal value distribution
        without simulating the path — each path's volatility trajectory is unique

Terminal values:
    terminal_value_i = Σ_t (w^T r_t)  — cumulative portfolio return over horizon
    CVaR and VaR computed from distribution of N terminal values

Shared utilities:
    _compute_terminal_values, _compute_cvar_var, _compute_percentiles
    are imported from monte_carlo — identical computation regardless of
    how paths were generated

This file contains pure computation logic only.
No MCP protocol code — that lives in server.py.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# Import shared terminal value and risk metric utilities from monte_carlo
from tools.monte_carlo import (
    _compute_cvar_var,
    _compute_percentiles,
    _compute_terminal_values,
)

# ── Constants ─────────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS      = 50


# ── Parameter validation ──────────────────────────────────────────────────────

def _validate_garch_params(
    garch_params: dict,
    current_vols: dict,
    symbols: list[str],
) -> None:
    """
    Validate GARCH parameters and current volatilities from the Risk Engine.

    Args:
        garch_params: dict[symbol, {omega, alpha, beta, nu}]
        current_vols: dict[symbol, float] — annualised σ_T per asset
        symbols:      list[str] — expected symbols

    Raises:
        ValueError: if any required field is missing or invalid
    """
    for symbol in symbols:
        if symbol not in garch_params:
            raise ValueError(
                f"GARCH parameters missing for symbol '{symbol}'. "
                f"Ensure compute_garch_forecast ran for all symbols."
            )
        if symbol not in current_vols:
            raise ValueError(
                f"Current volatility missing for symbol '{symbol}'. "
                f"Ensure compute_garch_forecast ran for all symbols."
            )

        p = garch_params[symbol]
        required_keys = {"omega", "alpha", "beta"}
        missing = required_keys - set(p.keys())
        if missing:
            raise ValueError(
                f"GARCH params for '{symbol}' missing keys: {missing}"
            )

        if p["alpha"] + p["beta"] >= 1.0:
            raise ValueError(
                f"GARCH model for '{symbol}' is non-stationary: "
                f"α + β = {p['alpha'] + p['beta']:.4f} ≥ 1.0"
            )

        if current_vols[symbol] <= 0:
            raise ValueError(
                f"Current volatility for '{symbol}' must be positive, "
                f"got {current_vols[symbol]}"
            )


# ── Correlation structure ─────────────────────────────────────────────────────

def _compute_cholesky(
    log_returns: dict[str, list[float]],
    symbols: list[str],
) -> np.ndarray:
    """
    Compute Cholesky decomposition of historical return correlation matrix.

    The Cholesky factor L satisfies L L^T = Σ_correlation
    where Σ_correlation is the correlation matrix (not covariance).

    We use correlation rather than covariance because individual asset
    volatilities are handled separately by the GARCH per-asset variance
    term. The Cholesky factor captures only the cross-asset correlation
    structure.

    V1: static correlation matrix from historical log-returns.
    V3 planned: DCC-GARCH time-varying correlation.

    Args:
        log_returns: dict[symbol, list[float]]
        symbols:     list[str] — asset order

    Returns:
        np.ndarray shape (N, N) — lower triangular Cholesky factor

    Raises:
        ValueError: if correlation matrix is not positive definite
    """
    returns_matrix = np.array([log_returns[s] for s in symbols])
    corr_matrix = np.corrcoef(returns_matrix)

    # Add small regularisation to ensure positive definiteness
    # (handles numerical issues with near-singular correlation matrices)
    n = len(symbols)
    corr_matrix += np.eye(n) * 1e-8

    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Correlation matrix is not positive definite. "
            "Assets may be too highly correlated or return series too short."
        )

    return L


# ── GARCH path generation ─────────────────────────────────────────────────────

def _generate_garch_paths(
    log_returns: dict[str, list[float]],
    weights: dict[str, float],
    garch_params: dict,
    current_vols: dict,
    symbols: list[str],
    horizon_days: int,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate N portfolio return paths using GARCH-conditional dynamics.

    For each simulation path i = 1..N:
        Initialise per-asset σ²_T from current_vols (annualised → daily)
        For each future step t = 1..H:
            Draw N(0,1) innovations z_t (shape: N_assets)
            Apply Cholesky to introduce cross-asset correlation:
                ε_t = L × z_t  (correlated standardised innovations)
            Scale by current conditional volatility:
                shock_t = σ_t × ε_t
            Compute asset return using constant mean (V1 assumption):
                r_asset_t = μ + shock_t
            Compute portfolio return:
                r_portfolio_t = Σ w_i × r_asset_i_t
            Update per-asset conditional variance for next step:
                σ²_{t+1} = ω + α × shock²_t + β × σ²_t
                (GARCH recursion — σ evolves based on this step's shock)

    The GARCH recursion means:
        Large shock at step t → elevated σ at step t+1 → possible clustering
        This is the key difference from run_monte_carlo where σ is constant

    Args:
        log_returns:   dict[symbol, list[float]] — for Cholesky computation
        weights:       dict[symbol, float] — portfolio weights
        garch_params:  dict[symbol, {omega, alpha, beta, nu}]
        current_vols:  dict[symbol, float] — annualised σ_T per asset
        symbols:       list[str] — asset order
        horizon_days:  int — simulation horizon H
        n_simulations: int — number of paths N
        rng:           numpy random generator

    Returns:
        np.ndarray shape (N, H) — portfolio daily returns across all paths
    """
    n_assets = len(symbols)
    w = np.array([weights[s] for s in symbols])

    # Cholesky factor for correlation structure
    L = _compute_cholesky(log_returns, symbols)

    # Extract GARCH parameters per asset
    omega = np.array([garch_params[s]["omega"] for s in symbols])
    alpha = np.array([garch_params[s]["alpha"] for s in symbols])
    beta  = np.array([garch_params[s]["beta"]  for s in symbols])

    # Convert annualised current volatility to daily variance
    # σ_annual = σ_daily × √252  →  σ²_daily = (σ_annual / √252)²
    sigma2_current = np.array([
        (current_vols[s] / np.sqrt(TRADING_DAYS_PER_YEAR)) ** 2
        for s in symbols
    ])

    # Output: N×H matrix of portfolio returns
    portfolio_paths = np.zeros((n_simulations, horizon_days))

    for i in range(n_simulations):
        # Each path starts from the same current volatility state σ_T
        sigma2_t = sigma2_current.copy()

        for t in range(horizon_days):
            # Draw independent standard normal innovations
            z = rng.standard_normal(n_assets)

            # Apply Cholesky to introduce cross-asset correlation
            # ε_t = L × z_t — correlated but still zero-mean, unit-variance
            eps_corr = L @ z

            # Scale by current conditional volatility per asset
            # shock_t = σ_t × ε_t
            sigma_t = np.sqrt(np.maximum(sigma2_t, 1e-10))
            shock_t = sigma_t * eps_corr

            # Asset returns (constant mean = 0 in V1)
            r_assets_t = shock_t

            # Portfolio return at this step
            portfolio_paths[i, t] = float(w @ r_assets_t)

            # Update conditional variance for next step via GARCH recursion
            # σ²_{t+1} = ω + α × shock² + β × σ²_t
            sigma2_t = omega + alpha * (shock_t ** 2) + beta * sigma2_t

    return portfolio_paths


# ── Main tool function ────────────────────────────────────────────────────────

def run_garch_simulation(
    log_returns: dict[str, list[float]],
    weights: dict[str, float],
    garch_params: dict,
    current_vols: dict,
    horizon_days: int = 252,
    n_simulations: int = 10000,
    random_seed: int | None = None,
) -> dict:
    """
    Monte Carlo simulation with GARCH-conditional volatility dynamics.

    Uses GARCH parameters pre-fitted by the Risk Engine (compute_garch_forecast).
    Generates N paths where σ_t evolves at every step via the GARCH recursion.

    This is the implementation of the run_garch_simulation MCP tool.
    Called by server.py — never called directly by other servers.

    Key distinction from run_monte_carlo:
        run_monte_carlo:      σ is constant — IID draws from static distribution
        run_garch_simulation: σ_t evolves — serial dependence, volatility clustering
        "The future will evolve from where volatility is RIGHT NOW"

    Args:
        log_returns:   dict[symbol, list[float]] — for Cholesky computation
        weights:       dict[symbol, float] — current OR optimal weights
        garch_params:  dict[symbol, {omega, alpha, beta, nu}]
                       — from GARCHResult.garch_params in AgentState
        current_vols:  dict[symbol, float]
                       — from GARCHResult.current_vols in AgentState
        horizon_days:  int — simulation horizon in trading days
        n_simulations: int — number of paths
        random_seed:   int | None — set for deterministic testing

    Returns:
        dict matching SimulationOutput schema in orchestrator/state.py:
            cvar_95:           float — Expected Shortfall at 95% (primary metric)
            cvar_99:           float — Expected Shortfall at 99%
            var_95:            float — VaR at 95% (context only)
            var_99:            float — VaR at 99% (context only)
            percentiles:       dict  — p10, p25, p50, p75, p90
            n_simulations:     int   — echoed
            distribution_used: str   — always "garch_student_t"
            fitted_nu:         None  — ν was fitted by Risk Engine, not here

    Raises:
        ValueError: if GARCH parameters are missing or invalid
        ValueError: if weights do not sum to 1
    """
    # ── Input validation ──────────────────────────────────────────
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")

    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")

    if not weights:
        raise ValueError("weights dict must not be empty")

    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got {weight_sum:.6f}."
        )

    symbols = list(weights.keys())

    # ── Validate GARCH parameters ─────────────────────────────────
    _validate_garch_params(garch_params, current_vols, symbols)

    # ── Random number generator ───────────────────────────────────
    rng = np.random.default_rng(random_seed)

    # ── Generate GARCH paths ──────────────────────────────────────
    paths = _generate_garch_paths(
        log_returns=log_returns,
        weights=weights,
        garch_params=garch_params,
        current_vols=current_vols,
        symbols=symbols,
        horizon_days=horizon_days,
        n_simulations=n_simulations,
        rng=rng,
    )

    # ── Terminal values ───────────────────────────────────────────
    terminal_values = _compute_terminal_values(paths)

    # ── CVaR and VaR ─────────────────────────────────────────────
    cvar_95, var_95 = _compute_cvar_var(terminal_values, 0.95)
    cvar_99, var_99 = _compute_cvar_var(terminal_values, 0.99)

    # ── Percentile distribution ───────────────────────────────────
    percentiles = _compute_percentiles(terminal_values)

    return {
        "cvar_95":           cvar_95,
        "cvar_99":           cvar_99,
        "var_95":            var_95,
        "var_99":            var_99,
        "percentiles":       percentiles,
        "n_simulations":     n_simulations,
        "distribution_used": "garch_student_t",
        "fitted_nu":         None,
    }