"""
servers/scenario_simulation/tools/monte_carlo.py

Implementation of the run_monte_carlo tool.

Responsibilities:
    - Compute portfolio returns from asset log-returns and weights
    - Fit a static distribution to the portfolio return series
    - Generate N simulated paths of length H days using antithetic variates
    - Compute terminal values (cumulative return at end of horizon)
    - Compute CVaR, VaR, and percentile distribution of terminal values

Design decisions:
    Weights source:
        Weights are passed explicitly as input — not read from AgentState.
        The orchestrator decides which weights to pass:
            current weights  → Portfolio.holdings (user's actual position)
            optimal weights  → optimisation_result.optimal_weights
        This tool is called twice in FULL analysis type:
            once with current weights → monte_carlo field
            once with optimal weights → monte_carlo_optimal field

    Static distribution assumption:
        All N×H draws come from the same fitted distribution.
        σ is constant across all simulation steps.
        This is the IID assumption — no temporal structure in volatility.
        For dynamic volatility, use run_garch_simulation.

    Antithetic variates:
        For each path i, we generate a mirror path with negated draws.
        This halves the variance of the CVaR estimator at no model cost.
        Effective n_simulations = n_simulations (N/2 pairs, N total paths).

    Terminal values:
        terminal_value_i = Σ_t r_portfolio_i_t  (sum over horizon)
        = total cumulative log-return over the horizon
        CVaR and VaR computed from the distribution of terminal values.
        Not path-by-path statistics — we care about the outcome at the
        end of the horizon, not individual daily moves.

    CVaR is the primary metric:
        CVaR gates the Compliance server.
        VaR is reported for context and interpretability only.

This file contains pure computation logic only.
No MCP protocol code — that lives in server.py.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats

# ── Constants ─────────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
VALID_DISTRIBUTIONS = {"student_t", "gaussian", "historical_bootstrap"}
MIN_OBSERVATIONS = 50  # minimum return series length for distribution fitting


# ── Portfolio return computation ──────────────────────────────────────────────

def _compute_portfolio_returns(
    log_returns: dict[str, list[float]],
    weights: dict[str, float],
) -> np.ndarray:
    """
    Compute daily portfolio log-returns as weighted sum of asset log-returns.

    r_portfolio_t ≈ Σ w_i × r_i_t

    Args:
        log_returns: dict[symbol, list[float]]
        weights:     dict[symbol, float] — must sum to 1

    Returns:
        np.ndarray shape (T,) — daily portfolio returns

    Raises:
        ValueError: if weights do not sum to 1
        ValueError: if symbols in weights not in log_returns
    """
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got {weight_sum:.6f}."
        )

    missing = set(weights.keys()) - set(log_returns.keys())
    if missing:
        raise ValueError(
            f"Symbols in weights not found in log_returns: {missing}"
        )

    symbols = list(weights.keys())
    n = len(log_returns[symbols[0]])
    portfolio_returns = np.zeros(n)

    for symbol, weight in weights.items():
        portfolio_returns += weight * np.array(log_returns[symbol])

    return portfolio_returns


# ── Distribution fitting ──────────────────────────────────────────────────────

def _fit_student_t(returns: np.ndarray) -> tuple[float, float, float]:
    """
    Fit a Student-t distribution to the portfolio return series via MLE.

    scipy.stats.t.fit returns (df, loc, scale) where:
        df   = degrees of freedom ν — controls tail thickness
        loc  = location parameter ≈ mean
        scale = scale parameter ≈ std × correction factor

    Args:
        returns: np.ndarray — daily portfolio returns

    Returns:
        tuple: (nu, mu, sigma)
            nu:    float — degrees of freedom
            mu:    float — location (daily mean)
            sigma: float — scale (daily volatility)
    """
    nu, mu, sigma = stats.t.fit(returns)
    return float(nu), float(mu), float(sigma)


def _fit_gaussian(returns: np.ndarray) -> tuple[float, float]:
    """
    Fit a Gaussian distribution to the portfolio return series.

    Args:
        returns: np.ndarray — daily portfolio returns

    Returns:
        tuple: (mu, sigma)
            mu:    float — daily mean
            sigma: float — daily standard deviation (ddof=1)
    """
    mu    = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    return mu, sigma


# ── Path generation ───────────────────────────────────────────────────────────

def _generate_student_t_paths(
    nu: float,
    mu: float,
    sigma: float,
    n_simulations: int,
    horizon_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate N×H matrix of IID Student-t draws with antithetic variates.

    Antithetic variates: for each of the N/2 base paths, we generate a
    mirror path with negated standardised draws. This reduces the variance
    of the CVaR estimator significantly at no additional model cost.

    The final matrix has n_simulations rows (N/2 base + N/2 antithetic).

    Args:
        nu:            float — Student-t degrees of freedom
        mu:            float — location (daily mean)
        sigma:         float — scale (daily volatility)
        n_simulations: int — total number of paths (must be even)
        horizon_days:  int — simulation horizon
        rng:           numpy random generator (seeded or unseeded)

    Returns:
        np.ndarray shape (n_simulations, horizon_days) — daily returns
    """
    half_n = n_simulations // 2

    # Draw standardised Student-t variates for base paths
    z_base = rng.standard_t(df=nu, size=(half_n, horizon_days))

    # Antithetic mirror paths — negate the standardised draws
    z_anti = -z_base

    # Stack base and antithetic paths
    z_all = np.vstack([z_base, z_anti])

    # Scale from standardised to actual returns: r = mu + sigma × z
    return mu + sigma * z_all


def _generate_gaussian_paths(
    mu: float,
    sigma: float,
    n_simulations: int,
    horizon_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate N×H matrix of IID Gaussian draws with antithetic variates.

    Args:
        mu:            float — daily mean
        sigma:         float — daily standard deviation
        n_simulations: int — total number of paths (must be even)
        horizon_days:  int — simulation horizon
        rng:           numpy random generator

    Returns:
        np.ndarray shape (n_simulations, horizon_days) — daily returns
    """
    half_n = n_simulations // 2

    z_base = rng.standard_normal(size=(half_n, horizon_days))
    z_anti = -z_base
    z_all  = np.vstack([z_base, z_anti])

    return mu + sigma * z_all


def _historical_bootstrap(
    portfolio_returns: np.ndarray,
    n_simulations: int,
    horizon_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate N×H matrix by resampling historical returns with replacement.

    No distribution is fitted — draws directly from the empirical distribution.
    Naturally preserves the fat-tail and skew properties of the actual data.
    Cannot generate scenarios worse than the historical worst return.

    Args:
        portfolio_returns: np.ndarray — historical portfolio returns
        n_simulations:     int — total number of paths
        horizon_days:      int — simulation horizon
        rng:               numpy random generator

    Returns:
        np.ndarray shape (n_simulations, horizon_days) — resampled returns
    """
    T = len(portfolio_returns)
    indices = rng.integers(0, T, size=(n_simulations, horizon_days))
    return portfolio_returns[indices]


# ── Terminal values and risk metrics ──────────────────────────────────────────

def _compute_terminal_values(paths: np.ndarray) -> np.ndarray:
    """
    Compute terminal portfolio value for each simulation path.

    terminal_value_i = Σ_t r_portfolio_i_t  (sum over horizon days)
    = cumulative log-return over the full horizon

    Row-wise sum of the N×H path matrix → N terminal values.

    Args:
        paths: np.ndarray shape (N, H) — daily portfolio returns

    Returns:
        np.ndarray shape (N,) — terminal cumulative returns
    """
    return np.sum(paths, axis=1)


def _compute_cvar_var(
    terminal_values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Compute CVaR (Expected Shortfall) and VaR from terminal values.

    VaR:  -percentile(terminal_values, 1-confidence_level)
          The loss threshold exceeded with probability 1-α.
          Expressed as positive loss number.

    CVaR: -mean(terminal_values[terminal_values < -VaR])
          Mean of all losses exceeding VaR threshold.
          Expressed as positive loss number.
          CVaR >= VaR always.

    Sign convention:
        terminal_values are returns — positive = gain, negative = loss
        VaR and CVaR are expressed as positive loss numbers

    Args:
        terminal_values:  np.ndarray shape (N,) — cumulative returns
        confidence_level: float — e.g. 0.95 for 95% confidence

    Returns:
        tuple: (cvar, var) — both positive loss numbers

    Raises:
        ValueError: if no terminal values fall below the VaR threshold
    """
    percentile = (1.0 - confidence_level) * 100
    var = float(-np.percentile(terminal_values, percentile))

    tail = terminal_values[terminal_values < -var]
    if len(tail) == 0:
        raise ValueError(
            f"No terminal values below VaR threshold at {confidence_level:.0%} "
            f"confidence. Try more simulations or a shorter horizon."
        )

    cvar = float(-np.mean(tail))
    return cvar, var


def _compute_percentiles(terminal_values: np.ndarray) -> dict:
    """
    Compute percentile distribution of terminal values.

    Args:
        terminal_values: np.ndarray shape (N,)

    Returns:
        dict with keys p10, p25, p50, p75, p90
    """
    p10, p25, p50, p75, p90 = np.percentile(
        terminal_values, [10, 25, 50, 75, 90]
    )
    return {
        "p10": float(p10),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
    }


# ── Main tool function ────────────────────────────────────────────────────────

def run_monte_carlo(
    log_returns: dict[str, list[float]],
    weights: dict[str, float],
    horizon_days: int = 252,
    n_simulations: int = 10000,
    confidence_levels: list[float] | None = None,
    distribution: Literal["student_t", "gaussian", "historical_bootstrap"] = "student_t",
    random_seed: int | None = None,
) -> dict:
    """
    Monte Carlo simulation with static IID distribution.

    Generates N simulated portfolio return paths, computes terminal values,
    and returns CVaR, VaR, and percentile distribution.

    This is the implementation of the run_monte_carlo MCP tool.
    Called by server.py — never called directly by other servers.

    Key design decisions:
        Weights are explicit input — orchestrator decides current vs optimal.
        Static distribution — same σ for all simulation steps.
        Antithetic variates — variance reduction on CVaR estimator.
        Terminal values — total cumulative return at end of horizon.

    Args:
        log_returns:       dict[symbol, list[float]] — from Market Data Server
        weights:           dict[symbol, float] — current OR optimal weights
        horizon_days:      int — simulation horizon in trading days
        n_simulations:     int — number of paths (must be even for antithetic)
        confidence_levels: list[float] — default [0.95, 0.99]
        distribution:      str — "student_t" | "gaussian" | "historical_bootstrap"
        random_seed:       int | None — set for deterministic testing

    Returns:
        dict matching SimulationOutput schema in orchestrator/state.py:
            cvar_95:           float — Expected Shortfall at 95%
            cvar_99:           float — Expected Shortfall at 99%
            var_95:            float — VaR at 95% (context only)
            var_99:            float — VaR at 99% (context only)
            percentiles:       dict  — p10, p25, p50, p75, p90
            n_simulations:     int   — echoed
            distribution_used: str   — echoed
            fitted_nu:         float | None — Student-t ν, None otherwise

    Raises:
        ValueError: if distribution is invalid
        ValueError: if weights do not sum to 1
        ValueError: if insufficient observations for fitting
    """
    # ── Input validation ──────────────────────────────────────────
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    if distribution not in VALID_DISTRIBUTIONS:
        raise ValueError(
            f"Invalid distribution '{distribution}'. "
            f"Must be one of: {VALID_DISTRIBUTIONS}"
        )

    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")

    if n_simulations < 2:
        raise ValueError(f"n_simulations must be >= 2, got {n_simulations}")

    # Ensure even number for antithetic variates
    if n_simulations % 2 != 0:
        n_simulations += 1

    # ── Portfolio returns ─────────────────────────────────────────
    portfolio_returns = _compute_portfolio_returns(log_returns, weights)

    if len(portfolio_returns) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Portfolio return series has {len(portfolio_returns)} observations. "
            f"Minimum {MIN_OBSERVATIONS} required for reliable simulation."
        )

    # ── Random number generator ───────────────────────────────────
    rng = np.random.default_rng(random_seed)

    # ── Generate paths ────────────────────────────────────────────
    fitted_nu = None

    if distribution == "student_t":
        nu, mu, sigma = _fit_student_t(portfolio_returns)
        fitted_nu = nu
        paths = _generate_student_t_paths(
            nu, mu, sigma, n_simulations, horizon_days, rng
        )

    elif distribution == "gaussian":
        mu, sigma = _fit_gaussian(portfolio_returns)
        paths = _generate_gaussian_paths(
            mu, sigma, n_simulations, horizon_days, rng
        )

    else:  # historical_bootstrap
        paths = _historical_bootstrap(
            portfolio_returns, n_simulations, horizon_days, rng
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
        "distribution_used": distribution,
        "fitted_nu":         fitted_nu,
    }