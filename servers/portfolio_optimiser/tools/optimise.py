"""
servers/portfolio_optimiser/tools/optimise.py

Implementation of the optimise_portfolio tool.

Responsibilities:
    - Compute expected returns and covariance matrix from log-returns
    - Scan the Efficient Frontier (Pareto frontier of return vs risk)
      using the Scanning method: N minimum-variance solves at fixed
      target return levels
    - Find the Maximum Sharpe portfolio (tangency point on the CML)
      using a dedicated Sharpe maximisation solve
    - Return frontier points and Maximum Sharpe weights

Two sequential steps (not alternative approaches):
    Step 1 — Scan Method:
        Maps the complete Efficient Frontier.
        N solves, each minimising w^T Σ w subject to w^T μ = r_i.
        Output: the shape of the frontier (N points).

    Step 2 — Maximum Sharpe Solve:
        Finds the tangency point on the frontier produced by Step 1.
        Single solve minimising -(w^T μ - rfr) / √(w^T Σ w).
        Output: the single best portfolio given the risk-free rate.

The Efficient Frontier is a property of the risky assets alone —
it does not change when the risk-free rate changes.
The Maximum Sharpe point moves along the frontier as rfr changes.

Solver versioning (via `solver` field in tool input):
    V1: "convex_qp"              — scipy SLSQP (current)
    V2: "differential_evolution" — scipy DE for non-smooth constraints
    V3: "nsga2"                  — pymoo NSGA-II for true multi-objective

This file contains pure computation logic only.
No MCP protocol code — that lives in server.py.

Mathematical notation:
    w  — weight vector (N × 1)
    μ  — expected return vector (N × 1), annualised
    Σ  — covariance matrix (N × N), annualised
    rfr — risk-free rate, annualised
    portfolio_return     = w^T μ
    portfolio_variance   = w^T Σ w
    portfolio_volatility = √(w^T Σ w)
    Sharpe ratio         = (w^T μ - rfr) / √(w^T Σ w)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# ── Constants ─────────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
VALID_SOLVERS = {"convex_qp", "differential_evolution", "nsga2"}

# Optimisation tolerance — convergence criterion for scipy
OPTIMIZER_TOL = 1e-10

# Number of random restarts for Maximum Sharpe solve
# Helps avoid local minima in non-convex Sharpe landscape
MAX_SHARPE_RESTARTS = 5


# ── Step 0: Statistical inputs ────────────────────────────────────────────────

def _compute_expected_returns(
    log_returns: dict[str, list[float]],
) -> tuple[np.ndarray, list[str]]:
    """
    Compute annualised expected return per asset from historical log-returns.

    expected_return_i = mean(log_returns_i) × 252

    Args:
        log_returns: dict[symbol, list[float]]

    Returns:
        tuple:
            mu:      np.ndarray shape (N,) — annualised expected returns
            symbols: list[str] — asset order (consistent with mu and Σ)
    """
    symbols = list(log_returns.keys())
    mu = np.array([
        np.mean(log_returns[s]) * TRADING_DAYS_PER_YEAR
        for s in symbols
    ])
    return mu, symbols


def _compute_covariance_matrix(
    log_returns: dict[str, list[float]],
    symbols: list[str],
) -> np.ndarray:
    """
    Compute annualised covariance matrix from historical log-returns.

    Σ_annualised = Σ_daily × 252

    Uses ddof=1 (sample covariance) — standard in finance.
    The covariance matrix captures both individual asset variance
    (diagonal) and cross-asset covariance (off-diagonal).

    Args:
        log_returns: dict[symbol, list[float]]
        symbols:     list[str] — asset order (must match mu order)

    Returns:
        np.ndarray shape (N, N) — annualised covariance matrix

    Raises:
        ValueError: if covariance matrix is not positive semi-definite
    """
    returns_matrix = np.array([log_returns[s] for s in symbols])
    cov_daily = np.cov(returns_matrix, ddof=1)
    cov_annual = cov_daily * TRADING_DAYS_PER_YEAR

    # Validate positive semi-definiteness
    # A valid covariance matrix must have non-negative eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov_annual)
    if np.any(eigenvalues < -1e-8):
        raise ValueError(
            "Covariance matrix is not positive semi-definite. "
            "Return series may be too short or contain identical assets."
        )

    return cov_annual


# ── Objective and constraint functions ────────────────────────────────────────

def _portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Compute portfolio variance: w^T Σ w

    This is the objective function for the Efficient Frontier scan.
    We minimise this for each target return level.

    Args:
        weights: np.ndarray shape (N,)
        cov:     np.ndarray shape (N, N) — annualised covariance matrix

    Returns:
        float — portfolio variance (not volatility)
    """
    return float(weights @ cov @ weights)


def _portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """
    Compute portfolio expected return: w^T μ

    Used in the target return constraint for the frontier scan.

    Args:
        weights: np.ndarray shape (N,)
        mu:      np.ndarray shape (N,) — annualised expected returns

    Returns:
        float — portfolio expected return
    """
    return float(weights @ mu)


def _negative_sharpe(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rfr: float,
) -> float:
    """
    Compute negative Sharpe ratio for minimisation.

    Sharpe = (w^T μ - rfr) / √(w^T Σ w)
    We minimise the negative to maximise Sharpe.

    A small epsilon is added to variance to prevent division by zero
    in degenerate cases (e.g. single-asset portfolio).

    Args:
        weights: np.ndarray shape (N,)
        mu:      np.ndarray shape (N,) — annualised expected returns
        cov:     np.ndarray shape (N, N) — annualised covariance matrix
        rfr:     float — annualised risk-free rate

    Returns:
        float — negative Sharpe ratio
    """
    port_return = weights @ mu
    port_variance = weights @ cov @ weights
    port_vol = np.sqrt(port_variance + 1e-12)
    return float(-(port_return - rfr) / port_vol)


# ── Constraint and bound builders ─────────────────────────────────────────────

def _build_bounds(
    n_assets: int,
    min_weight: float,
    max_weight: float,
) -> list[tuple[float, float]]:
    """
    Build weight bounds for scipy optimiser.

    Returns a list of (min, max) tuples — one per asset.
    scipy interpret this as: min_weight <= w_i <= max_weight for all i.
    """
    return [(min_weight, max_weight)] * n_assets


def _build_constraints(
    mu: np.ndarray,
    target_return: float | None = None,
) -> list[dict]:
    """
    Build scipy constraint list.

    Always includes: weights sum to 1 (equality constraint)
    Optionally includes: portfolio return equals target (equality constraint)

    Args:
        mu:            np.ndarray — expected returns
        target_return: float | None — if provided, adds return constraint

    Returns:
        list[dict] — scipy-compatible constraint specifications
    """
    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
            "jac": lambda w: np.ones(len(w)),
        }
    ]

    if target_return is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w, r=target_return: _portfolio_return(w, mu) - r,
            "jac": lambda w: mu,
        })

    return constraints


# ── Step 1: Efficient Frontier scan ──────────────────────────────────────────

def _scan_efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    symbols: list[str],
    n_points: int,
    min_weight: float,
    max_weight: float,
) -> list[dict]:
    """
    Scan the Efficient Frontier using the minimum-variance approach.

    For each target return level r in linspace(r_min, r_max, n_points):
        minimise:   w^T Σ w
        subject to: w^T μ = r     (achieve exactly this return)
                    Σ w_i = 1     (fully invested)
                    w_i >= min_weight
                    w_i <= max_weight

    The target return range is determined by the feasible portfolio
    returns given the weight constraints:
        r_min: return of the global minimum variance portfolio
        r_max: maximum achievable return given weight constraints

    Args:
        mu:         np.ndarray (N,) — annualised expected returns
        cov:        np.ndarray (N, N) — annualised covariance matrix
        symbols:    list[str] — asset names in same order as mu
        n_points:   int — number of frontier points to compute
        min_weight: float — minimum weight per asset
        max_weight: float — maximum weight per asset

    Returns:
        list[dict] — each dict: {volatility, expected_return, weights}
        Sorted by volatility ascending (left to right on frontier plot)
    """
    n_assets = len(symbols)
    bounds = _build_bounds(n_assets, min_weight, max_weight)

    # Determine feasible return range
    # r_min: equal-weight portfolio return (conservative lower bound)
    # r_max: maximum single-asset return achievable within bounds
    equal_weights = np.ones(n_assets) / n_assets
    r_min = float(equal_weights @ mu) * 0.5  # allow below equal-weight
    r_max = float(np.max(mu)) * max_weight + float(np.sum(np.sort(mu)[-2:])) * min_weight

    # Clip to realistic bounds
    r_min = max(r_min, float(np.min(mu)))
    r_max = min(r_max, float(np.max(mu)))

    target_returns = np.linspace(r_min, r_max, n_points)

    frontier_points = []
    initial_weights = equal_weights.copy()

    for target_r in target_returns:
        constraints = _build_constraints(mu, target_return=target_r)

        result = minimize(
            fun=_portfolio_variance,
            x0=initial_weights,
            args=(cov,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            tol=OPTIMIZER_TOL,
            options={"maxiter": 1000, "ftol": OPTIMIZER_TOL},
        )

        if not result.success:
            # Skip infeasible target return levels silently
            # (can occur near the edges of the feasible region)
            continue

        w = result.x
        port_vol = float(np.sqrt(w @ cov @ w))
        port_ret = float(w @ mu)

        frontier_points.append({
            "volatility":      port_vol,
            "expected_return": port_ret,
            "weights":         {s: float(w[i]) for i, s in enumerate(symbols)},
        })

        # Warm start — use previous solution as initial weights
        initial_weights = w.copy()

    # Sort by volatility ascending
    frontier_points.sort(key=lambda p: p["volatility"])

    return frontier_points


# ── Step 2: Maximum Sharpe solve ──────────────────────────────────────────────

def _find_max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    symbols: list[str],
    rfr: float,
    min_weight: float,
    max_weight: float,
) -> dict:
    """
    Find the Maximum Sharpe portfolio (tangency point on the CML).

    Solves:
        minimise:   -(w^T μ - rfr) / √(w^T Σ w)
        subject to: Σ w_i = 1
                    w_i >= min_weight
                    w_i <= max_weight

    No target return constraint — the solver is free to choose any
    return level. It will find the one that maximises return per
    unit of risk, which is the tangency point.

    Uses multiple random restarts (MAX_SHARPE_RESTARTS) to avoid
    local minima in the non-convex Sharpe landscape.

    Args:
        mu:         np.ndarray (N,) — annualised expected returns
        cov:        np.ndarray (N, N) — annualised covariance matrix
        symbols:    list[str] — asset names in same order as mu
        rfr:        float — annualised risk-free rate
        min_weight: float — minimum weight per asset
        max_weight: float — maximum weight per asset

    Returns:
        dict:
            weights:              dict[symbol, float]
            expected_return:      float
            portfolio_volatility: float
            sharpe_ratio:         float

    Raises:
        RuntimeError: if all restarts fail to converge
    """
    n_assets = len(symbols)
    bounds = _build_bounds(n_assets, min_weight, max_weight)
    constraints = _build_constraints(mu)

    best_result = None
    best_sharpe = -np.inf

    np.random.seed(42)  # reproducible restarts

    for restart in range(MAX_SHARPE_RESTARTS):
        if restart == 0:
            # First try: equal weights
            w0 = np.ones(n_assets) / n_assets
        else:
            # Subsequent tries: random Dirichlet weights
            w0 = np.random.dirichlet(np.ones(n_assets))
            # Clip to bounds
            w0 = np.clip(w0, min_weight, max_weight)
            w0 = w0 / w0.sum()

        result = minimize(
            fun=_negative_sharpe,
            x0=w0,
            args=(mu, cov, rfr),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            tol=OPTIMIZER_TOL,
            options={"maxiter": 1000, "ftol": OPTIMIZER_TOL},
        )

        if not result.success:
            continue

        w = result.x
        sharpe = -result.fun  # negate back to positive

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_result = w.copy()

    if best_result is None:
        raise RuntimeError(
            f"Maximum Sharpe optimisation failed after {MAX_SHARPE_RESTARTS} restarts. "
            f"Try relaxing weight constraints or check return series quality."
        )

    w = best_result
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    sharpe = (port_ret - rfr) / port_vol if port_vol > 0 else 0.0

    return {
        "weights":              {s: float(w[i]) for i, s in enumerate(symbols)},
        "expected_return":      port_ret,
        "portfolio_volatility": port_vol,
        "sharpe_ratio":         sharpe,
    }


# ── Main tool function ────────────────────────────────────────────────────────

def optimise_portfolio(
    log_returns: dict[str, list[float]],
    risk_free_rate: float = 0.065,
    n_frontier_points: int = 50,
    constraints: dict | None = None,
    solver: str = "convex_qp",
) -> dict:
    """
    Run Markowitz mean-variance optimisation.

    Two sequential steps:
        Step 1 — Scan Method → Efficient Frontier (N points)
        Step 2 — Sharpe Solve → Maximum Sharpe portfolio (1 point)

    This is the implementation of the optimise_portfolio MCP tool.
    Called by server.py — never called directly by other servers.

    Args:
        log_returns:       dict[symbol, list[float]] — from Market Data Server
        risk_free_rate:    float — annualised RFR, default 0.065
        n_frontier_points: int — number of frontier points, default 50
        constraints:       dict with optional keys:
                               min_weight:    float, default 0.0
                               max_weight:    float, default 1.0
                               sector_caps:   dict[str, float] (v2)
                               target_return: float | None
        solver:            str — "convex_qp" | "differential_evolution" | "nsga2"
                               only "convex_qp" implemented in v1

    Returns:
        dict matching OptimisationResult schema in orchestrator/state.py:
            optimal_weights:      dict[symbol, float]
            max_sharpe_weights:   dict[symbol, float]
            expected_return:      float
            portfolio_volatility: float
            sharpe_ratio:         float
            cml_slope:            float
            efficient_frontier:   list[{volatility, expected_return, weights}]
            solver_used:          str

    Raises:
        ValueError: if solver is invalid or inputs are malformed
        RuntimeError: if optimisation fails to converge
    """
    # ── Input validation ──────────────────────────────────────────
    if solver not in VALID_SOLVERS:
        raise ValueError(
            f"Invalid solver '{solver}'. Must be one of: {VALID_SOLVERS}"
        )

    if solver != "convex_qp":
        raise NotImplementedError(
            f"Solver '{solver}' is not implemented in v1. "
            f"Only 'convex_qp' is available. "
            f"'differential_evolution' is planned for v2, 'nsga2' for v3."
        )

    if not log_returns:
        raise ValueError("log_returns dict must not be empty")

    if len(log_returns) < 2:
        raise ValueError(
            f"At least 2 assets required for portfolio optimisation, "
            f"got {len(log_returns)}."
        )

    if risk_free_rate < 0:
        raise ValueError(
            f"risk_free_rate must be non-negative, got {risk_free_rate}"
        )

    if n_frontier_points < 2:
        raise ValueError(
            f"n_frontier_points must be >= 2, got {n_frontier_points}"
        )

    # ── Parse constraints ─────────────────────────────────────────
    constraints = constraints or {}
    min_weight    = float(constraints.get("min_weight", 0.0))
    max_weight    = float(constraints.get("max_weight", 1.0))
    target_return = constraints.get("target_return", None)

    if min_weight < 0:
        raise ValueError(
            f"min_weight must be >= 0 (no short selling in v1), "
            f"got {min_weight}"
        )

    if max_weight > 1.0:
        raise ValueError(
            f"max_weight must be <= 1.0, got {max_weight}"
        )

    if min_weight >= max_weight:
        raise ValueError(
            f"min_weight ({min_weight}) must be < max_weight ({max_weight})"
        )

    # ── Step 0: Compute statistical inputs ────────────────────────
    mu, symbols = _compute_expected_returns(log_returns)
    cov = _compute_covariance_matrix(log_returns, symbols)

    # ── Step 1: Efficient Frontier scan ──────────────────────────
    frontier_points = _scan_efficient_frontier(
        mu=mu,
        cov=cov,
        symbols=symbols,
        n_points=n_frontier_points,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    if len(frontier_points) == 0:
        raise RuntimeError(
            "Efficient Frontier scan produced no valid points. "
            "Try relaxing weight constraints or using more assets."
        )

    # ── Step 2: Maximum Sharpe solve ──────────────────────────────
    max_sharpe = _find_max_sharpe(
        mu=mu,
        cov=cov,
        symbols=symbols,
        rfr=risk_free_rate,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    # cml_slope is the slope of the Capital Market Line
    # = Maximum Sharpe ratio = (return - rfr) / volatility
    cml_slope = max_sharpe["sharpe_ratio"]

    return {
        "optimal_weights":      max_sharpe["weights"],
        "max_sharpe_weights":   max_sharpe["weights"],
        "expected_return":      max_sharpe["expected_return"],
        "portfolio_volatility": max_sharpe["portfolio_volatility"],
        "sharpe_ratio":         max_sharpe["sharpe_ratio"],
        "cml_slope":            cml_slope,
        "efficient_frontier":   frontier_points,
        "solver_used":          solver,
    }