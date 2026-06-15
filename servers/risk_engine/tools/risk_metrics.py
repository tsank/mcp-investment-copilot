"""
servers/risk_engine/tools/risk_metrics.py

Implementation of the compute_risk_metrics tool.

Responsibilities:
    - Compute portfolio log-returns from per-asset log-returns and weights
    - Compute Historical VaR at 95% and 99% (empirical percentile method)
    - Compute Historical CVaR at 95% and 99% (mean of tail)
    - Compute annualised Sharpe ratio
    - Compute maximum drawdown (peak-to-trough on cumulative return series)
    - Compute per-symbol annualised volatility

All metrics are backward-looking and non-parametric:
    - No distribution is fitted
    - No simulation is performed
    - VaR and CVaR are direct percentiles of observed return history
    - These are descriptive statistics of what actually happened

This file contains pure computation logic only.
No MCP protocol code — that lives in server.py.

Mathematical notes:
    Portfolio return approximation:
        r_portfolio_t ≈ Σ w_i × r_i_t  (weighted sum of log-returns)
        This is an approximation — log-returns are not strictly additive
        across assets. For daily returns of NSE large-caps the error is
        negligible (< 0.01% per day).

    VaR sign convention:
        VaR and CVaR are expressed as positive loss numbers.
        VaR_95 = 0.03 means "the portfolio lost more than 3% on 5% of days."

    Annualisation:
        252 trading days per year (NSE standard).
        Volatility:    std_daily × √252
        Sharpe ratio:  (mean_daily - rfr_daily) / std_daily × √252
        rfr_daily:     rfr_annual / 252
"""

from __future__ import annotations

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
CONFIDENCE_LEVELS = {0.95: 5.0, 0.99: 1.0}  # confidence → percentile


# ── Portfolio return computation ──────────────────────────────────────────────

def _compute_portfolio_returns(
    log_returns: dict[str, list[float]],
    weights: dict[str, float],
) -> np.ndarray:
    """
    Compute daily portfolio log-returns as weighted sum of asset log-returns.

    r_portfolio_t ≈ Σ w_i × r_i_t

    This is an approximation for log-returns. For simple returns it would
    be exact. For daily NSE large-cap returns the approximation error
    is negligible.

    Args:
        log_returns: dict[symbol, list[float]] — from Market Data Server
        weights:     dict[symbol, float] — portfolio weights, must sum to 1

    Returns:
        np.ndarray of shape (T,) — daily portfolio returns

    Raises:
        ValueError: if symbols in weights are not in log_returns
        ValueError: if return series have different lengths
        ValueError: if weights do not sum to approximately 1
    """
    # Validate weights sum to 1
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got {weight_sum:.6f}. "
            f"Please normalise weights before calling."
        )

    # Validate all weighted symbols have return data
    missing = set(weights.keys()) - set(log_returns.keys())
    if missing:
        raise ValueError(
            f"Symbols in weights not found in log_returns: {missing}"
        )

    # Validate all return series have the same length
    symbols = list(weights.keys())
    lengths = {s: len(log_returns[s]) for s in symbols}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"Return series have different lengths: {lengths}. "
            f"Ensure date alignment was performed in Market Data Server."
        )

    # Compute weighted portfolio returns
    n = lengths[symbols[0]]
    portfolio_returns = np.zeros(n)
    for symbol, weight in weights.items():
        portfolio_returns += weight * np.array(log_returns[symbol])

    return portfolio_returns


# ── VaR and CVaR ─────────────────────────────────────────────────────────────

def _compute_var(portfolio_returns: np.ndarray, percentile: float) -> float:
    """
    Compute Historical VaR at a given percentile.

    VaR = -percentile(r_portfolio, percentile)

    Expressed as a positive loss number.
    VaR_95: percentile=5.0  → 5th percentile of return distribution
    VaR_99: percentile=1.0  → 1st percentile of return distribution

    Args:
        portfolio_returns: np.ndarray of daily portfolio returns
        percentile:        float — e.g. 5.0 for VaR_95, 1.0 for VaR_99

    Returns:
        float — VaR expressed as positive loss
    """
    return float(-np.percentile(portfolio_returns, percentile))


def _compute_cvar(portfolio_returns: np.ndarray, var: float) -> float:
    """
    Compute Historical CVaR (Expected Shortfall) given a VaR threshold.

    CVaR = -mean(r_portfolio[r_portfolio < -var])

    CVaR is the mean of all returns that are worse than the VaR threshold.
    Expressed as a positive loss number.

    Args:
        portfolio_returns: np.ndarray of daily portfolio returns
        var:               float — VaR threshold (positive loss number)

    Returns:
        float — CVaR expressed as positive loss

    Raises:
        ValueError: if no returns fall below the VaR threshold
    """
    tail = portfolio_returns[portfolio_returns < -var]

    if len(tail) == 0:
        raise ValueError(
            f"No returns below VaR threshold of {var:.6f}. "
            f"Return series may be too short or VaR threshold too extreme."
        )

    return float(-np.mean(tail))


# ── Sharpe ratio ──────────────────────────────────────────────────────────────

def _compute_sharpe(
    portfolio_returns: np.ndarray,
    risk_free_rate: float,
) -> float:
    """
    Compute annualised Sharpe ratio.

    Sharpe = (mean_daily_return - rfr_daily) / std_daily × √252

    where rfr_daily = rfr_annual / 252

    Args:
        portfolio_returns: np.ndarray of daily portfolio returns
        risk_free_rate:    float — annualised risk-free rate e.g. 0.065

    Returns:
        float — annualised Sharpe ratio

    Raises:
        ValueError: if portfolio return standard deviation is zero
    """
    rfr_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    mean_daily = float(np.mean(portfolio_returns))
    std_daily = float(np.std(portfolio_returns, ddof=1))

    if std_daily < 1e-10:
        raise ValueError(
            "Portfolio return standard deviation is zero. "
            "Cannot compute Sharpe ratio — all returns are identical."
        )

    sharpe = (mean_daily - rfr_daily) / std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(sharpe)


# ── Maximum drawdown ──────────────────────────────────────────────────────────

def _compute_max_drawdown(prices: dict[str, list[float]], weights: dict[str, float]) -> float:
    """
    Compute maximum drawdown of the portfolio on the price series.

    Maximum drawdown = min(cumulative_return - running_max_cumulative_return)

    Steps:
        1. Compute weighted portfolio price index from individual prices
        2. Compute cumulative returns from portfolio price index
        3. Compute running maximum of cumulative returns
        4. Drawdown at each point = cumulative_return - running_max
        5. Maximum drawdown = minimum drawdown value

    Expressed as a negative fraction e.g. -0.23 means 23% peak-to-trough loss.

    Args:
        prices:  dict[symbol, list[float]] — daily closing prices
        weights: dict[symbol, float] — portfolio weights

    Returns:
        float — maximum drawdown as negative fraction
    """
    # Build weighted portfolio price index
    symbols = list(weights.keys())
    n = len(prices[symbols[0]])
    portfolio_price = np.zeros(n)

    for symbol, weight in weights.items():
        portfolio_price += weight * np.array(prices[symbol])

    # Compute cumulative return from portfolio price index
    # Normalise to start at 1.0
    portfolio_price = portfolio_price / portfolio_price[0]

    # Running maximum
    running_max = np.maximum.accumulate(portfolio_price)

    # Drawdown at each point
    drawdown = (portfolio_price - running_max) / running_max

    return float(np.min(drawdown))


# ── Per-symbol volatility ─────────────────────────────────────────────────────

def _compute_volatility(log_returns: dict[str, list[float]]) -> dict[str, float]:
    """
    Compute annualised volatility for each symbol.

    vol_i = std(r_i) × √252

    Uses ddof=1 (sample standard deviation) — standard in finance.

    Args:
        log_returns: dict[symbol, list[float]]

    Returns:
        dict[symbol, float] — annualised volatility per symbol
    """
    volatility = {}
    for symbol, returns in log_returns.items():
        arr = np.array(returns)
        vol = float(np.std(arr, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        volatility[symbol] = vol
    return volatility


# ── Main tool function ────────────────────────────────────────────────────────

def compute_risk_metrics(
    log_returns: dict[str, list[float]],
    weights: dict[str, float],
    prices: dict[str, list[float]],
    risk_free_rate: float = 0.065,
) -> dict:
    """
    Compute backward-looking, non-parametric risk metrics for a portfolio.

    This is the implementation of the compute_risk_metrics MCP tool.
    Called by server.py — never called directly by other servers.

    All metrics are empirical — derived directly from the observed
    return history. No distribution is fitted, no simulation is performed.

    Args:
        log_returns:    dict[symbol, list[float]] — from Market Data Server
        weights:        dict[symbol, float] — current portfolio weights
        prices:         dict[symbol, list[float]] — for max drawdown computation
        risk_free_rate: float — annualised RFR, default 0.065 (RBI repo rate proxy)

    Returns:
        dict matching the RiskMetricsResult schema in orchestrator/state.py:
            var_95:             float — Historical VaR at 95%
            var_99:             float — Historical VaR at 99%
            cvar_95:            float — Historical CVaR at 95% (primary metric)
            cvar_99:            float — Historical CVaR at 99%
            sharpe_ratio:       float — annualised Sharpe ratio
            max_drawdown:       float — peak-to-trough, negative fraction
            volatility:         dict[symbol, float] — annualised per symbol
            portfolio_return:   float — annualised mean portfolio return
            risk_free_rate:     float — echoed for audit
            computation_window: str  — echoed for audit

    Raises:
        ValueError: if weights do not sum to 1
        ValueError: if symbols in weights not in log_returns
        ValueError: if return series have different lengths
    """
    # ── Input validation ──────────────────────────────────────────
    if not weights:
        raise ValueError("weights dict must not be empty")

    if not log_returns:
        raise ValueError("log_returns dict must not be empty")

    if risk_free_rate < 0:
        raise ValueError(
            f"risk_free_rate must be non-negative, got {risk_free_rate}"
        )

    # ── Portfolio returns ─────────────────────────────────────────
    portfolio_returns = _compute_portfolio_returns(log_returns, weights)

    # ── VaR and CVaR ──────────────────────────────────────────────
    var_95 = _compute_var(portfolio_returns, percentile=5.0)
    var_99 = _compute_var(portfolio_returns, percentile=1.0)
    cvar_95 = _compute_cvar(portfolio_returns, var=var_95)
    cvar_99 = _compute_cvar(portfolio_returns, var=var_99)

    # ── Sharpe ratio ──────────────────────────────────────────────
    sharpe = _compute_sharpe(portfolio_returns, risk_free_rate)

    # ── Maximum drawdown ──────────────────────────────────────────
    max_dd = _compute_max_drawdown(prices, weights)

    # ── Per-symbol volatility ─────────────────────────────────────
    volatility = _compute_volatility(log_returns)

    # ── Annualised portfolio return ───────────────────────────────
    portfolio_return = float(np.mean(portfolio_returns) * TRADING_DAYS_PER_YEAR)

    return {
        "var_95":             var_95,
        "var_99":             var_99,
        "cvar_95":            cvar_95,
        "cvar_99":            cvar_99,
        "sharpe_ratio":       sharpe,
        "max_drawdown":       max_dd,
        "volatility":         volatility,
        "portfolio_return":   portfolio_return,
        "risk_free_rate":     risk_free_rate,
        "computation_window": "2y",
    }