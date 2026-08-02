"""
servers/risk_engine/tools/rolling_cvar.py

Implementation of the compute_rolling_cvar tool.

Responsibilities:
    - Construct the daily portfolio log-return series from per-asset
      log-returns and current weights.
    - Slide a fixed-length window (default 252 trading days) across that
      series and compute Historical CVaR at 95% on each window.
    - Return the resulting rolling-CVaR series plus its mean.

Why this exists:
    The Risk tab's "Historical Risk Evolution" chart previously plotted a
    seeded random walk anchored at the current empirical CVaR — fabricated
    data, disclosed only via an in-app note. This tool computes the real
    thing: how the portfolio's 252-day tail risk actually evolved over the
    observed history, holding today's weights fixed.

Method (matched to risk_metrics.py conventions — deliberately reused, not
re-derived, so the numbers reconcile):
    Portfolio return series:
        r_portfolio_t = Σ w_i · r_i_t   (weighted sum of log-returns)
        Identical to _compute_portfolio_returns. Same < 0.01%/day
        log-additivity approximation already documented there.

    Rolling CVaR at window end-day t (t indexed from window-1 .. T-1):
        window   = portfolio_returns[t-window+1 : t+1]     (length = window)
        VaR_95   = -percentile(window, 5.0)                (positive loss)
        CVaR_95  = -mean(window[window < -VaR_95])         (positive loss)
        Reuses _compute_var and _compute_cvar verbatim.

    Fixed weights:
        The rolling series answers "what would today's allocation's tail
        risk have looked like over the past window?" — a backtest of the
        current portfolio, not a reconstruction of historically-held
        weights (which the app does not track).

Relationship to the reported cvar_95:
    The reported RiskMetricsResult.cvar_95 is computed over the FULL
    available window (computation_window = "2y", ~497 returns). The LAST
    point of this rolling series uses only the trailing `window` days, so
    the two will generally differ. That is correct and expected — they are
    different estimation windows — and is more honest than forcing the
    endpoint to match by construction (as the old fabricated chart did).

Series length:
    With ~497 daily log-returns and a 252-day window, the rolling series
    has ~246 points (T - window + 1). The frontend labels the x-axis as
    "window end-day index" over that range.

This file contains pure computation logic only.
No MCP protocol code — that lives in server.py.
"""

from __future__ import annotations

import logging

import numpy as np

# Reuse the exact primitives the point-in-time risk metrics use, so the
# rolling series is measured identically to the reported cvar_95.
from servers.risk_engine.tools.risk_metrics import (
    TRADING_DAYS_PER_YEAR,
    _compute_cvar,
    _compute_portfolio_returns,
    _compute_var,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_WINDOW = 252            # trading days — 1 year rolling window
DEFAULT_WINDOWS = (21, 63, 252) # 1M / 3M / 1Y — frontend selector options
CVAR_PERCENTILE = 5.0           # 95% confidence → 5th percentile tail
MIN_SERIES_FOR_ROLLING = DEFAULT_WINDOW + 1   # need at least one full window


def compute_rolling_cvar(
    log_returns: dict[str, list[float]],
    weights: dict[str, float],
    window: int = DEFAULT_WINDOW,
    percentile: float = CVAR_PERCENTILE,
) -> dict:
    """
    Compute a rolling Historical CVaR(95%) series for the portfolio.

    This is the implementation of the compute_rolling_cvar MCP tool.
    Called by compute_risk node — never called directly by other servers.

    Args:
        log_returns: dict[symbol, list[float]] — from Market Data Server.
        weights:     dict[symbol, float] — current portfolio weights (sum to 1).
        window:      int — rolling window length in trading days (default 252).
        percentile:  float — tail percentile, 5.0 for CVaR_95 (default).

    Returns:
        dict matching RollingCVaRResult schema in orchestrator/state.py:
            rolling_cvar:  list[float] — CVaR_95 per window end-day (positive loss)
            window_end:    list[int]   — end-day index for each rolling point,
                                         aligned to the portfolio return series
            mean_cvar:     float       — mean of the rolling series
            window_size:   int         — echoed window length
            n_points:      int         — len(rolling_cvar)
            computation_window: str    — echoed audit string

    Raises:
        ValueError: if window < 2.
        ValueError: if the portfolio return series is shorter than one window.
        (Weight/length/symbol validation is delegated to
         _compute_portfolio_returns, matching compute_risk_metrics.)
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    # Reuses the same validation (weights sum to 1, symbols present, equal
    # lengths) and the same weighted-sum construction as compute_risk_metrics.
    portfolio_returns = _compute_portfolio_returns(log_returns, weights)
    n = len(portfolio_returns)

    if n < window + 1:
        raise ValueError(
            f"Return series has {n} observations — need at least {window + 1} "
            f"for a rolling {window}-day CVaR series. Use a shorter window or "
            f"a longer history."
        )

    rolling_cvar: list[float] = []
    rolling_vol:  list[float] = []
    window_end: list[int] = []

    # End-day index t runs from (window - 1) to (n - 1) inclusive.
    # The window is the trailing `window` returns ending at day t.
    for t in range(window - 1, n):
        w = portfolio_returns[t - window + 1 : t + 1]

        var = _compute_var(w, percentile=percentile)
        try:
            cvar = _compute_cvar(w, var=var)
        except ValueError:
            # Degenerate window with no observations strictly beyond the VaR
            # threshold (e.g. ties at the percentile boundary). Fall back to
            # the VaR itself as a conservative tail estimate rather than
            # dropping the point and breaking x-axis alignment.
            logger.warning(
                "compute_rolling_cvar: empty tail at end-day %d — "
                "falling back to VaR as CVaR estimate",
                t,
            )
            cvar = var

        # Annualised volatility on the same window — matches risk_metrics:
        # std_daily(ddof=1) × √252.
        vol = float(np.std(w, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

        rolling_cvar.append(cvar)
        rolling_vol.append(vol)
        window_end.append(t)

    mean_cvar = float(np.mean(rolling_cvar)) if rolling_cvar else 0.0
    mean_vol  = float(np.mean(rolling_vol)) if rolling_vol else 0.0

    return {
        "rolling_cvar":       rolling_cvar,
        "rolling_vol":        rolling_vol,
        "window_end":         window_end,
        "mean_cvar":          mean_cvar,
        "mean_vol":           mean_vol,
        "window_size":        window,
        "n_points":           len(rolling_cvar),
        "computation_window": "2y",
    }


def compute_rolling_risk(
    log_returns: dict[str, list[float]],
    weights: dict[str, float],
    windows: list[int] | None = None,
    percentile: float = CVAR_PERCENTILE,
) -> dict:
    """
    Compute rolling CVaR and rolling volatility across multiple windows in
    one call, so the frontend window selector (1M / 3M / 1Y) can switch
    between series client-side with no re-fetch and no re-computation.

    This is the implementation of the compute_rolling_risk MCP tool.
    Called by compute_risk node (current weights) and optimise node
    (optimal weights).

    Args:
        log_returns: dict[symbol, list[float]] — from Market Data Server.
        weights:     dict[symbol, float] — portfolio weights (sum to 1).
        windows:     list[int] — rolling windows in trading days.
                     Defaults to [21, 63, 252] (1M / 3M / 1Y).
        percentile:  float — tail percentile, 5.0 for CVaR_95.

    Returns:
        dict matching RollingRiskResult schema in orchestrator/state.py:
            windows: dict[str(window) -> RollingWindowResult-shaped dict],
                     one entry per window that the series was long enough to
                     support. A window longer than the available history is
                     skipped (logged) rather than raising, so a short history
                     still yields whatever windows it can support.
            computation_window: str — echoed audit string.

    Raises:
        ValueError: only if NO window can be supported (history shorter than
                    the smallest requested window + 1). Per-window shortfalls
                    are skipped, not fatal.
    """
    if windows is None:
        windows = list(DEFAULT_WINDOWS)

    out: dict[str, dict] = {}
    for window in windows:
        try:
            out[str(window)] = compute_rolling_cvar(
                log_returns=log_returns,
                weights=weights,
                window=window,
                percentile=percentile,
            )
        except ValueError as exc:
            logger.warning(
                "compute_rolling_risk: skipping window %d — %s", window, exc
            )

    if not out:
        raise ValueError(
            f"History too short for any requested window {windows}. "
            f"Need at least {min(windows) + 1} observations."
        )

    return {
        "windows":            out,
        "computation_window": "2y",
    }


def rolling_risk_to_state(raw: dict):
    """
    Map a compute_rolling_risk() dict into a RollingRiskResult state model.

    Shared by compute_risk (current weights) and optimise (optimal weights)
    so the mapping lives in exactly one place. Imported lazily to avoid a
    circular import (state imports nothing from tools; tools may import from
    state only inside functions).
    """
    from orchestrator.state import RollingRiskResult, RollingWindowResult

    return RollingRiskResult(
        windows={
            wk: RollingWindowResult(
                rolling_cvar=wd["rolling_cvar"],
                rolling_vol=wd["rolling_vol"],
                window_end=wd["window_end"],
                mean_cvar=wd["mean_cvar"],
                mean_vol=wd["mean_vol"],
                window_size=wd["window_size"],
                n_points=wd["n_points"],
            )
            for wk, wd in raw["windows"].items()
        },
        computation_window=raw["computation_window"],
    )