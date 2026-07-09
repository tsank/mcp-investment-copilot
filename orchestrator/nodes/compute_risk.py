"""
orchestrator/nodes/compute_risk.py

Node 3 of 7 — compute_risk

Responsibility:
    Call the Risk Engine Server to compute backward-looking risk metrics
    and GARCH volatility forecasts for the current portfolio.

Reads from AgentState:
    - market_data (MarketDataResult) — log_returns, prices, dates
    - portfolio   (Portfolio)        — holdings (weights), benchmark

Writes to AgentState:
    - risk_metrics  (RiskMetricsResult) — var_95/99, cvar_95/99, sharpe,
                                          max_drawdown, volatility,
                                          portfolio_return, risk_free_rate
    - garch_result  (GARCHResult)       — per_asset params, vol forecasts,
                                          garch_params, current_vols
                                          (handoff fields for simulate node)

MCP tools called (both against "risk_engine" server):
    1. compute_risk_metrics   → RiskMetricsResult
    2. compute_garch_forecast → GARCHResult

Design decisions:

    Single subprocess — both calls share one ClientSession.
    Same pattern as fetch_market_data: get_client() used directly so
    the subprocess is spawned once for both tool calls.

    GARCH horizon hardcoded to 10 days in v1:
        10 trading days = 2 weeks forward volatility forecast.
        Sufficient for short-term risk assessment and simulation seeding.
        Promoted to a configurable parameter in v2.

    Risk-free rate hardcoded to 0.065 (6.5% RBI repo rate proxy):
        Standard Indian market convention for Sharpe ratio computation.
        Promoted to environment variable in v2.

    Guard against None market_data:
        If fetch_market_data failed, market_data is None.
        This node detects that, records the error, and returns
        risk_metrics=None, garch_result=None.
        Downstream nodes (optimise, simulate, check_compliance) also guard.

    Two-phase GARCH (documented in GARCHResult docstring):
        Phase 1 — backward-looking: MLE fits {ω, α, β, ν} from history
        Phase 2 — forward-looking:  deterministic vol path σ_{T+1}..σ_{T+H}
        The garch_params and current_vols fields are the handoff to the
        simulate node — it does not re-fit, it draws paths from these params.
"""

from __future__ import annotations

import asyncio
import logging

from servers.risk_engine.tools.risk_metrics import compute_risk_metrics
from servers.risk_engine.tools.garch_forecast import compute_garch_forecast
from orchestrator.state import (
    AgentState,
    GARCHAssetResult,
    GARCHParams,
    GARCHResult,
    RiskMetricsResult,
)

logger = logging.getLogger(__name__)

# Hardcoded in v1
_RISK_FREE_RATE = 0.065   # RBI repo rate proxy — annualised
_GARCH_HORIZON  = 10      # trading days forward


async def compute_risk(state: AgentState) -> dict:
    """
    LangGraph node — compute_risk.

    Makes two sequential tool calls to the Risk Engine Server:
        1. compute_risk_metrics  → populates risk_metrics
        2. compute_garch_forecast → populates garch_result

    Args:
        state: Current AgentState. Reads: market_data, portfolio.

    Returns:
        dict with keys: risk_metrics, garch_result, execution_trace, errors.
    """
    # ── Guard: upstream failure ───────────────────────────────────────────────
    if state.market_data is None:
        logger.error("compute_risk: market_data is None — skipping")
        return {
            "risk_metrics":    None,
            "garch_result":    None,
            "execution_trace": state.execution_trace + ["compute_risk:skipped"],
            "errors":          state.errors + ["compute_risk: market_data is None"],
        }

    logger.info("compute_risk: starting — %d symbols", len(state.symbols))

    try:
        # --- Call 1: risk metrics -----------------------------
        # Direct function call (v2 - Option B). No subprocess, no JSON
        # round trip. compute_risk_metrics already returns a plain dict
        # of native Python types(confirmed: explicit float() casts
        # throughout risk_metrics.py), so risk_data is used exactly as 
        # risk_data was before, just without the json.loads() step.
        
        logger.info("compute_risk: calling compute_risk_metrics")
        risk_data = await asyncio.to_thread(
            compute_risk_metrics,
            log_returns = state.market_data.log_returns,
            weights = state.portfolio.holdings,
            prices = state.market_data.prices,
            risk_free_rate = _RISK_FREE_RATE,
        )
        logger.info(
            "compute_risk: risk_metrics ok - cvar_95=%.4f sharpe=%.4f",
            risk_data["cvar_95"], risk_data["sharpe_ratio"],
        )
        
        # --- Call 2: GARCH forecast ----------------------------
        # asyncio.to_thread() runs this synchronous, CPU heavy MLE fit
        # in a background thread rather than blocking the event loop
        # directly - matters once this orchestrator may be handling 
        # more than one request at a time (eg concurrent Lambda 
        # invocatons)
        logger.info("compute_risk: calling compute_garch_forecast")
        garch_data = await asyncio.to_thread(
            compute_garch_forecast,
            log_returns=state.market_data.log_returns,
            horizon_days =_GARCH_HORIZON,
        )
        logger.info(
            "compute_risk: garch_forecast ok - model=%s innovations=%s",
            garch_data["garch_model"], garch_data["innovations_used"],
        )

    except Exception as exc:
        logger.error("compute_risk: failed — %s", exc)
        return {
            "risk_metrics":    None,
            "garch_result":    None,
            "execution_trace": state.execution_trace + ["compute_risk:error"],
            "errors":          state.errors + [f"compute_risk: {exc}"],
        }

    # ── Build RiskMetricsResult ───────────────────────────────────────────────
    risk_metrics = RiskMetricsResult(
        var_95=risk_data["var_95"],
        var_99=risk_data["var_99"],
        cvar_95=risk_data["cvar_95"],
        cvar_99=risk_data["cvar_99"],
        sharpe_ratio=risk_data["sharpe_ratio"],
        max_drawdown=risk_data["max_drawdown"],
        volatility=risk_data["volatility"],
        portfolio_return=risk_data["portfolio_return"],
        risk_free_rate=risk_data["risk_free_rate"],
        computation_window=risk_data["computation_window"],
    )

    # ── Build GARCHResult ─────────────────────────────────────────────────────
    # Deserialise per_asset dict into GARCHAssetResult models
    per_asset: dict[str, GARCHAssetResult] = {}
    for symbol, asset_data in garch_data["per_asset"].items():
        try:
            per_asset[symbol] = GARCHAssetResult(
                params=GARCHParams(**asset_data["params"]),
                alpha_plus_beta=asset_data["alpha_plus_beta"],
                current_vol=asset_data["current_vol"],
                longrun_vol=asset_data["longrun_vol"],
                vol_forecast=asset_data["vol_forecast"],
                regime=asset_data["regime"],
                aic=asset_data["aic"],
                bic=asset_data["bic"],
                persistence_warning=asset_data["persistence_warning"],
            )
        except Exception as exc:
            logger.warning(
                "compute_risk: could not parse GARCHAssetResult for %s — %s",
                symbol, exc,
            )

    # Deserialise garch_params handoff field
    garch_params: dict[str, GARCHParams] = {}
    for symbol, p in garch_data["garch_params"].items():
        try:
            garch_params[symbol] = GARCHParams(**p)
        except Exception as exc:
            logger.warning(
                "compute_risk: could not parse GARCHParams for %s — %s",
                symbol, exc,
            )

    garch = GARCHResult(
        per_asset=per_asset,
        portfolio_vol_forecast=garch_data["portfolio_vol_forecast"],
        garch_model=garch_data["garch_model"],
        innovations_used=garch_data["innovations_used"],
        horizon_days=garch_data["horizon_days"],
        garch_params=garch_params,
        current_vols=garch_data["current_vols"],
    )

    return {
        "risk_metrics":    risk_metrics,
        "garch_result":    garch,
        "execution_trace": state.execution_trace + ["compute_risk:ok"],
        "errors":          state.errors,
    }