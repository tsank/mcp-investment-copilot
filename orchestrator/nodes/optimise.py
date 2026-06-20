"""
orchestrator/nodes/optimise.py

Node 4 of 7 — optimise

Responsibility:
    Call the Portfolio Optimiser Server to compute the Markowitz
    maximum-Sharpe portfolio and efficient frontier.

Reads from AgentState:
    - market_data  (MarketDataResult)  — log_returns
    - risk_metrics (RiskMetricsResult) — risk_free_rate (echoed back)

Writes to AgentState:
    - optimisation_result (OptimisationResult) — optimal_weights,
                                                  max_sharpe_weights,
                                                  expected_return,
                                                  portfolio_volatility,
                                                  sharpe_ratio,
                                                  cml_slope,
                                                  efficient_frontier,
                                                  solver_used

MCP tools called:
    1. optimise_portfolio (portfolio_optimiser server)

Design decisions:

    call_tool() convenience wrapper used here:
        optimise makes exactly one tool call to one server.
        No need to hold the ClientSession open for multiple calls.
        call_tool() opens, calls, closes in one step.

    risk_free_rate passed from risk_metrics:
        The same RFR used for Sharpe computation in compute_risk is
        passed to the optimiser for consistency. Both nodes must use
        the same RFR — otherwise the Sharpe ratios are not comparable.

    n_frontier_points hardcoded to 50:
        50 points gives a smooth efficient frontier curve for the
        Gradio UI visualisation. Sufficient for v1.

    solver hardcoded to "convex_qp":
        v1 uses SLSQP via scipy. Upgraded to differential_evolution in v2.

    Guard against None upstream fields:
        If market_data or risk_metrics is None, node skips and returns
        optimisation_result=None. The simulate node handles None
        optimisation_result by skipping monte_carlo_optimal and
        garch_sim_optimal (sets them to None in SimulationResult).

    FrontierPoint deserialisation:
        The efficient_frontier field is a list of dicts from the server.
        Each dict is deserialised into a FrontierPoint model.
        Malformed points are skipped with a warning — a partial frontier
        is better than a crash.
"""

from __future__ import annotations

import logging

from orchestrator.clients.mcp_client_factory import call_tool
from orchestrator.state import (
    AgentState,
    FrontierPoint,
    OptimisationResult,
)

logger = logging.getLogger(__name__)

# Hardcoded in v1
_N_FRONTIER_POINTS = 50
_SOLVER = "convex_qp"


async def optimise(state: AgentState) -> dict:
    """
    LangGraph node — optimise.

    Calls optimise_portfolio tool on the Portfolio Optimiser Server.
    Builds OptimisationResult from the response.

    Args:
        state: Current AgentState. Reads: market_data, risk_metrics.

    Returns:
        dict with keys: optimisation_result, execution_trace, errors.
    """
    # ── Guard: upstream failures ──────────────────────────────────────────────
    if state.market_data is None or state.risk_metrics is None:
        missing = []
        if state.market_data is None:
            missing.append("market_data")
        if state.risk_metrics is None:
            missing.append("risk_metrics")
        logger.error("optimise: skipping — %s is None", ", ".join(missing))
        return {
            "optimisation_result": None,
            "execution_trace":     state.execution_trace + ["optimise:skipped"],
            "errors":              state.errors + [f"optimise: {', '.join(missing)} is None"],
        }

    logger.info("optimise: starting — %d symbols", len(state.symbols))

    try:
        data = await call_tool(
            server_name="portfolio_optimiser",
            tool_name="optimise_portfolio",
            arguments={
                "log_returns":       state.market_data.log_returns,
                "risk_free_rate":    state.risk_metrics.risk_free_rate,
                "n_frontier_points": _N_FRONTIER_POINTS,
                "solver":            _SOLVER,
            },
        )
        logger.info(
            "optimise: ok — sharpe=%.4f return=%.4f vol=%.4f solver=%s",
            data["sharpe_ratio"],
            data["expected_return"],
            data["portfolio_volatility"],
            data["solver_used"],
        )

    except Exception as exc:
        logger.error("optimise: failed — %s", exc)
        return {
            "optimisation_result": None,
            "execution_trace":     state.execution_trace + ["optimise:error"],
            "errors":              state.errors + [f"optimise: {exc}"],
        }

    # ── Build OptimisationResult ──────────────────────────────────────────────
    frontier: list[FrontierPoint] = []
    for point in data.get("efficient_frontier", []):
        try:
            frontier.append(FrontierPoint(**point))
        except Exception as exc:
            logger.warning("optimise: could not parse FrontierPoint — %s", exc)

    optimisation_result = OptimisationResult(
        optimal_weights=data["optimal_weights"],
        max_sharpe_weights=data["max_sharpe_weights"],
        expected_return=data["expected_return"],
        portfolio_volatility=data["portfolio_volatility"],
        sharpe_ratio=data["sharpe_ratio"],
        cml_slope=data["cml_slope"],
        efficient_frontier=frontier,
        solver_used=data["solver_used"],
    )

    return {
        "optimisation_result": optimisation_result,
        "execution_trace":     state.execution_trace + ["optimise:ok"],
        "errors":              state.errors,
    }