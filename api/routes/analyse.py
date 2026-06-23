"""
api/routes/analyse.py

Route handler for POST /api/v1/analyse.

Responsibilities:
    1. Accept AnalyseRequest (validated by FastAPI/Pydantic)
    2. Build AgentState from request
    3. Invoke investment_graph (async)
    4. Extract result fields from final AgentState
    5. Return AnalyseResponse

This file is intentionally thin — all business logic lives in the
orchestrator. The route handler is pure translation: request → state → response.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from orchestrator.graph import investment_graph
from orchestrator.state import AgentState, Portfolio

from api.schemas.request import AnalyseRequest
from api.schemas.response import (
    AnalyseResponse,
    ComplianceResponse,
    OptimisationResponse,
    PercentileResponse,
    RiskMetricsResponse,
    SimulationOutputResponse,
    SimulationResponse,
    ViolationResponse,
    WarningResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/analyse",
    response_model=AnalyseResponse,
    summary="Analyse a portfolio",
    description=(
        "Run a full investment analysis pipeline: market data → risk → "
        "optimisation → simulation → compliance → recommendation. "
        "Returns a structured response with recommendation prose and "
        "all computed quantitative results."
    ),
)
async def analyse(request: AnalyseRequest) -> AnalyseResponse:
    """
    POST /api/v1/analyse

    Invokes the LangGraph investment analysis pipeline and returns
    the full structured result.
    """
    logger.info(
        "analyse: query=%r symbols=%s",
        request.query[:60],
        list(request.portfolio.holdings.keys()),
    )

    # ── Build AgentState ──────────────────────────────────────────────────────
    initial_state = AgentState(
        query=request.query,
        portfolio=Portfolio(
            holdings=request.portfolio.holdings,
            total_value=request.portfolio.total_value,
            benchmark=request.portfolio.benchmark,
        ),
    )

    # ── Invoke graph ──────────────────────────────────────────────────────────
    try:
        final_state: AgentState = await investment_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("analyse: graph invocation failed — %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline failed: {exc}",
        )

    logger.info(
        "analyse: completed — trace=%s errors=%s",
        final_state.execution_trace,
        final_state.errors,
    )

    # ── Build response ────────────────────────────────────────────────────────
    return AnalyseResponse(
        recommendation=final_state.final_recommendation or "Analysis completed — no recommendation generated.",
        compliance=_build_compliance(final_state),
        risk_metrics=_build_risk_metrics(final_state),
        simulation=_build_simulation(final_state),
        optimisation=_build_optimisation(final_state),
        analysis_type=final_state.analysis_type.value,
        execution_trace=final_state.execution_trace,
        errors=final_state.errors,
    )


# ── Response builders ─────────────────────────────────────────────────────────

def _build_compliance(state: AgentState) -> ComplianceResponse | None:
    if state.compliance_result is None:
        return None
    cr = state.compliance_result
    return ComplianceResponse(
        passed=cr.passed,
        violations=[ViolationResponse(**v.dict()) for v in cr.violations],
        warnings=[WarningResponse(**w.dict()) for w in cr.warnings],
        rules_profile=cr.rules_profile,
        rules_version=cr.rules_version,
    )


def _build_risk_metrics(state: AgentState) -> RiskMetricsResponse | None:
    if state.risk_metrics is None:
        return None
    rm = state.risk_metrics
    return RiskMetricsResponse(
        var_95=rm.var_95,
        var_99=rm.var_99,
        cvar_95=rm.cvar_95,
        cvar_99=rm.cvar_99,
        sharpe_ratio=rm.sharpe_ratio,
        max_drawdown=rm.max_drawdown,
        volatility=rm.volatility,
        portfolio_return=rm.portfolio_return,
        risk_free_rate=rm.risk_free_rate,
        computation_window=rm.computation_window,
    )


def _build_simulation(state: AgentState) -> SimulationResponse | None:
    if state.simulation_result is None:
        return None
    sr = state.simulation_result

    def _sim_out(s) -> SimulationOutputResponse | None:
        if s is None:
            return None
        return SimulationOutputResponse(
            cvar_95=s.cvar_95,
            cvar_99=s.cvar_99,
            var_95=s.var_95,
            var_99=s.var_99,
            percentiles=PercentileResponse(**s.percentiles.dict()),
            n_simulations=s.n_simulations,
            distribution_used=s.distribution_used,
            fitted_nu=s.fitted_nu,
        )

    return SimulationResponse(
        monte_carlo=_sim_out(sr.monte_carlo),
        monte_carlo_optimal=_sim_out(sr.monte_carlo_optimal),
        garch_sim=_sim_out(sr.garch_sim),
        garch_sim_optimal=_sim_out(sr.garch_sim_optimal),
        regime_warning=sr.regime_warning,
    )


def _build_optimisation(state: AgentState) -> OptimisationResponse | None:
    if state.optimisation_result is None:
        return None
    opt = state.optimisation_result
    return OptimisationResponse(
        optimal_weights=opt.optimal_weights,
        expected_return=opt.expected_return,
        portfolio_volatility=opt.portfolio_volatility,
        sharpe_ratio=opt.sharpe_ratio,
        cml_slope=opt.cml_slope,
        solver_used=opt.solver_used,
    )