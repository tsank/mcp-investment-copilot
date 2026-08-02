"""
orchestrator/nodes/check_compliance.py

Node 6 of 7 — check_compliance

Responsibility:
    Call the Compliance Server to check the CURRENT portfolio weights
    against the retail_conservative_v1.0 ruleset.

Reads from AgentState:
    - portfolio          (Portfolio)          — holdings (current weights)
    - market_data        (MarketDataResult)   — fundamentals (for sector_map)
    - risk_metrics       (RiskMetricsResult)  — var_95, cvar_95
    - simulation_result  (SimulationResult)   — cvar_95 override if available

Writes to AgentState:
    - compliance_result (ComplianceResult) — passed, violations, warnings,
                                             rules_version, rules_profile

MCP tools called:
    1. check_compliance (compliance server)

Design decisions:

    rules_profile hardcoded to "retail_conservative" (v1 decision):
        Locked in architecture alignment — not a variable, not parsed
        from the query. Promoted to AgentState field in v2 when the
        Gradio UI adds a profile selector dropdown.

    rules_version hardcoded to "v1.0":
        Single ruleset version in v1. Promoted alongside rules_profile in v2.

    CVaR source priority:
        The compliance tool needs a cvar_95 value for the CVAR_THRESHOLD rule.
        Priority order (best available wins):
            1. simulation_result.garch_sim.cvar_95  — GARCH-conditional,
               reflects current vol regime (most accurate)
            2. simulation_result.monte_carlo.cvar_95 — static IID distribution
            3. risk_metrics.cvar_95                 — empirical, backward-looking
        This ensures the compliance check uses the most forward-looking
        risk estimate available, consistent with CVaR being a gating metric.

    Compliance always checks CURRENT weights, not optimal:
        The compliance tool receives state.portfolio.holdings, never
        optimisation_result.optimal_weights. The compliance question is:
        "Is the user's ACTUAL position compliant?" — not a hypothetical.

    sector_map construction:
        Built from market_data.fundamentals — symbol → sector string.
        Symbols with missing fundamentals fall into "Unknown" sector grouping
        inside the compliance tool (already handled there).

    var_95 passed for completeness:
        The retail_conservative ruleset gates on CVaR, not VaR.
        var_95 is passed anyway — it is part of the tool's input schema
        and may be used by future ruleset versions.
"""

from __future__ import annotations

import logging

from servers.compliance.tools.check_compliance import check_compliance as _check_compliance_tool
from orchestrator.state import (
    AgentState,
    ComplianceResult,
    ComplianceWarning,
    Violation,
)

logger = logging.getLogger(__name__)

# Hardcoded in v1 — promoted to AgentState fields in v2
_RULES_PROFILE = "retail_conservative"
_RULES_VERSION = "v1.0"


async def check_compliance(state: AgentState) -> dict:
    """
    LangGraph node — check_compliance.

    Selects the best available CVaR estimate, builds sector_map from
    fundamentals, and calls the check_compliance tool.

    Args:
        state: Current AgentState.
               Reads: portfolio, market_data, risk_metrics, simulation_result.

    Returns:
        dict with keys: compliance_result, execution_trace, errors.
    """
    # ── Guard: risk_metrics is the minimum requirement ────────────────────────
    # We need at least var_95 and cvar_95 from risk_metrics.
    # market_data is needed for sector_map but failure is non-fatal there.
    if state.risk_metrics is None:
        logger.error("check_compliance: skipping — risk_metrics is None")
        return {
            "compliance_result": None,
            "execution_trace":   state.execution_trace + ["check_compliance:skipped"],
            "errors":            state.errors + ["check_compliance: risk_metrics is None"],
        }

    # ── Select best CVaR estimate ─────────────────────────────────────────────
    cvar_95, cvar_source = _select_cvar(state)
    var_95 = state.risk_metrics.var_95
    logger.info(
        "check_compliance: using cvar_95=%.4f (source=%s) var_95=%.4f",
        cvar_95, cvar_source, var_95,
    )

    # ── Build sector_map ──────────────────────────────────────────────────────
    sector_map: dict[str, str] = {}
    if state.market_data is not None:
        for symbol, fd in state.market_data.fundamentals.items():
            sector_map[symbol] = fd.sector
    else:
        logger.warning("check_compliance: market_data is None — sector_map will be empty")

    logger.info(
        "check_compliance: sector_map has %d entries, %d symbols in portfolio",
        len(sector_map), len(state.portfolio.holdings),
    )

    # ── Call compliance tool ──────────────────────────────────────────────────
    try:
        # Direct function call (v2 - Option B). No subprocess, no JSON round
        # trip, no asyncio.to_thread() - rule evaluation against a YAML 
        # ruleset is light, not CPU-heavy work worthy threading.
        data = _check_compliance_tool(
            weights=state.portfolio.holdings,
            sector_map=sector_map,
            var_95=var_95,
            cvar_95=cvar_95,
            rules_profile=_RULES_PROFILE,
            rules_version=_RULES_VERSION,
        )
        logger.info(
            "check_compliance: ok — passed=%s violations=%d warnings=%d",
            data["passed"], len(data["violations"]), len(data["warnings"]),
        )

    except Exception as exc:
        logger.error("check_compliance: failed — %s", exc)
        return {
            "compliance_result": None,
            "execution_trace":   state.execution_trace + ["check_compliance:error"],
            "errors":            state.errors + [f"check_compliance: {exc}"],
        }

    # ── Build ComplianceResult ────────────────────────────────────────────────
    violations = [Violation(**v) for v in data["violations"]]
    warnings   = [ComplianceWarning(**w) for w in data["warnings"]]

    compliance_result = ComplianceResult(
        passed=data["passed"],
        violations=violations,
        warnings=warnings,
        rules_version=data["rules_version"],
        rules_profile=data["rules_profile"],
        cvar_95=cvar_95,
        cvar_source=cvar_source,
    )

    return {
        "compliance_result": compliance_result,
        "execution_trace":   state.execution_trace + ["check_compliance:ok"],
        "errors":            state.errors,
    }


# ── Helper ────────────────────────────────────────────────────────────────────

def _select_cvar(state: AgentState) -> tuple[float, str]:
    """
    Select the best available CVaR estimate from AgentState.

    Priority:
        1. simulation_result.garch_sim.cvar_95   — GARCH-conditional (best)
        2. simulation_result.monte_carlo.cvar_95  — IID static
        3. risk_metrics.cvar_95                  — empirical (fallback)

    Returns:
        tuple[float, str] — (cvar_95_value, source_label)
    """
    if (
        state.simulation_result is not None
        and state.simulation_result.garch_sim is not None
    ):
        return state.simulation_result.garch_sim.cvar_95, "garch_sim"

    if (
        state.simulation_result is not None
        and state.simulation_result.monte_carlo is not None
    ):
        return state.simulation_result.monte_carlo.cvar_95, "monte_carlo"

    return state.risk_metrics.cvar_95, "risk_metrics"