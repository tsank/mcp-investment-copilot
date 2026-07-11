"""
orchestrator/nodes/simulate.py

Node 5 of 7 — simulate

Responsibility:
    Call the Scenario Simulation Server to run four simulation variants:
        1. monte_carlo         — IID static, CURRENT weights
        2. monte_carlo_optimal — IID static, OPTIMAL weights (if available)
        3. garch_sim           — GARCH-conditional, CURRENT weights
        4. garch_sim_optimal   — GARCH-conditional, OPTIMAL weights (if available)

    The current vs optimal comparison answers the core investment question:
    "How much tail risk does rebalancing achieve?"

Reads from AgentState:
    - market_data          (MarketDataResult)    — log_returns
    - portfolio            (Portfolio)           — holdings (current weights)
    - garch_result         (GARCHResult)         — garch_params, current_vols
    - optimisation_result  (OptimisationResult)  — optimal_weights (may be None)

Writes to AgentState:
    - simulation_result (SimulationResult) — four SimulationOutput fields
                                             + regime_warning flag

MCP tools called (both against "scenario_simulation" server):
    1. run_monte_carlo      — called up to twice (current + optimal weights)
    2. run_garch_simulation — called up to twice (current + optimal weights)

Design decisions:

    Optimal weight calls are conditional:
        If optimisation_result is None (RISK or SIMULATION analysis type,
        or if optimise node failed), monte_carlo_optimal and garch_sim_optimal
        are set to None. SimulationResult documents this as expected behaviour.

    GARCH handoff pattern:
        garch_result.garch_params and garch_result.current_vols are passed
        directly to run_garch_simulation. The Simulator does not re-fit —
        it draws stochastic paths from these already-fitted parameters.
        This ensures consistency between Risk Engine metrics and Simulator.

    regime_warning:
        Set to True if CVaR from garch_sim diverges materially from
        monte_carlo CVaR for current weights. Threshold: 20% relative
        difference. Signals the current volatility regime is elevated
        relative to the historical average embedded in monte_carlo.

    Simulation parameters hardcoded in v1:
        horizon_days=252    — 1 trading year
        n_simulations=10000 — sufficient for stable CVaR at 95%
        distribution="student_t" — fat tails appropriate for NSE equities
        random_seed=42      — reproducible results for debugging

    Non-fatal individual failures:
        If one of the four calls fails, the others proceed.
        SimulationResult fields are set to None for the failed run.
        regime_warning is only computed if both garch_sim and monte_carlo
        are available.
"""

from __future__ import annotations

import logging
import asyncio

from servers.scenario_simulation.tools.monte_carlo import run_monte_carlo
from servers.scenario_simulation.tools.garch_simulation import run_garch_simulation

from orchestrator.state import (
    AgentState,
    PercentileDistribution,
    SimulationOutput,
    SimulationResult,
)

logger = logging.getLogger(__name__)

# Hardcoded in v1
_HORIZON_DAYS   = 252
_N_SIMULATIONS  = 1_000
_DISTRIBUTION   = "student_t"
_RANDOM_SEED    = 42
_REGIME_WARNING_THRESHOLD = 0.20   # 20% relative CVaR divergence


async def simulate(state: AgentState) -> dict:
    """
    LangGraph node — simulate.

    Makes up to four direct function calls: run_monte_carlo and
    run_garch_simulation, each called for current weights and
    (if available) optimal weights.

    Args:
        state: Current AgentState.
               Reads: market_data, portfolio, garch_result, optimisation_result.

    Returns:
        dict with keys: simulation_result, execution_trace, errors.
    """
    # ── Guard: upstream failures ──────────────────────────────────────────────
    if state.market_data is None or state.garch_result is None:
        missing = []
        if state.market_data is None:
            missing.append("market_data")
        if state.garch_result is None:
            missing.append("garch_result")
        logger.error("simulate: skipping — %s is None", ", ".join(missing))
        return {
            "simulation_result": None,
            "execution_trace":   state.execution_trace + ["simulate:skipped"],
            "errors":            state.errors + [f"simulate: {', '.join(missing)} is None"],
        }

    logger.info("simulate: starting — %d symbols", len(state.symbols))

    # ── Determine weight sets ─────────────────────────────────────────────────
    current_weights = state.portfolio.holdings
    optimal_weights = (
        state.optimisation_result.optimal_weights
        if state.optimisation_result is not None
        else None
    )

    # ── Serialise GARCH handoff fields ────────────────────────────────────────
    garch_params_dict = {
        symbol: params.dict()
        for symbol, params in state.garch_result.garch_params.items()
    }
    current_vols_dict = state.garch_result.current_vols

    # ── Run all four simulations ──────────────────────────────────────────────
    mc_current:  SimulationOutput | None = None
    mc_optimal:  SimulationOutput | None = None
    gc_current:  SimulationOutput | None = None
    gc_optimal:  SimulationOutput | None = None
    errors_this_node: list[str] = []

    # ── 1. Monte Carlo — current weights ────────────────────────────────────────
    # Direct function call (v2 — Option B). No subprocess, no JSON round-trip.
    # Wrapped in asyncio.to_thread() — genuinely CPU-heavy (10,000 simulated
    # paths). Each of the four calls is individually try/excepted, same as
    # before — a failure in one shouldn't prevent the other three from
    # running, matching the "non-fatal individual failures" design decision.
    try:
        logger.info("simulate: run_monte_carlo (current weights)")
        mc_current = _parse_simulation_output(
            await asyncio.to_thread(
                run_monte_carlo,
                log_returns=state.market_data.log_returns,
                weights=current_weights,
                horizon_days=_HORIZON_DAYS,
                n_simulations=_N_SIMULATIONS,
                distribution=_DISTRIBUTION,
                random_seed=_RANDOM_SEED,
            )
        )
        logger.info("simulate: mc_current ok — cvar_95=%.4f", mc_current.cvar_95)
    except Exception as exc:
        logger.warning("simulate: run_monte_carlo (current) failed — %s", exc)
        errors_this_node.append(f"simulate.mc_current: {exc}")

    # ── 2. Monte Carlo — optimal weights ────────────────────────────────────────
    if optimal_weights is not None:
        try:
            logger.info("simulate: run_monte_carlo (optimal weights)")
            mc_optimal = _parse_simulation_output(
                await asyncio.to_thread(
                    run_monte_carlo,
                    log_returns=state.market_data.log_returns,
                    weights=optimal_weights,
                    horizon_days=_HORIZON_DAYS,
                    n_simulations=_N_SIMULATIONS,
                    distribution=_DISTRIBUTION,
                    random_seed=_RANDOM_SEED,
                )
            )
            logger.info("simulate: mc_optimal ok — cvar_95=%.4f", mc_optimal.cvar_95)
        except Exception as exc:
            logger.warning("simulate: run_monte_carlo (optimal) failed — %s", exc)
            errors_this_node.append(f"simulate.mc_optimal: {exc}")
    else:
        logger.info("simulate: skipping mc_optimal — no optimisation_result")

    # ── 3. GARCH simulation — current weights ───────────────────────────────────
    try:
        logger.info("simulate: run_garch_simulation (current weights)")
        gc_current = _parse_simulation_output(
            await asyncio.to_thread(
                run_garch_simulation,
                log_returns=state.market_data.log_returns,
                weights=current_weights,
                garch_params=garch_params_dict,
                current_vols=current_vols_dict,
                horizon_days=_HORIZON_DAYS,
                n_simulations=_N_SIMULATIONS,
                random_seed=_RANDOM_SEED,
            )
        )
        logger.info("simulate: gc_current ok — cvar_95=%.4f", gc_current.cvar_95)
    except Exception as exc:
        logger.warning("simulate: run_garch_simulation (current) failed — %s", exc)
        errors_this_node.append(f"simulate.gc_current: {exc}")

    # ── 4. GARCH simulation — optimal weights ───────────────────────────────────
    if optimal_weights is not None:
        try:
            logger.info("simulate: run_garch_simulation (optimal weights)")
            gc_optimal = _parse_simulation_output(
                await asyncio.to_thread(
                    run_garch_simulation,
                    log_returns=state.market_data.log_returns,
                    weights=optimal_weights,
                    garch_params=garch_params_dict,
                    current_vols=current_vols_dict,
                    horizon_days=_HORIZON_DAYS,
                    n_simulations=_N_SIMULATIONS,
                    random_seed=_RANDOM_SEED,
                )
            )
            logger.info("simulate: gc_optimal ok — cvar_95=%.4f", gc_optimal.cvar_95)
        except Exception as exc:
            logger.warning("simulate: run_garch_simulation (optimal) failed — %s", exc)
            errors_this_node.append(f"simulate.gc_optimal: {exc}")
    else:
        logger.info("simulate: skipping gc_optimal — no optimisation_result")

    # ── Compute regime_warning ────────────────────────────────────────────────
    regime_warning = False
    if mc_current is not None and gc_current is not None:
        relative_diff = abs(gc_current.cvar_95 - mc_current.cvar_95) / max(mc_current.cvar_95, 1e-8)
        regime_warning = relative_diff > _REGIME_WARNING_THRESHOLD
        if regime_warning:
            logger.warning(
                "simulate: regime_warning=True — "
                "mc_cvar_95=%.4f gc_cvar_95=%.4f relative_diff=%.2f",
                mc_current.cvar_95, gc_current.cvar_95, relative_diff,
            )

    # ── Build SimulationResult ────────────────────────────────────────────────
    simulation_result = SimulationResult(
        monte_carlo=mc_current,
        monte_carlo_optimal=mc_optimal,
        garch_sim=gc_current,
        garch_sim_optimal=gc_optimal,
        regime_warning=regime_warning,
    )

    status = "ok" if not errors_this_node else "partial"
    return {
        "simulation_result": simulation_result,
        "execution_trace":   state.execution_trace + [f"simulate:{status}"],
        "errors":            state.errors + errors_this_node,
    }


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_simulation_output(data: dict) -> SimulationOutput:
    """
    Deserialise a raw tool response dict into a SimulationOutput model.

    Both run_monte_carlo and run_garch_simulation return the same schema,
    so this helper is shared between all four simulation calls.
    """
    return SimulationOutput(
        cvar_95=data["cvar_95"],
        cvar_99=data["cvar_99"],
        var_95=data["var_95"],
        var_99=data["var_99"],
        percentiles=PercentileDistribution(**data["percentiles"]),
        n_simulations=data["n_simulations"],
        distribution_used=data["distribution_used"],
        fitted_nu=data.get("fitted_nu"),
    )