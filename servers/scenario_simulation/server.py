"""
servers/scenario_simulation/server.py

MCP Server entry point for the Scenario Simulation Server.

Responsibilities:
    - Initialise the MCP Server instance
    - Register tool schemas via @server.list_tools()
    - Route tool calls via @server.call_tool()
    - Start stdio transport for local development

This file is intentionally thin — all computation logic lives in tools/.
This file handles protocol wiring only.

Tools exposed:
    run_monte_carlo      — static distribution Monte Carlo (IID draws)
                           fits its own distribution from log_returns
                           σ constant across all simulation steps
                           "The future will look statistically like the past on average"

    run_garch_simulation — GARCH-conditional Monte Carlo (serially dependent)
                           consumes pre-fitted GARCH params from Risk Engine
                           σ_t evolves via GARCH recursion at every step
                           "The future will evolve from where volatility is right now"

Both tools:
    - Accept explicit weights (current OR optimal — orchestrator decides)
    - Return CVaR as the primary decision metric
    - Return VaR for context only
    - Support random_seed for deterministic testing

Transport:
    Local development : stdio (stdin/stdout pipes)
    AWS production    : HTTP+SSE (transport changes, protocol identical)

Usage:
    python servers/scenario_simulation/server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    ServerCapabilities,
    TextContent,
    Tool,
)

from tools.monte_carlo import run_monte_carlo
from tools.garch_simulation import run_garch_simulation

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-scenario-simulation")

# ── Server instance ───────────────────────────────────────────────────────────
server = Server("mcp-scenario-simulation")


# ── Tool schemas ──────────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="run_monte_carlo",
        description=(
            "Monte Carlo simulation drawing N paths from a static fitted distribution. "
            "Fits the distribution (Student-t, Gaussian, or historical bootstrap) "
            "directly from the historical log-return series — no pre-fitted parameters needed. "
            "Returns are assumed IID — no temporal structure in volatility (σ is constant). "
            "Generates N×H simulated paths using antithetic variates for variance reduction. "
            "Terminal value = cumulative portfolio return over the full horizon. "
            "CVaR (Expected Shortfall) is the primary output — the mean loss in the "
            "worst (1-α) fraction of simulated paths. "
            "VaR is reported for context only. "
            "Weights are explicit input — call twice for current vs optimal comparison: "
            "    current weights  → stored in SimulationResult.monte_carlo "
            "    optimal weights  → stored in SimulationResult.monte_carlo_optimal "
            "Use as the baseline forward-looking risk estimate. "
            "For dynamic volatility regime-aware simulation, use run_garch_simulation. "
            "Call after compute_risk_metrics and before check_compliance."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "log_returns": {
                    "type": "object",
                    "description": (
                        "dict[symbol, list[float]] — pre-computed log-returns "
                        "from get_price_history output. "
                        "Used both for distribution fitting and historical bootstrap."
                    ),
                },
                "weights": {
                    "type": "object",
                    "description": (
                        "dict[symbol, float] — portfolio weights, must sum to 1. "
                        "Pass Portfolio.holdings for current portfolio risk. "
                        "Pass optimisation_result.optimal_weights for optimal portfolio risk."
                    ),
                },
                "horizon_days": {
                    "type": "integer",
                    "description": (
                        "Simulation horizon in trading days. "
                        "e.g. 21 (1 month), 63 (1 quarter), 252 (1 year). "
                        "Default 252."
                    ),
                    "default": 252,
                },
                "n_simulations": {
                    "type": "integer",
                    "description": (
                        "Number of simulation paths. "
                        "Default 10000. Must be even (antithetic variates). "
                        "Higher values reduce estimation error in CVaR."
                    ),
                    "default": 10000,
                },
                "distribution": {
                    "type": "string",
                    "enum": ["student_t", "gaussian", "historical_bootstrap"],
                    "description": (
                        "'student_t': fit Student-t with empirical ν — default. "
                        "Captures fat tails present in NSE equity returns. "
                        "'gaussian': fit normal distribution — baseline comparison. "
                        "Underestimates tail risk for equity returns. "
                        "'historical_bootstrap': resample empirical returns directly. "
                        "No distributional assumption. Cannot generate scenarios "
                        "worse than historical worst."
                    ),
                    "default": "student_t",
                },
                "random_seed": {
                    "type": "integer",
                    "description": (
                        "Random seed for deterministic simulation. "
                        "Set for testing and reproducibility. "
                        "Omit (null) for stochastic production behaviour."
                    ),
                },
            },
            "required": ["log_returns", "weights"],
        },
    ),
    Tool(
        name="run_garch_simulation",
        description=(
            "Monte Carlo simulation drawing N paths from a GARCH-conditional process. "
            "Consumes pre-fitted GARCH parameters from compute_garch_forecast "
            "in the Risk Engine — does NOT re-fit the model. "
            "σ_t evolves at every simulation step via the GARCH recursion: "
            "    σ²_{t+1} = ω + α·ε²_t + β·σ²_t "
            "Captures volatility clustering — large shocks raise future volatility. "
            "Preserves cross-asset correlation via Cholesky decomposition. "
            "Starting volatility σ_T reflects the CURRENT market regime — "
            "elevated if recent market has been turbulent, suppressed if calm. "
            "This is the regime-aware simulation: "
            "    'The future will evolve from where volatility is right now.' "
            "Compare CVaR from run_garch_simulation vs run_monte_carlo to detect "
            "regime elevation — large divergence signals current vol is abnormal. "
            "Weights are explicit input — call twice for current vs optimal comparison: "
            "    current weights  → stored in SimulationResult.garch_sim "
            "    optimal weights  → stored in SimulationResult.garch_sim_optimal "
            "Requires compute_garch_forecast to have run upstream. "
            "Call after compute_garch_forecast and before check_compliance."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "log_returns": {
                    "type": "object",
                    "description": (
                        "dict[symbol, list[float]] — pre-computed log-returns "
                        "from get_price_history output. "
                        "Used for Cholesky correlation computation only — "
                        "NOT for parameter fitting (that was done by Risk Engine)."
                    ),
                },
                "weights": {
                    "type": "object",
                    "description": (
                        "dict[symbol, float] — portfolio weights, must sum to 1. "
                        "Pass Portfolio.holdings for current portfolio risk. "
                        "Pass optimisation_result.optimal_weights for optimal portfolio risk."
                    ),
                },
                "garch_params": {
                    "type": "object",
                    "description": (
                        "dict[symbol, {omega, alpha, beta, nu}] — fitted GARCH parameters "
                        "from GARCHResult.garch_params in AgentState. "
                        "Populated by compute_garch_forecast in the Risk Engine."
                    ),
                },
                "current_vols": {
                    "type": "object",
                    "description": (
                        "dict[symbol, float] — current conditional volatility σ_T "
                        "per asset, annualised. "
                        "From GARCHResult.current_vols in AgentState. "
                        "This is the starting point for the GARCH recursion — "
                        "reflects the current volatility regime."
                    ),
                },
                "horizon_days": {
                    "type": "integer",
                    "description": (
                        "Simulation horizon in trading days. "
                        "Default 252."
                    ),
                    "default": 252,
                },
                "n_simulations": {
                    "type": "integer",
                    "description": (
                        "Number of simulation paths. "
                        "Default 10000."
                    ),
                    "default": 10000,
                },
                "random_seed": {
                    "type": "integer",
                    "description": (
                        "Random seed for deterministic simulation. "
                        "Set for testing. Omit for production."
                    ),
                },
            },
            "required": ["log_returns", "weights", "garch_params", "current_vols"],
        },
    ),
]


# ── Tool registration ─────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    Called automatically by the mcp SDK when the orchestrator sends
    a tools/list request at startup.
    """
    logger.info("tools/list requested — returning %d tools", len(TOOLS))
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Called automatically by the mcp SDK when the orchestrator sends
    a tools/call request.

    Routes to the correct tool function based on name.
    Results are serialised to JSON and wrapped in TextContent.
    """
    logger.info("tools/call: %s arguments=%s", name, list(arguments.keys()))

    if name == "run_monte_carlo":
        result = run_monte_carlo(
            log_returns=arguments["log_returns"],
            weights=arguments["weights"],
            horizon_days=arguments.get("horizon_days", 252),
            n_simulations=arguments.get("n_simulations", 10000),
            distribution=arguments.get("distribution", "student_t"),
            random_seed=arguments.get("random_seed", None),
        )

    elif name == "run_garch_simulation":
        result = run_garch_simulation(
            log_returns=arguments["log_returns"],
            weights=arguments["weights"],
            garch_params=arguments["garch_params"],
            current_vols=arguments["current_vols"],
            horizon_days=arguments.get("horizon_days", 252),
            n_simulations=arguments.get("n_simulations", 10000),
            random_seed=arguments.get("random_seed", None),
        )

    else:
        raise ValueError(
            f"Unknown tool: '{name}'. "
            f"Available tools: {[t.name for t in TOOLS]}"
        )

    logger.info("tools/call: %s completed successfully", name)

    return [TextContent(type="text", text=json.dumps(result))]


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    """
    Start the MCP server with stdio transport.
    """
    logger.info("Starting mcp-scenario-simulation server (stdio transport)")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-scenario-simulation",
                server_version="1.0.0",
                capabilities=ServerCapabilities(tools={}),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())