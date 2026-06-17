"""
servers/portfolio_optimiser/server.py

MCP Server entry point for the Portfolio Optimiser Server.

Responsibilities:
    - Initialise the MCP Server instance
    - Register tool schemas via @server.list_tools()
    - Route tool calls via @server.call_tool()
    - Start stdio transport for local development

This file is intentionally thin — all computation logic lives in tools/.
This file handles protocol wiring only.

Tools exposed:
    optimise_portfolio — Markowitz mean-variance optimisation
                         Efficient Frontier (Scanning method) +
                         Maximum Sharpe portfolio (tangency point)

Solver versioning:
    V1: convex_qp              — scipy SLSQP (current)
    V2: differential_evolution — scipy DE for non-smooth constraints
    V3: nsga2                  — pymoo NSGA-II for true multi-objective

Transport:
    Local development : stdio (stdin/stdout pipes)
    AWS production    : HTTP+SSE (transport changes, protocol identical)

Usage:
    python servers/portfolio_optimiser/server.py
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

from tools.optimise import optimise_portfolio

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-portfolio-optimiser")

# ── Server instance ───────────────────────────────────────────────────────────
server = Server("mcp-portfolio-optimiser")


# ── Tool schemas ──────────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="optimise_portfolio",
        description=(
            "Run Markowitz mean-variance optimisation on a fixed set of assets "
            "to identify the Efficient Frontier (Pareto frontier of return vs risk) "
            "and the Maximum Sharpe portfolio (tangency point on the Capital Market Line). "
            "Two sequential steps: "
            "(1) Scanning Method — runs N minimum-variance solves at fixed target "
            "return levels to map the complete Efficient Frontier. Every point on "
            "the frontier is Pareto-optimal: no other portfolio achieves the same "
            "return with lower risk, or the same risk with higher return. "
            "(2) Maximum Sharpe Solve — finds the tangency point where a line from "
            "the risk-free rate touches the frontier. This is the portfolio with "
            "the highest risk-adjusted return (return per unit of volatility). "
            "The asset universe is fixed — optimisation finds the best weight "
            "combination across the given symbols, not which symbols to hold. "
            "V1 solver: convex_qp (scipy SLSQP). "
            "V2 solver: differential_evolution (non-smooth constraints). "
            "V3 solver: nsga2 (true multi-objective Pareto via pymoo). "
            "Call this after compute_risk_metrics and before check_compliance."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "log_returns": {
                    "type": "object",
                    "description": (
                        "dict[symbol, list[float]] — pre-computed log-returns "
                        "from get_price_history output. "
                        "Minimum 2 assets required. "
                        "All series must have identical length (date-aligned)."
                    ),
                },
                "risk_free_rate": {
                    "type": "number",
                    "description": (
                        "Annualised risk-free rate as a decimal. "
                        "Default 0.065 (6.5% — RBI repo rate proxy for INR). "
                        "Used to identify the Maximum Sharpe portfolio (tangency point). "
                        "The Efficient Frontier shape does not depend on this value — "
                        "only the tangency point location changes with rfr."
                    ),
                    "default": 0.065,
                },
                "n_frontier_points": {
                    "type": "integer",
                    "description": (
                        "Number of points to compute on the Efficient Frontier. "
                        "Default 50. Higher values give smoother frontier curve "
                        "at the cost of computation time. "
                        "Minimum 2."
                    ),
                    "default": 50,
                },
                "constraints": {
                    "type": "object",
                    "description": (
                        "Portfolio weight constraints. All fields optional. "
                        "min_weight: minimum weight per asset, default 0.0 "
                        "(0.0 = assets can be excluded, no short selling). "
                        "max_weight: maximum weight per asset, default 1.0 "
                        "(e.g. 0.30 = no single asset can exceed 30%). "
                        "target_return: if set, runs a single solve at this "
                        "return level instead of full frontier scan. "
                        "sector_caps: dict[sector, float] — v2 feature, ignored in v1."
                    ),
                    "properties": {
                        "min_weight":    {"type": "number", "default": 0.0},
                        "max_weight":    {"type": "number", "default": 1.0},
                        "target_return": {"type": "number"},
                        "sector_caps":   {"type": "object"},
                    },
                },
                "solver": {
                    "type": "string",
                    "enum": ["convex_qp", "differential_evolution", "nsga2"],
                    "description": (
                        "Optimisation solver. "
                        "'convex_qp': scipy SLSQP — default, v1 only. "
                        "'differential_evolution': scipy DE — v2, non-smooth constraints. "
                        "'nsga2': pymoo NSGA-II — v3, true multi-objective Pareto. "
                        "Only 'convex_qp' is implemented in v1."
                    ),
                    "default": "convex_qp",
                },
            },
            "required": ["log_returns"],
        },
    ),
]


# ── Tool registration ─────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    Called automatically by the mcp SDK when the orchestrator sends
    a tools/list request at startup.

    Returns the list of Tool objects defined above.
    The orchestrator's MCP client caches these schemas and injects them
    into the LLM context — the LLM never reads this directly.
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

    Args:
        name:      Tool name — must match a name in TOOLS list
        arguments: Tool input arguments

    Returns:
        list[TextContent] — single element containing JSON result string

    Raises:
        ValueError: if tool name is not recognised
    """
    logger.info("tools/call: %s arguments=%s", name, list(arguments.keys()))

    if name == "optimise_portfolio":
        result = optimise_portfolio(
            log_returns=arguments["log_returns"],
            risk_free_rate=arguments.get("risk_free_rate", 0.065),
            n_frontier_points=arguments.get("n_frontier_points", 50),
            constraints=arguments.get("constraints", {}),
            solver=arguments.get("solver", "convex_qp"),
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

    stdio_server() opens stdin/stdout as async streams.
    server.run() starts the JSON-RPC event loop:
        - reads requests from stdin
        - dispatches to list_tools() or call_tool()
        - writes responses to stdout
        - runs until the orchestrator closes the connection
    """
    logger.info("Starting mcp-portfolio-optimiser server (stdio transport)")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-portfolio-optimiser",
                server_version="1.0.0",
                capabilities=ServerCapabilities(tools={}),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())