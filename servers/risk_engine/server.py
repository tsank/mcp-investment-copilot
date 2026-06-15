"""
servers/risk_engine/server.py

MCP Server entry point for the Risk Engine Server.

Responsibilities:
    - Initialise the MCP Server instance
    - Register tool schemas via @server.list_tools()
    - Route tool calls via @server.call_tool()
    - Start stdio transport for local development

This file is intentionally thin — all computation logic lives in tools/.
This file handles protocol wiring only.

Tools exposed:
    compute_risk_metrics   — backward-looking, non-parametric risk metrics
    compute_garch_forecast — GARCH(1,1) fitting and volatility term structure

Transport:
    Local development : stdio (stdin/stdout pipes)
    AWS production    : HTTP+SSE (transport changes, protocol identical)

Usage:
    python servers/risk_engine/server.py
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

from tools.risk_metrics import compute_risk_metrics
from tools.garch_forecast import compute_garch_forecast

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-risk-engine")

# ── Server instance ───────────────────────────────────────────────────────────
server = Server("mcp-risk-engine")


# ── Tool schemas ──────────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="compute_risk_metrics",
        description=(
            "Compute backward-looking, non-parametric risk metrics for a portfolio "
            "given pre-computed log-return series and portfolio weights. "
            "All metrics are derived directly from the empirical return distribution "
            "over the historical window — no distribution is fitted, no simulation "
            "is performed. "
            "Returns Historical VaR at 95% and 99% (percentile of empirical returns), "
            "Historical CVaR at 95% and 99% (mean of tail beyond VaR threshold), "
            "annualised Sharpe ratio, maximum drawdown (peak-to-trough), and "
            "per-symbol annualised volatility. "
            "CVaR is the primary decision metric — VaR is reported for context only. "
            "Call this after get_price_history and before optimisation, simulation, "
            "or compliance checks. "
            "These are descriptive statistics of what actually happened — "
            "not forecasts. For forward-looking risk, use compute_garch_forecast "
            "and run_monte_carlo."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "log_returns": {
                    "type": "object",
                    "description": (
                        "dict[symbol, list[float]] — pre-computed log-returns "
                        "from get_price_history output. "
                        "All series must have identical length (date-aligned)."
                    ),
                },
                "weights": {
                    "type": "object",
                    "description": (
                        "dict[symbol, float] — current portfolio weights. "
                        "Must sum to 1.0. "
                        "Symbols must be a subset of log_returns keys."
                    ),
                },
                "prices": {
                    "type": "object",
                    "description": (
                        "dict[symbol, list[float]] — daily closing prices "
                        "from get_price_history output. "
                        "Required for maximum drawdown computation on price series."
                    ),
                },
                "risk_free_rate": {
                    "type": "number",
                    "description": (
                        "Annualised risk-free rate as a decimal. "
                        "Default 0.065 (6.5% — RBI repo rate proxy for INR). "
                        "Used in Sharpe ratio computation."
                    ),
                    "default": 0.065,
                },
            },
            "required": ["log_returns", "weights", "prices"],
        },
    ),
    Tool(
        name="compute_garch_forecast",
        description=(
            "Fit a constant-mean GARCH(1,1) model with Student-t innovations "
            "to the log-return series of each asset via Maximum Likelihood Estimation. "
            "Captures volatility clustering (ARCH effect) — the empirically documented "
            "tendency for high-volatility periods to persist in equity markets, "
            "strongly present in Indian NSE equities. "
            "Two phases: "
            "(1) Estimation — fits {ω, α, β, ν} from historical log-returns. "
            "(2) Forecasting — projects deterministic expected volatility path "
            "σ_{T+1}..σ_{T+H} via GARCH mean-reversion formula. "
            "This is the EXPECTED volatility path (mean of GARCH process), "
            "not a stochastic simulation. "
            "The full stochastic distribution is produced by run_garch_simulation "
            "in the Scenario Simulation Server, which consumes the fitted parameters "
            "from this tool via AgentState. "
            "V1 assumptions: constant mean, GARCH(1,1) fixed order, Student-t innovations. "
            "V2 will add ARMA mean with auto-selected order. "
            "Use when the query involves forward-looking volatility assessment, "
            "regime detection, or when the Simulator requires GARCH-conditional paths."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "log_returns": {
                    "type": "object",
                    "description": (
                        "dict[symbol, list[float]] — pre-computed log-returns "
                        "from get_price_history output. "
                        "Minimum 100 observations per symbol required."
                    ),
                },
                "horizon_days": {
                    "type": "integer",
                    "description": (
                        "Forecast horizon in trading days. "
                        "e.g. 21 (1 month), 63 (1 quarter), 252 (1 year). "
                        "Default 21."
                    ),
                    "default": 21,
                },
                "model": {
                    "type": "string",
                    "enum": ["garch", "egarch", "gjr_garch"],
                    "description": (
                        "GARCH model variant. "
                        "'garch': standard GARCH(1,1) — default. "
                        "'egarch': captures leverage effect. "
                        "'gjr_garch': asymmetric response to positive/negative shocks."
                    ),
                    "default": "garch",
                },
                "innovations": {
                    "type": "string",
                    "enum": ["student_t", "gaussian"],
                    "description": (
                        "Innovation distribution. "
                        "'student_t': Student-t with fitted ν — default, "
                        "captures fat tails present in NSE equity returns. "
                        "'gaussian': Normal — baseline comparison only."
                    ),
                    "default": "student_t",
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

    if name == "compute_risk_metrics":
        result = compute_risk_metrics(
            log_returns=arguments["log_returns"],
            weights=arguments["weights"],
            prices=arguments["prices"],
            risk_free_rate=arguments.get("risk_free_rate", 0.065),
        )

    elif name == "compute_garch_forecast":
        result = compute_garch_forecast(
            log_returns=arguments["log_returns"],
            horizon_days=arguments.get("horizon_days", 21),
            model=arguments.get("model", "garch"),
            innovations=arguments.get("innovations", "student_t"),
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
    logger.info("Starting mcp-risk-engine server (stdio transport)")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-risk-engine",
                server_version="1.0.0",
                capabilities=ServerCapabilities(tools={}),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())