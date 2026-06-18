"""
servers/compliance/server.py

MCP Server entry point for the Compliance Server.

Responsibilities:
    - Initialise the MCP Server instance
    - Register tool schemas via @server.list_tools()
    - Route tool calls via @server.call_tool()
    - Start stdio transport for local development

This file is intentionally thin — all computation logic lives in tools/.
This file handles protocol wiring only.

Tools exposed:
    check_compliance — validate portfolio against versioned YAML ruleset
                       stateless gatekeeper — always last before synthesis
                       CVaR (not VaR) is the primary gating metric
                       rules_version mandatory for audit trail

Transport:
    Local development : stdio (stdin/stdout pipes)
    AWS production    : HTTP+SSE (transport changes, protocol identical)

Usage:
    python servers/compliance/server.py
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

from tools.check_compliance import check_compliance

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-compliance")

# ── Server instance ───────────────────────────────────────────────────────────
server = Server("mcp-compliance")


# ── Tool schemas ──────────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="check_compliance",
        description=(
            "Validate a portfolio allocation against a versioned compliance ruleset. "
            "This is the gatekeeper — always called as the final computation step "
            "before the synthesis node generates a recommendation. "
            "Checks the user's CURRENT portfolio weights (Portfolio.holdings), "
            "not the optimal weights from the Portfolio Optimiser. "
            "Reason: compliance checks what the user actually holds, "
            "not a hypothetical rebalanced position. "
            "Rules checked (v1): "
            "    SINGLE_ASSET_CAP — no single asset weight exceeds limit (hard). "
            "    SECTOR_CAP       — no single sector weight exceeds limit (hard). "
            "    CVAR_THRESHOLD   — portfolio CVaR_95 does not exceed limit (hard). "
            "    MIN_ASSETS       — minimum number of active assets held (soft). "
            "    MIN_POSITION_SIZE — no token positions below minimum size (soft). "
            "CVaR (not VaR) gates the risk threshold check — CVaR captures "
            "tail loss magnitude, VaR only identifies the threshold. "
            "rules_version is mandatory and echoed in every response — "
            "same portfolio + same rules_version always produces same result, "
            "enabling deterministic replay of historical compliance decisions. "
            "Rule severity: hard = blocks recommendation, soft = warning only. "
            "passed=True only if zero hard violations. "
            "Soft violations are reported as warnings regardless of passed status."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "description": (
                        "dict[symbol, float] — current portfolio weights. "
                        "Must sum to 1.0. "
                        "Use Portfolio.holdings from AgentState — "
                        "the user's actual current allocation."
                    ),
                },
                "sector_map": {
                    "type": "object",
                    "description": (
                        "dict[symbol, str] — symbol to sector mapping. "
                        "Derived from MarketDataResult.fundamentals[symbol].sector. "
                        "Required for SECTOR_CAP rule. "
                        "e.g. {'RELIANCE.NS': 'Energy', 'TCS.NS': 'Technology'}"
                    ),
                },
                "var_95": {
                    "type": "number",
                    "description": (
                        "Historical VaR at 95% confidence — from compute_risk_metrics. "
                        "Included for audit trail context. "
                        "NOT used as the gating metric — CVaR gates risk threshold."
                    ),
                },
                "cvar_95": {
                    "type": "number",
                    "description": (
                        "CVaR (Expected Shortfall) at 95% confidence. "
                        "Primary gating metric for CVAR_THRESHOLD rule. "
                        "Use from Simulator output if available (forward-looking). "
                        "Fall back to Risk Engine CVaR if Simulator did not run."
                    ),
                },
                "rules_profile": {
                    "type": "string",
                    "enum": ["retail_conservative", "institutional"],
                    "description": (
                        "Investor profile determining which ruleset to apply. "
                        "'retail_conservative': tighter limits, suitable for "
                        "individual investors. "
                        "'institutional': looser limits, suitable for "
                        "institutional mandates. "
                        "Default 'retail_conservative'."
                    ),
                    "default": "retail_conservative",
                },
                "rules_version": {
                    "type": "string",
                    "description": (
                        "Version of the ruleset to apply. "
                        "Mandatory — must be specified explicitly. "
                        "Echoed in response to close the audit loop. "
                        "e.g. 'v1.0'. "
                        "Same portfolio + same rules_version always "
                        "produces the same compliance result."
                    ),
                    "default": "v1.0",
                },
            },
            "required": ["weights", "sector_map", "var_95", "cvar_95"],
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

    if name == "check_compliance":
        result = check_compliance(
            weights=arguments["weights"],
            sector_map=arguments["sector_map"],
            var_95=arguments["var_95"],
            cvar_95=arguments["cvar_95"],
            rules_profile=arguments.get("rules_profile", "retail_conservative"),
            rules_version=arguments.get("rules_version", "v1.0"),
        )

    else:
        raise ValueError(
            f"Unknown tool: '{name}'. "
            f"Available tools: {[t.name for t in TOOLS]}"
        )

    logger.info(
        "tools/call: %s completed — passed=%s violations=%d warnings=%d",
        name,
        result["passed"],
        len(result["violations"]),
        len(result["warnings"]),
    )

    return [TextContent(type="text", text=json.dumps(result))]


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    """
    Start the MCP server with stdio transport.
    """
    logger.info("Starting mcp-compliance server (stdio transport)")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-compliance",
                server_version="1.0.0",
                capabilities=ServerCapabilities(tools={}),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())