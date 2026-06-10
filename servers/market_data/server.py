"""
servers/market_data/server.py

MCP Server entry point for the Market Data Server.

Responsibilities:
    - Initialise the MCP Server instance
    - Register tool schemas via @server.list_tools()
    - Route tool calls via @server.call_tool()
    - Start stdio transport for local development

This file is intentionally thin — all computation logic lives in tools/.
This file handles protocol wiring only.

Transport:
    Local development : stdio (stdin/stdout pipes)
    AWS production    : HTTP+SSE (transport changes, protocol identical)

Usage:
    python servers/market_data/server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp.types import (
    ServerCapabilities,
    TextContent,
    Tool,
)

from tools.price_history import get_price_history
from tools.fundamentals import get_fundamentals

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-market-data")

# ── Server instance ───────────────────────────────────────────────────────────
server = Server("mcp-market-data")


# ── Tool schemas ──────────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="get_price_history",
        description=(
            "Retrieve historical daily closing prices and pre-computed log-returns "
            "for a list of NSE stock symbols over a specified period. "
            "Log-returns are pre-computed here and are the single source of truth — "
            "all downstream servers (Risk Engine, Optimiser, Simulator) consume "
            "log_returns from this output, never raw prices directly. "
            "Must be called before any risk, optimisation, or simulation computation. "
            "Data source is controlled by MARKET_DATA_SOURCE environment variable: "
            "'fixture' (default, deterministic CSV for testing) or 'live' (yFinance, v2)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "NSE ticker symbols e.g. ['RELIANCE.NS', 'TCS.NS']",
                },
                "period": {
                    "type": "string",
                    "description": "Historical period: '1y', '2y', '3y', '5y'. Default '2y'.",
                    "default": "2y",
                },
                "return_type": {
                    "type": "string",
                    "enum": ["log", "simple"],
                    "description": "Return type: 'log' (default) or 'simple'.",
                    "default": "log",
                },
            },
            "required": ["symbols"],
        },
    ),
    Tool(
        name="get_fundamentals",
        description=(
            "Retrieve fundamental data for a list of NSE symbols. "
            "Returns P/E ratio, market capitalisation (INR crores), sector, "
            "industry, dividend yield, and market cap tier (large/mid/small "
            "per SEBI definition). "
            "The sector field is consumed by the Compliance server for sector "
            "concentration checks. "
            "The industry field provides finer-grained classification for "
            "future compliance rules. "
            "Use when the query involves valuation, sector analysis, or "
            "market cap tier filtering (e.g. 'should I increase mid-cap allocation')."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "NSE ticker symbols e.g. ['RELIANCE.NS', 'TCS.NS']",
                },
            },
            "required": ["symbols"],
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
    All tool functions are synchronous — called directly here.
    Results are serialised to JSON and wrapped in TextContent.

    Args:
        name:      Tool name — must match a name in TOOLS list
        arguments: Tool input arguments — passed as **kwargs to tool function

    Returns:
        list[TextContent] — single element containing JSON result string

    Raises:
        ValueError: if tool name is not recognised
    """
    logger.info("tools/call: %s arguments=%s", name, list(arguments.keys()))

    if name == "get_price_history":
        result = get_price_history(
            symbols=arguments["symbols"],
            period=arguments.get("period", "2y"),
            return_type=arguments.get("return_type", "log"),
        )

    elif name == "get_fundamentals":
        result = get_fundamentals(
            symbols=arguments["symbols"],
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
    logger.info("Starting mcp-market-data server (stdio transport)")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-market-data",
                server_version="1.0.0",
                capabilities=ServerCapabilities(tools={}),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())