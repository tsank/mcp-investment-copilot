"""
orchestrator/clients/mcp_client_factory.py

Factory for creating MCP client sessions connected to individual servers
via stdio subprocess transport.

Design decisions:
    - One ClientSession per tool call — stateless, no connection pooling.
      Each node opens a session, makes its calls, closes it. This keeps
      each node fully isolated and avoids shared-state bugs across nodes.

    - Context manager pattern — callers use `async with get_client(...) as
      (session, tools):` which guarantees cleanup even on exceptions.

    - Server discovery by name — callers pass a server name string
      (e.g. "market_data"), not a path. The factory resolves the path
      to the server's server.py. This decouples nodes from filesystem layout.

    - tools/list called on connect — the factory fetches available tools
      from the server at session open time. Nodes can inspect available
      tools before calling, though in practice they call by known name.

Server name → server.py mapping:
    "market_data"        → servers/market_data/server.py
    "risk_engine"        → servers/risk_engine/server.py
    "portfolio_optimiser"→ servers/portfolio_optimiser/server.py
    "scenario_simulation"→ servers/scenario_simulation/server.py
    "compliance"         → servers/compliance/server.py

Usage (inside any node):
    from orchestrator.clients.mcp_client_factory import get_client

    async with get_client("market_data") as (session, tools):
        result = await session.call_tool(
            "get_price_history",
            {"symbols": ["RELIANCE.NS", "TCS.NS"], "period": "2y"}
        )
        data = json.loads(result.content[0].text)
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool

# ── Server registry ───────────────────────────────────────────────────────────

# Project root: two levels up from this file
#   this file:    orchestrator/clients/mcp_client_factory.py
#   project root: ../../
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Map server name → path to its server.py entry point
SERVER_REGISTRY: dict[str, Path] = {
    "market_data":         PROJECT_ROOT / "servers" / "market_data"         / "server.py",
    "risk_engine":         PROJECT_ROOT / "servers" / "risk_engine"          / "server.py",
    "portfolio_optimiser": PROJECT_ROOT / "servers" / "portfolio_optimiser"  / "server.py",
    "scenario_simulation": PROJECT_ROOT / "servers" / "scenario_simulation"  / "server.py",
    "compliance":          PROJECT_ROOT / "servers" / "compliance"            / "server.py",
}

VALID_SERVERS = list(SERVER_REGISTRY.keys())


# ── Factory ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def get_client(
    server_name: str,
) -> AsyncGenerator[tuple[ClientSession, list[Tool]], None]:
    """
    Async context manager that spawns a server subprocess and opens
    a connected MCP ClientSession.

    Lifecycle:
        1. Validate server_name against registry
        2. Build StdioServerParameters (points to server.py)
        3. stdio_client() spawns the subprocess, opens stdin/stdout pipes
        4. ClientSession.initialize() performs MCP handshake
        5. session.list_tools() fetches available tool schemas
        6. Yield (session, tools) to the caller
        7. On exit: session closes, subprocess terminates

    Args:
        server_name: One of the keys in SERVER_REGISTRY.
                     e.g. "market_data", "risk_engine"

    Yields:
        tuple[ClientSession, list[Tool]]:
            session — use session.call_tool(name, arguments) to invoke tools
            tools   — list of Tool schemas from the server (for inspection)

    Raises:
        ValueError: if server_name is not in SERVER_REGISTRY
        FileNotFoundError: if server.py does not exist at the registered path
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    if server_name not in SERVER_REGISTRY:
        raise ValueError(
            f"Unknown server: '{server_name}'. "
            f"Valid servers: {VALID_SERVERS}"
        )

    server_path = SERVER_REGISTRY[server_name]

    if not server_path.exists():
        raise FileNotFoundError(
            f"Server entry point not found: {server_path}. "
            f"Ensure the server has been implemented before calling get_client."
        )

    # ── Subprocess parameters ─────────────────────────────────────────────────
    # Use the same Python interpreter that is running the orchestrator.
    # This ensures the correct conda environment is used — critical because
    # each server has its own requirements.txt installed in the shared env.
    server_params = StdioServerParameters(
        command=sys.executable,       # e.g. /path/to/miniconda/envs/mcp-investment-copilot/bin/python
        args=[str(server_path)],      # e.g. /path/to/servers/market_data/server.py
        env=None,                     # inherit the current environment (includes .env vars)
    )

    # ── Open connection ───────────────────────────────────────────────────────
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:

            # MCP handshake — required before any tool calls
            await session.initialize()

            # Fetch tool schemas — nodes use these for inspection/logging
            tools_response = await session.list_tools()
            tools: list[Tool] = tools_response.tools

            yield session, tools


# ── Convenience helper ────────────────────────────────────────────────────────

async def call_tool(
    server_name: str,
    tool_name: str,
    arguments: dict,
) -> dict:
    """
    Convenience wrapper: open a client, call one tool, return parsed result.

    Suitable for nodes that make exactly one tool call per server.
    For nodes making multiple calls to the same server (e.g. compute_risk
    calls compute_risk_metrics then compute_garch_forecast), use get_client
    directly as a context manager to avoid spawning two subprocesses.

    Args:
        server_name: Server to connect to (key in SERVER_REGISTRY)
        tool_name:   Name of the tool to call
        arguments:   Tool input arguments as a dict

    Returns:
        Parsed dict from the tool's JSON response

    Raises:
        ValueError: if server_name is invalid
        KeyError:   if tool response content is malformed
    """
    async with get_client(server_name) as (session, _tools):
        result = await session.call_tool(tool_name, arguments)
        return json.loads(result.content[0].text)