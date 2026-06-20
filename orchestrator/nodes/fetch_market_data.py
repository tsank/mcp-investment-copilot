"""
orchestrator/nodes/fetch_market_data.py

Node 2 of 7 — fetch_market_data

Responsibility:
    Call the Market Data Server to retrieve price history and fundamental
    data for all symbols in state.symbols. Package results into
    MarketDataResult and write to state.market_data.

Reads from AgentState:
    - symbols   (list[str])  — set by parse_query node
    - portfolio (Portfolio)  — used for period default

Writes to AgentState:
    - market_data (MarketDataResult) — prices, log_returns, dates,
                                       fundamentals, source, period,
                                       missing_fundamentals

MCP tools called (both against "market_data" server):
    1. get_price_history  → prices, log_returns, dates, source, period
    2. get_fundamentals   → fundamentals, missing_fundamentals

Design decisions:

    Single subprocess — both tool calls share one ClientSession.
    get_client() is used directly (not call_tool()) so the subprocess
    is spawned once and both calls go over the same stdio pipe.
    This halves subprocess overhead for this node.

    Period hardcoded to "2y" in v1:
        Consistent with fixture data downloaded by scripts/download_fixtures.py.
        Promoted to a configurable parameter in v2 when live feed is added.

    return_type hardcoded to "log":
        Log-returns are the single source of truth for all downstream
        computation. Simple returns are never passed to Risk, Optimiser,
        or Simulator. This is an architectural invariant, not a user option.

    Fundamentals failure is non-fatal:
        If get_fundamentals returns missing symbols, they are recorded in
        MarketDataResult.missing_fundamentals. The graph continues.
        Compliance will have no sector_map entries for missing symbols —
        they fall into "Unknown" sector grouping per the compliance tool design.

    Fatal failure (price history):
        If get_price_history fails, the node appends to state.errors and
        returns market_data=None. Downstream nodes must guard against None.
        The synthesise node will generate a partial/error recommendation.
"""

from __future__ import annotations

import logging

from orchestrator.clients.mcp_client_factory import get_client
from orchestrator.state import AgentState, FundamentalData, MarketDataResult

logger = logging.getLogger(__name__)

# Hardcoded in v1 — promoted to AgentState field in v2
_PERIOD = "2y"
_RETURN_TYPE = "log"


async def fetch_market_data(state: AgentState) -> dict:
    """
    LangGraph node — fetch_market_data.

    Opens one ClientSession to the Market Data Server and makes two
    sequential tool calls: get_price_history then get_fundamentals.

    Args:
        state: Current AgentState. Reads: symbols, portfolio.

    Returns:
        dict with keys: market_data, execution_trace, errors.
        LangGraph merges this into AgentState.
    """
    logger.info("fetch_market_data: symbols=%s period=%s", state.symbols, _PERIOD)

    try:
        async with get_client("market_data") as (session, _tools):

            # ── Call 1: price history ─────────────────────────────────────────
            logger.info("fetch_market_data: calling get_price_history")
            price_result = await session.call_tool(
                "get_price_history",
                {
                    "symbols":     state.symbols,
                    "period":      _PERIOD,
                    "return_type": _RETURN_TYPE,
                },
            )
            import json
            price_data = json.loads(price_result.content[0].text)
            logger.info(
                "fetch_market_data: get_price_history ok — %d symbols, %d dates",
                len(price_data["prices"]),
                len(price_data["dates"]),
            )

            # ── Call 2: fundamentals ──────────────────────────────────────────
            logger.info("fetch_market_data: calling get_fundamentals")
            fund_result = await session.call_tool(
                "get_fundamentals",
                {"symbols": state.symbols},
            )
            fund_data = json.loads(fund_result.content[0].text)
            logger.info(
                "fetch_market_data: get_fundamentals ok — %d found, %d missing",
                len(fund_data["fundamentals"]),
                len(fund_data.get("missing", [])),
            )

    except Exception as exc:
        logger.error("fetch_market_data: failed — %s", exc)
        return {
            "market_data":     None,
            "execution_trace": state.execution_trace + ["fetch_market_data:error"],
            "errors":          state.errors + [f"fetch_market_data: {exc}"],
        }

    # ── Build MarketDataResult ────────────────────────────────────────────────
    # Deserialise fundamentals dict into FundamentalData models
    fundamentals: dict[str, FundamentalData] = {}
    for symbol, fd in fund_data["fundamentals"].items():
        try:
            fundamentals[symbol] = FundamentalData(**fd)
        except Exception as exc:
            logger.warning(
                "fetch_market_data: could not parse FundamentalData for %s — %s",
                symbol, exc,
            )

    market_data = MarketDataResult(
        prices=price_data["prices"],
        log_returns=price_data["log_returns"],
        dates=price_data["dates"],
        fundamentals=fundamentals,
        source=price_data["source"],
        period=price_data["period"],
        missing_fundamentals=fund_data.get("missing", []),
    )

    return {
        "market_data":     market_data,
        "execution_trace": state.execution_trace + ["fetch_market_data:ok"],
        "errors":          state.errors,
    }