"""
orchestrator/graph.py

LangGraph StateGraph — wires all 7 nodes into the investment analysis pipeline.

This file has three responsibilities:
    1. Import all nodes
    2. Define route_after_risk() — the conditional edge function
    3. Build and compile the StateGraph

The compiled graph is the "agent" — consumed by the FastAPI entry point.

Graph topology:

    START
      → parse_query
      → fetch_market_data
      → compute_risk
      → [route_after_risk]
            "risk"         → check_compliance
            "optimisation" → optimise → check_compliance
            "simulation"   → simulate → check_compliance
            "full"         → optimise → simulate → check_compliance
      → synthesise
      → END

All four paths converge at check_compliance → synthesise → END.

Design decisions:

    MemorySaver checkpointer (optional):
        Enabled when thread_id is provided to graph.invoke().
        Allows mid-graph state inspection and replay for debugging.
        Disabled in production FastAPI invocations (stateless per request).

    Synchronous compile():
        graph.compile() is called once at module import time.
        The compiled graph object is reused across all requests.
        Thread-safe — LangGraph compiled graphs are stateless objects.

    route_after_risk returns node name strings:
        LangGraph conditional edges expect the routing function to return
        the name of the next node as a string, exactly matching the name
        used in graph.add_node(). Mismatches raise KeyError at runtime.

    No parallel execution in v1:
        optimise → simulate is sequential even for FULL analysis type.
        simulate needs optimisation_result.optimal_weights to run
        monte_carlo_optimal and garch_sim_optimal.
        Parallel execution (via Send()) deferred to v2.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.state import AgentState, AnalysisType
from orchestrator.nodes.parse_query import parse_query
from orchestrator.nodes.fetch_market_data import fetch_market_data
from orchestrator.nodes.compute_risk import compute_risk
from orchestrator.nodes.optimise import optimise
from orchestrator.nodes.simulate import simulate
from orchestrator.nodes.check_compliance import check_compliance
from orchestrator.nodes.synthesise import synthesise

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)


# ── Conditional edge function ─────────────────────────────────────────────────

def route_after_risk(state: AgentState) -> str:
    """
    Conditional edge — called after compute_risk node completes.

    Reads state.analysis_type and returns the name of the next node.
    LangGraph uses this string to look up the next node in the graph.

    Routing:
        RISK         → "check_compliance"  (skip optimise and simulate)
        OPTIMISATION → "optimise"          (optimise then compliance)
        SIMULATION   → "simulate"          (simulate then compliance)
        FULL         → "optimise"          (optimise → simulate → compliance)

    Note: FULL routes to "optimise" first (not "simulate") because
    simulate needs optimal_weights from optimise for the optimal-weight
    simulation runs. Sequential by design in v1.

    Args:
        state: Current AgentState after compute_risk has written
               risk_metrics and garch_result.

    Returns:
        str — name of the next node to execute.
    """
    analysis_type = state.analysis_type
    logger.info("route_after_risk: analysis_type=%s", analysis_type.value)

    if analysis_type == AnalysisType.RISK:
        return "check_compliance"

    if analysis_type == AnalysisType.OPTIMISATION:
        return "optimise"

    if analysis_type == AnalysisType.SIMULATION:
        return "simulate"

    # FULL — default
    return "optimise"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(use_checkpointer: bool = False) -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Args:
        use_checkpointer: If True, attaches MemorySaver for state
                          inspection and replay. Use during development/debugging.
                          Set to False in production (stateless per request).

    Returns:
        Compiled LangGraph graph — call graph.invoke(state) to run.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("parse_query",        parse_query)
    graph.add_node("fetch_market_data",  fetch_market_data)
    graph.add_node("compute_risk",       compute_risk)
    graph.add_node("optimise",           optimise)
    graph.add_node("simulate",           simulate)
    graph.add_node("check_compliance",   check_compliance)
    graph.add_node("synthesise",         synthesise)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "parse_query")

    # ── Fixed edges ───────────────────────────────────────────────────────────
    graph.add_edge("parse_query",       "fetch_market_data")
    graph.add_edge("fetch_market_data", "compute_risk")

    # ── Conditional edge after compute_risk ───────────────────────────────────
    graph.add_conditional_edges(
        "compute_risk",
        route_after_risk,
        {
            "check_compliance": "check_compliance",   # RISK path
            "optimise":         "optimise",           # OPTIMISATION + FULL path
            "simulate":         "simulate",           # SIMULATION path
        },
    )

    # ── OPTIMISATION path: optimise → check_compliance ────────────────────────
    # Also serves as the first half of the FULL path
    graph.add_edge("optimise", "simulate")

    # ── SIMULATION path and FULL path converge here ───────────────────────────
    graph.add_edge("simulate", "check_compliance")

    # ── Final fixed edges — all paths converge ────────────────────────────────
    graph.add_edge("check_compliance", "synthesise")
    graph.add_edge("synthesise",       END)

    # ── Compile ───────────────────────────────────────────────────────────────
    checkpointer = MemorySaver() if use_checkpointer else None
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info(
        "graph: compiled — nodes=%s checkpointer=%s",
        list(graph.nodes.keys()) if hasattr(graph, 'nodes') else "see graph definition",
        "MemorySaver" if use_checkpointer else "none",
    )

    return compiled


# ── Module-level compiled graph ───────────────────────────────────────────────
# Compiled once at import time — reused across all FastAPI requests.
# Use build_graph(use_checkpointer=True) in development for state inspection.
investment_graph = build_graph(use_checkpointer=False)