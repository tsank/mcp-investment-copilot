"""
orchestrator/state.py

AgentState — the single state object that flows through the entire LangGraph graph.
Every node reads from and writes to this object. No node reads from or writes to
any other shared data structure.

Ownership rule: each result field is populated by exactly one node.
Consuming nodes read from fields populated by upstream nodes — they never
re-populate them.

File structure (top to bottom):
    1. Imports
    2. Enums
    3. Input models       (Portfolio)
    4. Nested sub-models  (building blocks for server output models)
    5. Server output models (one per server, named after the tool output)
    6. AgentState         (assembles everything — defined last)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class AnalysisType(str, Enum):
    """
    Drives LangGraph conditional routing after parse_query node.

    RISK:           fetch_market_data → compute_risk → check_compliance → synthesise
    OPTIMISATION:   fetch_market_data → compute_risk → optimise → check_compliance → synthesise
    SIMULATION:     fetch_market_data → compute_risk → simulate → check_compliance → synthesise
    FULL:           fetch_market_data → compute_risk → optimise + simulate (parallel)
                    → check_compliance → synthesise
    """
    RISK         = "risk"
    OPTIMISATION = "optimisation"
    SIMULATION   = "simulation"
    FULL         = "full"


# ── Input Models ──────────────────────────────────────────────────────────────

class Portfolio(BaseModel):
    """
    Current portfolio holdings provided by the user at query time.
    This is the input to the system — not a computed result.

    holdings: symbol → weight, must sum to 1.0
    total_value: portfolio value in INR
    benchmark: optional benchmark symbol for Sharpe comparison, e.g. "^NSEI"
    """
    holdings:    dict[str, float]   # e.g. {"RELIANCE.NS": 0.25, "TCS.NS": 0.20, ...}
    total_value: float              # INR, e.g. 1_000_000.0
    benchmark:   Optional[str] = None


# ── Nested Sub-Models: MarketDataResult ───────────────────────────────────────

class FundamentalData(BaseModel):
    """
    Fundamental data for a single symbol.
    Nested inside MarketDataResult.fundamentals.
    Produced by: get_fundamentals tool.
    Consumed by: check_compliance node (sector_map field).
    """
    pe_ratio:         Optional[float] = None
    market_cap_cr:    Optional[float] = None   # INR crores
    sector:           str                       # e.g. "Financial Services"
    industry:         Optional[str] = None
    dividend_yield:   float = 0.0
    market_cap_tier:  str                       # "large" | "mid" | "small"
    currency:         str = "INR"
    exchange:         Optional[str] = None
    long_name:        Optional[str] = None


# ── Server Output Model: Market Data ──────────────────────────────────────────

class MarketDataResult(BaseModel):
    """
    Output of the Market Data Server.
    Populated by: fetch_market_data node.
    Consumed by: compute_risk, optimise, simulate nodes.

    log_returns is the single source of truth for all downstream computation.
    All servers consume log_returns — never raw prices directly.
    prices is retained for max drawdown computation in the Risk Engine.
    """
    prices:              dict[str, list[float]]          # symbol → daily closing prices
    log_returns:         dict[str, list[float]]          # symbol → daily log-returns (pre-computed)
    dates:                list[str]                        # ISO 8601, aligned across all symbols
    fundamentals:         dict[str, FundamentalData]      # symbol → fundamental data
    source:               str                              # "fixture" | "live"
    period:               str                              # e.g. "2y" — echoed for audit
    missing_fundamentals: list[str] = Field(default_factory=list)


# ── Server Output Model: Risk Engine ──────────────────────────────────────────

class RiskMetricsResult(BaseModel):
    """
    Output of compute_risk_metrics tool in the Risk Engine Server.
    Populated by: compute_risk node.
    Consumed by: check_compliance node (var_95, cvar_95).
                 synthesise node (full result).

    All metrics are backward-looking — empirical, non-parametric.
    VaR and CVaR are direct percentiles of the observed return distribution.
    No distribution is fitted. No simulation is performed.
    """
    var_95:             float          # 5th percentile of empirical portfolio returns
    var_99:             float          # 1st percentile of empirical portfolio returns
    cvar_95:            float          # mean of returns below var_95 — primary risk metric
    cvar_99:            float          # mean of returns below var_99
    sharpe_ratio:       float          # annualised: (mean_return - rfr) / std × √252
    max_drawdown:       float          # peak-to-trough, negative fraction e.g. -0.23
    volatility:         dict[str, float]   # symbol → annualised volatility
    portfolio_return:   float          # annualised mean portfolio return
    risk_free_rate:     float          # echoed back — rfr used in computation
    computation_window: str            # e.g. "2y" — echoed for audit


# ── Nested Sub-Models: GARCHResult ────────────────────────────────────────────

class GARCHParams(BaseModel):
    """
    Fitted GARCH(1,1) parameters for a single asset.
    Nested inside GARCHAssetResult.
    Passed to Scenario Simulation Server via AgentState for run_garch_simulation.

    The Simulator consumes these parameters directly — it does not re-fit the model.
    This ensures parameter consistency between the Risk Engine and Simulator.
    """
    omega:  float    # ω — long-run variance component
    alpha:  float    # α — ARCH term, shock sensitivity
    beta:   float    # β — GARCH term, volatility persistence
    nu:     Optional[float] = None   # Student-t degrees of freedom
                                     # None if innovations = "gaussian"


class GARCHAssetResult(BaseModel):
    """
    Full GARCH output for a single asset.
    Nested inside GARCHResult.per_asset.
    """
    params:              GARCHParams
    alpha_plus_beta:     float          # persistence: close to 1 → slow mean reversion
    current_vol:         float          # σ_T — conditional vol at end of history, annualised
    longrun_vol:         float          # √(ω / (1 - α - β)) — unconditional volatility
    vol_forecast:        list[float]    # σ_{T+1}..σ_{T+H}, deterministic expected path
    regime:              str            # "elevated" | "normal" | "suppressed"
    aic:                 float          # Akaike Information Criterion
    bic:                 float          # Bayesian Information Criterion
    persistence_warning: bool           # True if alpha_plus_beta >= 1.0


# ── Server Output Model: GARCH ────────────────────────────────────────────────

class GARCHResult(BaseModel):
    """
    Output of compute_garch_forecast tool in the Risk Engine Server.
    Populated by: compute_risk node (second tool call).
    Consumed by: simulate node (run_garch_simulation tool).

    Two phases:
    Phase 1 — Estimation (backward-looking):
        Fits {ω, α, β, ν} from historical log-returns via MLE.
        Recovers σ_T by running the recursion through history.
    Phase 2 — Forecasting (forward-looking):
        Projects deterministic expected volatility path σ_{T+1}..σ_{T+H}.
        This is the mean of what GARCH predicts — not a simulated path.

    The full stochastic distribution is produced by run_garch_simulation
    in the Simulator, which draws N random paths from this fitted process.

    Handoff fields (garch_params, current_vols) are structured for direct
    consumption by the Simulator — no transformation required.
    """
    per_asset:              dict[str, GARCHAssetResult]
    portfolio_vol_forecast: list[float]     # weighted portfolio vol forecast path
    garch_model:            str             # "garch" | "egarch" | "gjr_garch"
    innovations_used:       str             # "student_t" | "gaussian"
    horizon_days:           int             # forecast horizon — echoed for audit

    # Handoff fields — consumed directly by run_garch_simulation via AgentState
    garch_params:   dict[str, GARCHParams]  # symbol → fitted params
    current_vols:   dict[str, float]        # symbol → σ_T (starting point for Simulator)


# ── Nested Sub-Models: OptimisationResult ─────────────────────────────────────

class FrontierPoint(BaseModel):
    """
    A single point on the Efficient Frontier (Pareto frontier).
    Nested inside OptimisationResult.efficient_frontier.

    Each point is a Pareto-optimal portfolio — no other portfolio
    can achieve higher return at the same volatility, or lower
    volatility at the same return.
    weights are included for drill-down and what-if analysis.
    """
    volatility:      float
    expected_return: float
    weights:         dict[str, float]   # portfolio weights at this frontier point


# ── Server Output Model: Portfolio Optimiser ──────────────────────────────────

class OptimisationResult(BaseModel):
    """
    Output of optimise_portfolio tool in the Portfolio Optimiser Server.
    Populated by: optimise node.
    Consumed by: check_compliance node (optimal_weights).
                 synthesise node (full result).

    Efficient Frontier computed via Scanning method (v1: convex_qp SLSQP).
    Maximum Sharpe portfolio is the tangency point on the Capital Market Line.
    solver_used is echoed back — confirms which solver path executed.
    Future solvers: differential_evolution (v2), nsga2 (v3).
    """
    optimal_weights:      dict[str, float]   # Maximum Sharpe portfolio weights
    max_sharpe_weights:   dict[str, float]   # explicit copy — same as optimal_weights
    expected_return:      float              # annualised, for max Sharpe portfolio
    portfolio_volatility: float              # annualised, for max Sharpe portfolio
    sharpe_ratio:         float              # of max Sharpe portfolio
    cml_slope:            float              # slope of Capital Market Line
    efficient_frontier:   list[FrontierPoint]
    solver_used:          str                # "convex_qp" | "differential_evolution" | "nsga2"


# ── Nested Sub-Models: SimulationResult ───────────────────────────────────────

class PercentileDistribution(BaseModel):
    """
    Percentile distribution of terminal portfolio returns.
    Nested inside SimulationOutput.
    """
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float


class SimulationOutput(BaseModel):
    """
    Output of a single simulation run — either run_monte_carlo or
    run_garch_simulation. Used twice inside SimulationResult.

    CVaR is the primary decision metric — the mean of losses in the
    worst (1-α) fraction of simulated paths.
    VaR is reported for context only — not used for gating decisions.

    distribution_used confirms which generative process was used:
    - "student_t": IID draws from fitted Student-t (run_monte_carlo)
    - "gaussian":  IID draws from fitted Gaussian (run_monte_carlo)
    - "historical_bootstrap": resampled empirical returns (run_monte_carlo)
    - "garch_student_t": ARMA-GARCH process with Student-t innovations
                         (run_garch_simulation)
    """
    cvar_95:           float                  # Expected Shortfall at 95% — primary metric
    cvar_99:           float                  # Expected Shortfall at 99%
    var_95:            float                  # VaR at 95% — context only
    var_99:            float                  # VaR at 99% — context only
    percentiles:       PercentileDistribution
    n_simulations:     int
    distribution_used: str
    fitted_nu:         Optional[float] = None  # Student-t ν — None for non-t distributions


# ── Server Output Model: Scenario Simulation ──────────────────────────────────

class SimulationResult(BaseModel):
    """
    Combined output of both simulation tools in the Scenario Simulation Server.
    Populated by: simulate node.
    Consumed by: check_compliance node (cvar_95 from best available result).
                 synthesise node (full result).

    monte_carlo: static distribution Monte Carlo (IID draws)
        "The future will look statistically like the past on average."

    garch_sim: GARCH-conditional Monte Carlo (serially dependent draws)
        "The future will evolve from where volatility is right now."
        Uses fitted GARCH params from GARCHResult via AgentState.
        Only populated if compute_garch_forecast was called upstream.

    regime_warning: True if CVaR(garch_sim) diverges materially from
        CVaR(monte_carlo) — signals current volatility regime is elevated
        relative to historical average. Surfaced in final recommendation.
    """
    monte_carlo:    Optional[SimulationOutput] = None
    garch_sim:      Optional[SimulationOutput] = None
    regime_warning: bool = False                # True if |cvar_garch - cvar_mc| > threshold


# ── Nested Sub-Models: ComplianceResult ───────────────────────────────────────

class Violation(BaseModel):
    """
    A single rule violation detected by the Compliance Server.
    Nested inside ComplianceResult.violations.

    severity "hard": recommendation must not proceed — blocking violation.
    severity "soft": recommendation may proceed with explicit caveats.
    """
    rule_id:     str     # e.g. "SINGLE_ASSET_CAP"
    description: str     # human-readable, suitable for synthesis node
    severity:    str     # "hard" | "soft"
    value:       float   # actual value that triggered the violation
    limit:       float   # the rule limit that was breached


class ComplianceWarning(BaseModel):
    """
    A non-blocking compliance warning.
    Nested inside ComplianceResult.warnings.
    Surfaced in recommendation but does not block it.
    """
    rule_id:     str
    description: str
    value:       float


# ── Server Output Model: Compliance ───────────────────────────────────────────

class ComplianceResult(BaseModel):
    """
    Output of check_compliance tool in the Compliance Server.
    Populated by: check_compliance node.
    Consumed by: synthesise node.

    The Compliance Server is the gatekeeper — always the final computation
    node before synthesis, regardless of analysis type.

    Stateless design: rules_version is mandatory in every request and
    echoed in every response. The same request + same rules_version always
    produces the same result — enables deterministic replay of historical
    compliance decisions for audit purposes.

    CVaR (not VaR) is the gating metric for risk threshold checks.
    """
    passed:        bool
    violations:    list[Violation]
    warnings:      list[ComplianceWarning]
    rules_version: str    # echoed back — closes the audit loop
    rules_profile: str    # echoed back — e.g. "retail_conservative"


# ── AgentState ────────────────────────────────────────────────────────────────

class AgentState(BaseModel):
    """
    The single state object that flows through the entire LangGraph graph.

    Ownership:
        Each result field is populated by exactly one node.
        No node writes to another node's field.
        errors is append-only — any node may append, none may clear.
        execution_trace is append-only — records node execution order.

    Default None for all result fields:
        A None field means the node has not yet run or has failed.
        The error_handler node inspects None fields to determine failures.
        The synthesise node generates a partial recommendation if non-critical
        fields are None.

    Data flow:
        parse_query         → populates: symbols, analysis_type
        fetch_market_data   → populates: market_data
        compute_risk        → populates: risk_metrics, garch_result
        optimise            → populates: optimisation_result
        simulate            → populates: simulation_result
        check_compliance    → populates: compliance_result
        synthesise          → populates: final_recommendation
    """

    # ── Input fields — provided at query time ─────────────────────
    query:         str
    portfolio:     Portfolio

    # ── Derived by parse_query node ───────────────────────────────
    symbols:       list[str] = Field(default_factory=list)
    analysis_type: AnalysisType = AnalysisType.FULL

    # ── Populated by fetch_market_data node ───────────────────────
    # Source: get_price_history + get_fundamentals tools
    market_data:   Optional[MarketDataResult] = None

    # ── Populated by compute_risk node ────────────────────────────
    # Source: compute_risk_metrics tool
    # Consumed by: optimise, simulate, check_compliance, synthesise
    risk_metrics:  Optional[RiskMetricsResult] = None

    # Source: compute_garch_forecast tool
    # Consumed by: simulate node (run_garch_simulation tool)
    # Handoff: garch_result.garch_params + garch_result.current_vols
    #          passed to Simulator via this field
    garch_result:  Optional[GARCHResult] = None

    # ── Populated by optimise node ────────────────────────────────
    # Source: optimise_portfolio tool
    # Consumed by: check_compliance, synthesise
    optimisation_result: Optional[OptimisationResult] = None

    # ── Populated by simulate node ────────────────────────────────
    # Source: run_monte_carlo + run_garch_simulation tools
    # Consumed by: check_compliance (cvar_95), synthesise
    simulation_result:   Optional[SimulationResult] = None

    # ── Populated by check_compliance node ────────────────────────
    # Source: check_compliance tool
    # Consumed by: synthesise
    compliance_result:   Optional[ComplianceResult] = None

    # ── Populated by synthesise node ──────────────────────────────
    # Final LLM-generated recommendation grounded in all computed fields
    final_recommendation: Optional[str] = None

    # ── System fields — maintained by orchestrator ────────────────
    errors:          list[str] = Field(default_factory=list)
    # Append-only. Any node may append on failure. None may clear.
    # Format: "{node_name}: {error_message}"

    execution_trace: list[str] = Field(default_factory=list)
    # Append-only. Each node appends its name on entry.
    # Used for observability and debugging.
    # Format: "{node_name}:{status}" e.g. "compute_risk:ok", "optimise:skipped"
