"""
api/schemas/response.py

Pydantic models for the POST /api/v1/analyse response body.

AnalyseResponse is the full structured state returned to the client.
All result fields are Optional — any node may have failed or been
skipped depending on analysis_type.

Design decision — full structured response (Option 2):
    Returns recommendation + compliance + risk + simulation + optimisation.
    Gradio UI can render:
        - recommendation as prose text
        - compliance as pass/fail indicator + violation list
        - risk_metrics as a key numbers table
        - simulation as CVaR comparison panel
        - optimisation as weight delta table
    This enables richer UI in v1 without requiring v2 React dashboard.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Compliance ────────────────────────────────────────────────────────────────

class ViolationResponse(BaseModel):
    rule_id:     str
    description: str
    severity:    str
    value:       float
    limit:       float


class WarningResponse(BaseModel):
    rule_id:     str
    description: str
    value:       float


class ComplianceResponse(BaseModel):
    passed:        bool
    violations:    list[ViolationResponse]
    warnings:      list[WarningResponse]
    rules_profile: str
    rules_version: str


# ── Risk metrics ──────────────────────────────────────────────────────────────

class RiskMetricsResponse(BaseModel):
    var_95:             float
    var_99:             float
    cvar_95:            float
    cvar_99:            float
    sharpe_ratio:       float
    max_drawdown:       float
    volatility:         dict[str, float]
    portfolio_return:   float
    risk_free_rate:     float
    computation_window: str


# ── Simulation ────────────────────────────────────────────────────────────────

class PercentileResponse(BaseModel):
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float


class SimulationOutputResponse(BaseModel):
    cvar_95:           float
    cvar_99:           float
    var_95:            float
    var_99:            float
    percentiles:       PercentileResponse
    n_simulations:     int
    distribution_used: str
    fitted_nu:         Optional[float] = None


class SimulationResponse(BaseModel):
    monte_carlo:          Optional[SimulationOutputResponse] = None
    monte_carlo_optimal:  Optional[SimulationOutputResponse] = None
    garch_sim:            Optional[SimulationOutputResponse] = None
    garch_sim_optimal:    Optional[SimulationOutputResponse] = None
    regime_warning:       bool = False


# ── Optimisation ──────────────────────────────────────────────────────────────

class OptimisationResponse(BaseModel):
    optimal_weights:      dict[str, float]
    expected_return:      float
    portfolio_volatility: float
    sharpe_ratio:         float
    cml_slope:            float
    solver_used:          str


# ── Top-level response ────────────────────────────────────────────────────────

class AnalyseResponse(BaseModel):
    """
    Full structured response from POST /api/v1/analyse.

    All result fields are Optional — a field is None if:
        - The corresponding node was skipped (analysis_type routing)
        - The node failed (error recorded in errors list)

    recommendation is always present — synthesise node generates a
    partial/error recommendation even if upstream nodes failed.
    """
    recommendation: str = Field(
        ...,
        description="GPT-4o generated investment recommendation.",
    )
    compliance:     Optional[ComplianceResponse]    = None
    risk_metrics:   Optional[RiskMetricsResponse]   = None
    simulation:     Optional[SimulationResponse]    = None
    optimisation:   Optional[OptimisationResponse]  = None
    analysis_type:  str = Field(
        ...,
        description="Analysis type that was executed: risk|optimisation|simulation|full",
    )
    execution_trace: list[str] = Field(
        default_factory=list,
        description="Node execution trace for observability.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal errors encountered during analysis.",
    )