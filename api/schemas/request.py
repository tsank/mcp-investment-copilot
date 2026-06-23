"""
api/schemas/request.py

Pydantic models for the POST /api/v1/analyse request body.

AnalyseRequest is the single entry point into the system.
It maps directly onto AgentState input fields:
    query     → AgentState.query
    portfolio → AgentState.portfolio

Validation rules mirror the constraints in orchestrator/state.py:
    - holdings must be non-empty
    - holdings weights must sum to 1.0 (±0.01 tolerance)
    - total_value must be positive
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PortfolioRequest(BaseModel):
    """
    Portfolio holdings provided by the user.
    Maps directly to orchestrator.state.Portfolio.
    """
    holdings:    dict[str, float] = Field(
        ...,
        description="Symbol → weight mapping. Weights must sum to 1.0.",
        example={"RELIANCE.NS": 0.25, "TCS.NS": 0.20,
                 "INFY.NS": 0.20, "HDFCBANK.NS": 0.20, "ICICIBANK.NS": 0.15},
    )
    total_value: float = Field(
        ...,
        gt=0,
        description="Total portfolio value in INR.",
        example=1_000_000.0,
    )
    benchmark: Optional[str] = Field(
        None,
        description="Optional benchmark symbol e.g. '^NSEI'.",
    )

    @field_validator("holdings")
    @classmethod
    def holdings_not_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("holdings must contain at least one symbol")
        return v

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "PortfolioRequest":
        total = sum(self.holdings.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Portfolio weights must sum to 1.0 (got {total:.4f}). "
                f"Adjust weights before submitting."
            )
        return self


class AnalyseRequest(BaseModel):
    """
    Request body for POST /api/v1/analyse.

    query:     Natural language question about the portfolio.
               parse_query node extracts symbols and analysis_type from this.
    portfolio: Current holdings and total value.
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language query about the portfolio.",
        example="Analyse my portfolio and suggest rebalancing.",
    )
    portfolio: PortfolioRequest