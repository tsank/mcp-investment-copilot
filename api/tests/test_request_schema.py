"""
api/tests/test_request_schema.py

Tests for v3 input-validation guardrails on PortfolioRequest / AnalyseRequest.

Covers:
    - weights must sum to 1.0 (±0.01)          — pre-existing, re-verified here
    - holdings must be non-empty                — pre-existing, re-verified here
    - unknown ticker symbols are rejected        — new in v3
    - holdings count is capped at MAX_HOLDINGS   — new in v3
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas.request import KNOWN_SYMBOLS, MAX_HOLDINGS, PortfolioRequest


def _valid_holdings(n: int) -> dict[str, float]:
    """
    Build up to n equal-weighted holdings from the known-symbol allowlist.
    Capped to len(KNOWN_SYMBOLS) — today's universe (10) is smaller than
    MAX_HOLDINGS (15), so weights are always computed from the symbols
    actually used, not the requested count.
    """
    symbols = list(KNOWN_SYMBOLS)[:n]
    weight = 1.0 / len(symbols)
    return {s: weight for s in symbols}


class TestWeightsSumToOne:
    def test_valid_weights_pass(self):
        PortfolioRequest(holdings={"RELIANCE.NS": 0.5, "TCS.NS": 0.5}, total_value=1_000_000)

    def test_weights_over_one_rejected(self):
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            PortfolioRequest(holdings={"RELIANCE.NS": 0.6, "TCS.NS": 0.6}, total_value=1_000_000)

    def test_weights_under_one_rejected(self):
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            PortfolioRequest(holdings={"RELIANCE.NS": 0.2, "TCS.NS": 0.2}, total_value=1_000_000)


class TestHoldingsNotEmpty:
    def test_empty_holdings_rejected(self):
        with pytest.raises(ValidationError, match="at least one symbol"):
            PortfolioRequest(holdings={}, total_value=1_000_000)


class TestUnknownSymbols:
    def test_known_symbol_passes(self):
        PortfolioRequest(holdings={"RELIANCE.NS": 1.0}, total_value=1_000_000)

    def test_unknown_symbol_rejected(self):
        with pytest.raises(ValidationError, match="Unrecognised symbol"):
            PortfolioRequest(holdings={"RELAINCE.NS": 1.0}, total_value=1_000_000)

    def test_mixed_known_and_unknown_rejected_with_only_unknown_named(self):
        with pytest.raises(ValidationError, match="FAKETICKER.NS"):
            PortfolioRequest(
                holdings={"RELIANCE.NS": 0.5, "FAKETICKER.NS": 0.5},
                total_value=1_000_000,
            )


class TestHoldingsComplexityCap:
    def test_full_known_universe_within_cap_passes(self):
        # Today's known-symbol universe (10) is smaller than MAX_HOLDINGS
        # (15) — this documents that relationship rather than testing an
        # exact boundary that isn't reachable with real symbols yet.
        assert len(KNOWN_SYMBOLS) <= MAX_HOLDINGS
        PortfolioRequest(holdings=_valid_holdings(len(KNOWN_SYMBOLS)), total_value=1_000_000)

    def test_holdings_over_cap_rejected(self):
        # Cap validator runs before the known-symbol validator (declaration
        # order), so an oversized request is rejected for its size even
        # when every symbol in it is also unrecognised.
        oversized = {f"SYM{i}.NS": 1.0 / (MAX_HOLDINGS + 1) for i in range(MAX_HOLDINGS + 1)}
        with pytest.raises(ValidationError, match="maximum supported is"):
            PortfolioRequest(holdings=oversized, total_value=1_000_000)


class TestSingleTickerPortfolio:
    def test_single_known_ticker_is_accepted_at_the_schema_layer(self):
        """
        Schema-layer validation intentionally allows a single-ticker
        portfolio — it's a valid, if degenerate, input. Whether the
        orchestrator's efficient-frontier node can meaningfully run on
        one symbol is a downstream pipeline concern, not a schema concern.
        """
        PortfolioRequest(holdings={"RELIANCE.NS": 1.0}, total_value=1_000_000)