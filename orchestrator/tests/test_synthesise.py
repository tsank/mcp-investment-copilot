"""
orchestrator/tests/test_synthesise.py
 
Tests for v3 compliance cross-check guardrail in the synthesise node.
 
Covers:
    - a compliance breach ALWAYS produces a prepended deterministic alert,
      even when the mocked LLM response completely ignores the breach —
      proving the guardrail does not depend on LLM behaviour
    - a passing compliance result never gets an alert prepended
      (clean pass-through, no unnecessary noise)
    - compliance_result=None does not crash the node
    - the observability check correctly detects whether the LLM's own
      prose mentioned the violated rule_id
    - the LLM-call-failure fallback path also gets the alert prepended,
      since a breach existing is exactly when this matters most
"""
 
from __future__ import annotations
 
import json
from unittest.mock import AsyncMock, patch
 
import pytest
 
from orchestrator.nodes.synthesise import (
    synthesise,
    _build_compliance_alert,
    _recommendation_mentions_violations,
)
from orchestrator.state import (
    AgentState,
    Portfolio,
    ComplianceResult,
    Violation,
)
 
 
def _make_state(compliance_result=None) -> AgentState:
    return AgentState(
        query="Analyse my portfolio.",
        portfolio=Portfolio(
            holdings={"RELIANCE.NS": 0.6, "TCS.NS": 0.4},
            total_value=1_000_000.0,
        ),
        compliance_result=compliance_result,
    )
 
 
def _breach_result() -> ComplianceResult:
    return ComplianceResult(
        passed=False,
        violations=[
            Violation(
                rule_id="SECTOR_CAP",
                description="Sector 'Financial Services' weight 60.0% exceeds sector cap of 40.0%",
                severity="hard",
                value=0.60,
                limit=0.40,
            ),
        ],
        warnings=[],
        rules_version="v1.0",
        rules_profile="retail_conservative",
    )
 
 
def _passing_result() -> ComplianceResult:
    return ComplianceResult(
        passed=True,
        violations=[],
        warnings=[],
        rules_version="v1.0",
        rules_profile="retail_conservative",
    )
 
 
def _mock_llm_response(text: str):
    mock_message = AsyncMock()
    mock_message.content = text
    mock_choice = AsyncMock()
    mock_choice.message = mock_message
    mock_response = AsyncMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = AsyncMock()
    mock_response.usage.total_tokens = 100
    return mock_response
 
 
class TestComplianceAlertPrepended:
    @pytest.mark.asyncio
    async def test_alert_prepended_even_when_llm_ignores_the_breach(self):
        """
        The core guarantee: even if the LLM's own prose says something
        reassuring and never mentions the violation at all, the
        deterministic alert must still be prepended. This is what makes
        it a guardrail rather than a hope.
        """
        state = _make_state(compliance_result=_breach_result())
        mock_response = _mock_llm_response(
            "RECOMMENDATION: Your portfolio looks well-positioned. Consider "
            "holding your current allocation for the next quarter."
        )
 
        with patch(
            "orchestrator.nodes.synthesise._openai.chat.completions.create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await synthesise(state)
 
        assert result["final_recommendation"].startswith(
            "⚠️ SYSTEM COMPLIANCE ALERT"
        )
        assert "SECTOR_CAP" in result["final_recommendation"]
        assert "well-positioned" in result["final_recommendation"]  # LLM text still present, just after the alert
 
    @pytest.mark.asyncio
    async def test_observability_error_logged_when_llm_omits_violation(self):
        state = _make_state(compliance_result=_breach_result())
        mock_response = _mock_llm_response("Everything looks fine, no concerns.")
 
        with patch(
            "orchestrator.nodes.synthesise._openai.chat.completions.create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await synthesise(state)
 
        assert any(
            "did not explicitly mention" in e for e in result["errors"]
        )
 
    @pytest.mark.asyncio
    async def test_no_observability_error_when_llm_does_mention_it(self):
        state = _make_state(compliance_result=_breach_result())
        mock_response = _mock_llm_response(
            "COMPLIANCE STATUS: SECTOR_CAP violation detected. RECOMMENDATION: rebalance."
        )
 
        with patch(
            "orchestrator.nodes.synthesise._openai.chat.completions.create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await synthesise(state)
 
        assert not any("did not explicitly mention" in e for e in result["errors"])
        # Alert is still prepended regardless — the guarantee doesn't relax
        # just because the model happened to get it right this time.
        assert result["final_recommendation"].startswith("⚠️ SYSTEM COMPLIANCE ALERT")
 
 
class TestNoAlertWhenNotNeeded:
    @pytest.mark.asyncio
    async def test_no_alert_when_compliance_passed(self):
        state = _make_state(compliance_result=_passing_result())
        mock_response = _mock_llm_response("RECOMMENDATION: portfolio is compliant, no action needed.")
 
        with patch(
            "orchestrator.nodes.synthesise._openai.chat.completions.create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await synthesise(state)
 
        assert "SYSTEM COMPLIANCE ALERT" not in result["final_recommendation"]
        assert result["final_recommendation"] == "RECOMMENDATION: portfolio is compliant, no action needed."
 
    @pytest.mark.asyncio
    async def test_no_crash_when_compliance_result_is_none(self):
        state = _make_state(compliance_result=None)
        mock_response = _mock_llm_response("RECOMMENDATION: insufficient data.")
 
        with patch(
            "orchestrator.nodes.synthesise._openai.chat.completions.create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await synthesise(state)
 
        assert "SYSTEM COMPLIANCE ALERT" not in result["final_recommendation"]
 
 
class TestFallbackPathGetsAlertToo:
    @pytest.mark.asyncio
    async def test_llm_failure_with_breach_still_prepends_alert(self):
        state = _make_state(compliance_result=_breach_result())
 
        with patch(
            "orchestrator.nodes.synthesise._openai.chat.completions.create",
            new=AsyncMock(side_effect=RuntimeError("API unavailable")),
        ):
            result = await synthesise(state)
 
        assert result["final_recommendation"].startswith("⚠️ SYSTEM COMPLIANCE ALERT")
        assert "SECTOR_CAP" in result["final_recommendation"]
        assert result["execution_trace"] == ["synthesise:error"]
 
    @pytest.mark.asyncio
    async def test_llm_failure_without_breach_no_alert(self):
        state = _make_state(compliance_result=_passing_result())
 
        with patch(
            "orchestrator.nodes.synthesise._openai.chat.completions.create",
            new=AsyncMock(side_effect=RuntimeError("API unavailable")),
        ):
            result = await synthesise(state)
 
        assert "SYSTEM COMPLIANCE ALERT" not in result["final_recommendation"]
 
 
class TestHelperFunctionsDirectly:
    def test_build_compliance_alert_includes_all_violations(self):
        alert = _build_compliance_alert(_breach_result())
        assert "SECTOR_CAP" in alert
        assert "0.600" in alert  # value
        assert "0.400" in alert  # limit
 
    def test_mentions_violations_true_case(self):
        cr = _breach_result()
        assert _recommendation_mentions_violations(
            "This breaches SECTOR_CAP rules.", cr
        ) is True
 
    def test_mentions_violations_false_case(self):
        cr = _breach_result()
        assert _recommendation_mentions_violations(
            "Everything is fine.", cr
        ) is False
 
    def test_mentions_violations_case_insensitive(self):
        cr = _breach_result()
        assert _recommendation_mentions_violations(
            "this violates sector_cap", cr
        ) is True
 
