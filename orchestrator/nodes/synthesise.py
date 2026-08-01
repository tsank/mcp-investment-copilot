"""
orchestrator/nodes/synthesise.py
 
Node 7 of 7 — synthesise
 
Responsibility:
    Generate a structured natural language investment recommendation
    grounded in all computed results in AgentState. Single GPT-4o call
    (Option A synthesis — v1 decision).
 
Reads from AgentState:
    - query               (str)
    - portfolio           (Portfolio)
    - symbols             (list[str])
    - analysis_type       (AnalysisType)
    - market_data         (MarketDataResult)   — may be None
    - risk_metrics        (RiskMetricsResult)  — may be None
    - garch_result        (GARCHResult)         — may be None
    - optimisation_result (OptimisationResult) — may be None
    - simulation_result   (SimulationResult)   — may be None
    - compliance_result   (ComplianceResult)   — may be None
    - errors              (list[str])
 
Writes to AgentState:
    - final_recommendation (str) — structured recommendation prose
 
Design decisions:
 
    Option A — single prompt (v1):
        All result fields serialised into one structured prompt.
        One GPT-4o call produces final_recommendation.
        Option B (section-by-section, 4 LLM calls) deferred to v2
        when the React dashboard needs per-tab summaries.
 
    GPT-4o (not mini):
        synthesise is a generation task — interpreting numbers, forming
        judgements, writing clear prose. GPT-4o-mini is for classification
        (parse_query). GPT-4o is for generation (synthesise).
 
    Partial results handled gracefully:
        Each section of the prompt is only included if the corresponding
        AgentState field is not None. If compliance_result is None,
        the compliance section is omitted. The LLM is instructed to
        note when data is unavailable rather than hallucinate.
 
    Compliance gate surfaced explicitly:
        If compliance_result.passed is False, the recommendation opens
        with a hard warning before any other content. Hard violations
        are never buried in the middle of the response.
 
    regime_warning surfaced explicitly:
        If simulation_result.regime_warning is True, the recommendation
        includes a volatility regime warning prominently.
 
    Errors surfaced:
        If state.errors is non-empty, the recommendation notes that
        some computations failed and results may be partial.
 
    Temperature 0.3:
        Slightly above zero to allow natural prose variation while
        keeping the recommendation factually grounded. Not 0 because
        financial prose at temperature 0 tends to be repetitive.
 
    Output format:
        The LLM is instructed to produce structured sections:
            1. Compliance Status
            2. Risk Assessment
            3. Portfolio Optimisation (if available)
            4. Scenario Analysis (if available)
            5. Recommendation
        Plain prose — no markdown headers in v1 (Gradio renders plain text).
"""
 
from __future__ import annotations
 
import json
import logging
 
from openai import AsyncOpenAI
 
from orchestrator.state import AgentState
 
logger = logging.getLogger(__name__)
 
_openai = AsyncOpenAI()
 
_SYSTEM_PROMPT = """You are a senior portfolio analyst for Indian equity markets.
 
You will receive computed quantitative results for a portfolio analysis request.
Generate a structured investment recommendation grounded strictly in the provided data.
 
Rules:
- Never hallucinate numbers. Only reference figures explicitly provided.
- If a section's data is marked UNAVAILABLE, note this briefly and move on.
- Open with COMPLIANCE STATUS — if there are hard violations, state them first and prominently.
- If regime_warning is True, include a volatility regime warning in the risk section.
- Be direct and specific. Avoid generic financial advice.
- Write in plain prose — no markdown, no bullet points, no headers with # symbols.
- Structure your response with these labelled sections (write the label then a colon):
    COMPLIANCE STATUS, RISK ASSESSMENT, PORTFOLIO OPTIMISATION, SCENARIO ANALYSIS, RECOMMENDATION
- Keep total response under 500 words.
- All monetary values are in INR. All returns are annualised unless stated otherwise."""
 
 
# ── v3 guardrail: compliance cross-check ───────────────────────────────────────
#
# The LLM is instructed (system prompt) to open with COMPLIANCE STATUS and
# state hard violations prominently — but that instruction has no code-level
# guarantee behind it. An LLM call can, on any given invocation, soften,
# bury, or omit a breach even when the compliance data was correctly passed
# in. For a financial recommendation, that's the one failure mode that
# actually matters: not "unhelpful," but "actively wrong."
#
# Fix: when compliance_result.passed is False, a deterministic, non-LLM
# alert is ALWAYS prepended — plain string construction from structured
# violation data, independent of what the model did or didn't say. This
# applies on both the success and the LLM-call-failure fallback path,
# since a breach existing is exactly the moment safety information matters
# most, not less.
#
# A lightweight secondary check (_recommendation_mentions_violations) scans
# the LLM's own text for the violated rule_id strings, purely for
# observability — it never blocks or edits the response, it only logs and
# records an error entry if the model silently dropped a violation, so this
# is traceable rather than invisible.
 
def _build_compliance_alert(compliance_result) -> str:
    """
    Deterministically construct a compliance breach alert from structured
    violation data. No LLM involved — this text is guaranteed correct by
    construction and guaranteed to appear whenever compliance_result.passed
    is False, regardless of what synthesise's own LLM call produces.
    """
    lines = ["⚠️ SYSTEM COMPLIANCE ALERT — this portfolio fails one or more hard rules:"]
    for v in compliance_result.violations:
        lines.append(
            f"  • {v.rule_id} ({v.severity}): value={v.value:.3f}, "
            f"limit={v.limit:.3f} — {v.description}"
        )
    lines.append(
        "Any recommendation below must be read in light of this breach; "
        "it does not override or resolve it."
    )
    return "\n".join(lines)
 
 
def _recommendation_mentions_violations(recommendation: str, compliance_result) -> bool:
    """
    Observability check only — never blocks or edits the response.
 
    Returns True if the LLM's own prose referenced at least one violated
    rule_id. False is logged and recorded in state.errors as a signal that
    the model didn't follow the system prompt's instruction on this call —
    useful for noticing drift over time, not something the user needs to
    see (the deterministic alert already guarantees they see the breach).
    """
    text_upper = recommendation.upper()
    return any(v.rule_id.upper() in text_upper for v in compliance_result.violations)
 
 
async def synthesise(state: AgentState) -> dict:
    """
    LangGraph node — synthesise.
 
    Serialises all AgentState result fields into a structured prompt
    and calls GPT-4o to generate the final investment recommendation.
 
    Args:
        state: Current AgentState. Reads all result fields.
 
    Returns:
        dict with keys: final_recommendation, execution_trace, errors.
    """
    logger.info("synthesise: starting — analysis_type=%s", state.analysis_type.value)
 
    prompt = _build_prompt(state)
    logger.info("synthesise: prompt length=%d chars", len(prompt))
 
    try:
        response = await _openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=800,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
 
        recommendation = response.choices[0].message.content.strip()
        logger.info(
            "synthesise: ok — %d chars, %d tokens used",
            len(recommendation),
            response.usage.total_tokens,
        )
 
    except Exception as exc:
        logger.error("synthesise: LLM call failed — %s", exc)
        fallback = (
            f"Portfolio analysis completed with errors. "
            f"Unable to generate recommendation due to: {exc}. "
            f"Errors encountered: {'; '.join(state.errors) if state.errors else 'none'}."
        )
        errors_out = state.errors + [f"synthesise: {exc}"]
 
        # v3 guardrail: a breach existing is exactly when this matters most —
        # apply the same deterministic alert even on the failure path.
        if state.compliance_result is not None and not state.compliance_result.passed:
            fallback = _build_compliance_alert(state.compliance_result) + "\n\n" + fallback
 
        return {
            "final_recommendation": fallback,
            "execution_trace":      state.execution_trace + ["synthesise:error"],
            "errors":               errors_out,
        }
 
    errors_out = list(state.errors)
 
    # v3 guardrail: compliance cross-check.
    if state.compliance_result is not None and not state.compliance_result.passed:
        if not _recommendation_mentions_violations(recommendation, state.compliance_result):
            logger.warning(
                "synthesise: LLM recommendation did not reference the "
                "compliance violation(s) — deterministic alert prepended "
                "as the safety net regardless."
            )
            errors_out.append(
                "synthesise: LLM recommendation did not explicitly mention "
                "the compliance violation — see prepended system alert"
            )
        recommendation = _build_compliance_alert(state.compliance_result) + "\n\n" + recommendation
 
    return {
        "final_recommendation": recommendation,
        "execution_trace":      state.execution_trace + ["synthesise:ok"],
        "errors":               errors_out,
    }
 
 
# ── Prompt builder ────────────────────────────────────────────────────────────
 
def _build_prompt(state: AgentState) -> str:
    """
    Serialise all AgentState result fields into a structured prompt.
 
    Each section is only included if the corresponding field is not None.
    Fields that are None are noted as UNAVAILABLE so the LLM does not
    attempt to reference them.
    """
    sections: list[str] = []
 
    # ── Query context ─────────────────────────────────────────────────────────
    sections.append(f"USER QUERY: {state.query}")
    sections.append(f"ANALYSIS TYPE: {state.analysis_type.value.upper()}")
    sections.append(
        f"PORTFOLIO: {json.dumps(state.portfolio.holdings)} "
        f"| Total value: INR {state.portfolio.total_value:,.0f}"
    )
 
    # ── Compliance ────────────────────────────────────────────────────────────
    if state.compliance_result is not None:
        cr = state.compliance_result
        viols = [
            f"{v.rule_id} ({v.severity}): value={v.value:.3f} limit={v.limit:.3f} — {v.description}"
            for v in cr.violations
        ]
        warns = [
            f"{w.rule_id}: value={w.value:.3f} — {w.description}"
            for w in cr.warnings
        ]
        sections.append(
            f"COMPLIANCE [{cr.rules_profile} {cr.rules_version}]: "
            f"passed={cr.passed} | "
            f"violations={viols if viols else 'none'} | "
            f"warnings={warns if warns else 'none'}"
        )
    else:
        sections.append("COMPLIANCE: UNAVAILABLE")
 
    # ── Risk metrics ──────────────────────────────────────────────────────────
    if state.risk_metrics is not None:
        rm = state.risk_metrics
        sections.append(
            f"RISK METRICS (empirical, {rm.computation_window}): "
            f"VaR_95={rm.var_95:.4f} | CVaR_95={rm.cvar_95:.4f} | "
            f"VaR_99={rm.var_99:.4f} | CVaR_99={rm.cvar_99:.4f} | "
            f"Sharpe={rm.sharpe_ratio:.3f} | MaxDrawdown={rm.max_drawdown:.4f} | "
            f"AnnualisedReturn={rm.portfolio_return:.4f} | "
            f"RFR={rm.risk_free_rate:.4f}"
        )
        vol_str = " | ".join(f"{s}={v:.4f}" for s, v in rm.volatility.items())
        sections.append(f"ASSET VOLATILITIES (annualised): {vol_str}")
    else:
        sections.append("RISK METRICS: UNAVAILABLE")
 
    # ── GARCH ─────────────────────────────────────────────────────────────────
    if state.garch_result is not None:
        gr = state.garch_result
        regime_str = " | ".join(
            f"{s}={a.regime}(persist={a.alpha_plus_beta:.3f},warn={a.persistence_warning})"
            for s, a in gr.per_asset.items()
        )
        sections.append(
            f"GARCH ({gr.garch_model}, {gr.innovations_used}): "
            f"horizon={gr.horizon_days}d | regimes: {regime_str}"
        )
        port_vol_str = " ".join(f"{v:.4f}" for v in gr.portfolio_vol_forecast[:5])
        sections.append(
            f"PORTFOLIO VOL FORECAST (next 5 of {gr.horizon_days} days): {port_vol_str}"
        )
    else:
        sections.append("GARCH FORECAST: UNAVAILABLE")
 
    # ── Optimisation ──────────────────────────────────────────────────────────
    if state.optimisation_result is not None:
        opt = state.optimisation_result
        sections.append(
            f"OPTIMISATION (solver={opt.solver_used}): "
            f"optimal_weights={json.dumps({k: round(v, 4) for k, v in opt.optimal_weights.items()})} | "
            f"ExpectedReturn={opt.expected_return:.4f} | "
            f"Volatility={opt.portfolio_volatility:.4f} | "
            f"Sharpe={opt.sharpe_ratio:.3f} | "
            f"CML_slope={opt.cml_slope:.3f} | "
            f"frontier_points={len(opt.efficient_frontier)}"
        )
        # Weight delta: current vs optimal
        current = state.portfolio.holdings
        delta = {
            s: round(opt.optimal_weights.get(s, 0) - current.get(s, 0), 4)
            for s in set(list(current.keys()) + list(opt.optimal_weights.keys()))
        }
        sections.append(f"WEIGHT DELTA (optimal - current): {json.dumps(delta)}")
    else:
        sections.append("OPTIMISATION: UNAVAILABLE")
 
    # ── Simulation ────────────────────────────────────────────────────────────
    if state.simulation_result is not None:
        sr = state.simulation_result
        sections.append(f"REGIME WARNING: {sr.regime_warning}")
 
        def sim_str(label: str, s) -> str:
            if s is None:
                return f"{label}: UNAVAILABLE"
            return (
                f"{label} (n={s.n_simulations}, dist={s.distribution_used}): "
                f"CVaR_95={s.cvar_95:.4f} | CVaR_99={s.cvar_99:.4f} | "
                f"VaR_95={s.var_95:.4f} | "
                f"p10={s.percentiles.p10:.4f} p50={s.percentiles.p50:.4f} p90={s.percentiles.p90:.4f}"
            )
 
        sections.append(sim_str("MONTE CARLO (current weights)", sr.monte_carlo))
        sections.append(sim_str("MONTE CARLO (optimal weights)", sr.monte_carlo_optimal))
        sections.append(sim_str("GARCH SIM (current weights)", sr.garch_sim))
        sections.append(sim_str("GARCH SIM (optimal weights)", sr.garch_sim_optimal))
    else:
        sections.append("SIMULATION: UNAVAILABLE")
 
    # ── Errors ────────────────────────────────────────────────────────────────
    if state.errors:
        sections.append(f"COMPUTATION ERRORS: {'; '.join(state.errors)}")
 
    return "\n\n".join(sections)
