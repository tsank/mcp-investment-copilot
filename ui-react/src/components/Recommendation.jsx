// src/components/Recommendation.jsx
//
// Tab 2 — 💡 AI Recommendation
//
// Responsibilities:
//   1. Full-width compliance banner (green COMPLIANT / red BREACH)
//   2. GPT-4o generated recommendation prose
//   3. Execution trace (collapsed by default — for debugging)
//
// Props received from App.js:
//   recommendation  {string|null}  GPT-4o prose from final_recommendation
//   compliance      {Object|null}  compliance result with passed, violations, warnings
//   executionTrace  {Array|null}   node execution trace e.g. ["parse_query:ok", ...]
//   loading         {boolean}      true while API call is in progress
//
// Design:
//   Compliance banner is the signature element — full width, impossible to miss.
//   Green gradient for COMPLIANT, red gradient for BREACH.
//   Recommendation text is plain prose — sections labelled by GPT-4o itself.

import { useState } from "react";
import { C } from "../constants";

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = {
  container: {
    padding: "24px",
    maxWidth: "100%",
  },
  // Compliance banner — full width, prominent
  banner: (passed) => ({
    display:         "flex",
    alignItems:      "center",
    gap:             "16px",
    padding:         "20px 28px",
    borderRadius:    "10px",
    border:          `1.5px solid ${passed ? C.green : C.red}`,
    background:      passed
      ? "linear-gradient(135deg, #064e3b, #065f46)"
      : "linear-gradient(135deg, #450a0a, #7f1d1d)",
    marginBottom:    "24px",
  }),
  bannerIcon: {
    fontSize: "2.2rem",
    flexShrink: 0,
  },
  bannerTitle: (passed) => ({
    color:       passed ? C.green : C.red,
    fontSize:    "1.25rem",
    fontWeight:  700,
    letterSpacing: "2px",
    fontFamily:  "IBM Plex Mono, monospace",
    margin:      0,
  }),
  bannerSubtitle: (passed) => ({
    color:      passed ? "#6ee7b7" : "#fca5a5",
    fontSize:   "0.82rem",
    marginTop:  "4px",
    fontFamily: "IBM Plex Mono, monospace",
  }),
  violation: {
    marginTop:  "6px",
    color:      "#fca5a5",
    fontSize:   "0.82rem",
    fontFamily: "IBM Plex Mono, monospace",
  },
  warning: {
    marginTop:  "4px",
    color:      "#fde68a",
    fontSize:   "0.82rem",
    fontFamily: "IBM Plex Mono, monospace",
  },
  // Recommendation prose card
  proseCard: {
    backgroundColor: C.card,
    borderRadius:    "10px",
    padding:         "24px 28px",
    border:          `1px solid ${C.blue}22`,
    marginBottom:    "16px",
  },
  proseTitle: {
    color:        C.blue,
    fontSize:     "0.82rem",
    fontWeight:   600,
    letterSpacing:"1px",
    marginBottom: "16px",
    fontFamily:   "Inter, sans-serif",
    textTransform:"uppercase",
  },
  // Render each section of the recommendation as a paragraph
  // GPT-4o labels sections: COMPLIANCE STATUS:, RISK ASSESSMENT:, etc.
  prose: {
    color:      C.white,
    fontSize:   "0.92rem",
    lineHeight: 1.75,
    fontFamily: "Inter, sans-serif",
    whiteSpace: "pre-wrap",   // preserve line breaks from GPT-4o
  },
  // Loading placeholder
  loadingCard: {
    backgroundColor: C.card,
    borderRadius:    "10px",
    padding:         "40px",
    textAlign:       "center",
    border:          `1px solid ${C.blue}22`,
    color:           C.slate,
    fontFamily:      "Inter, sans-serif",
    fontSize:        "0.9rem",
  },
  // Empty state — before first analysis
  emptyState: {
    backgroundColor: C.card,
    borderRadius:    "10px",
    padding:         "60px 40px",
    textAlign:       "center",
    border:          `1px dashed ${C.blue}44`,
    color:           C.slate,
    fontFamily:      "Inter, sans-serif",
  },
  emptyIcon: {
    fontSize:     "2.5rem",
    marginBottom: "12px",
  },
  emptyText: {
    fontSize: "0.95rem",
    color:    C.slate,
  },
  // Execution trace — collapsed by default
  traceToggle: {
    background:  "none",
    border:      `1px solid ${C.blue}44`,
    color:       C.slate,
    borderRadius:"6px",
    padding:     "6px 14px",
    cursor:      "pointer",
    fontSize:    "0.78rem",
    fontFamily:  "IBM Plex Mono, monospace",
    marginTop:   "12px",
  },
  traceBox: {
    backgroundColor: "#0D1628",
    borderRadius:    "6px",
    padding:         "12px 16px",
    marginTop:       "8px",
    fontFamily:      "IBM Plex Mono, monospace",
    fontSize:        "0.78rem",
    color:           C.slate,
    border:          `1px solid ${C.blue}22`,
  },
};

// ── ComplianceBanner subcomponent ──────────────────────────────────────────────
// Extracted as a named function (not exported) — only used inside this file.
// Equivalent to a private method in Python.

function ComplianceBanner({ compliance }) {
  if (!compliance) return null;

  const { passed, violations = [], warnings = [], rules_profile, rules_version } = compliance;

  return (
    <div style={styles.banner(passed)}>
      <span style={styles.bannerIcon}>{passed ? "✅" : "❌"}</span>
      <div>
        <h2 style={styles.bannerTitle(passed)}>
          {passed ? "COMPLIANT" : "COMPLIANCE BREACH"}
        </h2>
        <div style={styles.bannerSubtitle(passed)}>
          {rules_profile} {rules_version}
          {" — "}
          {violations.length} violation{violations.length !== 1 ? "s" : ""}
          {" · "}
          {warnings.length} warning{warnings.length !== 1 ? "s" : ""}
        </div>

        {/* Violations list */}
        {violations.map((v, i) => (
          <div key={i} style={styles.violation}>
            ⛔ {v.rule_id}: {v.description}
          </div>
        ))}

        {/* Warnings list */}
        {warnings.map((w, i) => (
          <div key={i} style={styles.warning}>
            ⚠️ {w.rule_id}: {w.description}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function Recommendation({ recommendation, compliance, executionTrace, loading }) {

  // Local state — trace visibility toggle
  // Only this component needs to know if trace is expanded
  const [showTrace, setShowTrace] = useState(false);

  // ── Loading state ──────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loadingCard}>
          <div style={{ fontSize: "2rem", marginBottom: "12px" }}>⏳</div>
          <div>Running LangGraph pipeline…</div>
          <div style={{ fontSize: "0.80rem", marginTop: "8px", color: C.slate }}>
            parse_query → fetch_market_data → compute_risk → optimise → simulate → check_compliance → synthesise
          </div>
        </div>
      </div>
    );
  }

  // ── Empty state — before first analysis ───────────────────────────────────
  if (!recommendation) {
    return (
      <div style={styles.container}>
        <div style={styles.emptyState}>
          <div style={styles.emptyIcon}>💡</div>
          <div style={styles.emptyText}>
            Submit your portfolio in the <strong>My Portfolio</strong> tab to see the AI recommendation.
          </div>
        </div>
      </div>
    );
  }

  // ── Result state ───────────────────────────────────────────────────────────
  return (
    <div style={styles.container}>

      {/* Compliance banner — always first */}
      <ComplianceBanner compliance={compliance} />

      {/* Recommendation prose */}
      <div style={styles.proseCard}>
        <div style={styles.proseTitle}>Investment Recommendation</div>
        <div style={styles.prose}>{recommendation}</div>
      </div>

      {/* Execution trace — collapsed by default */}
      {executionTrace && executionTrace.length > 0 && (
        <>
          <button
            style={styles.traceToggle}
            onClick={() => setShowTrace(prev => !prev)}
          >
            {showTrace ? "▲ Hide" : "▼ Show"} execution trace ({executionTrace.length} nodes)
          </button>

          {showTrace && (
            <div style={styles.traceBox}>
              {executionTrace.map((step, i) => {
                // Colour code trace steps by status
                const color = step.includes(":ok")      ? C.green
                            : step.includes(":error")   ? C.red
                            : step.includes(":skipped") ? C.amber
                            : step.includes(":partial") ? C.amber
                            : C.slate;
                return (
                  <div key={i} style={{ color, marginBottom: "2px" }}>
                    {i + 1}. {step}
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}

    </div>
  );
}
