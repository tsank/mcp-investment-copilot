// src/components/ComplianceCheck.jsx
//
// Tab 3 — ✅ Compliance Check
//
// Responsibilities:
//   1. Compliance detail — rules profile, violations table, warnings table
//   2. Risk metrics table — CVaR, VaR, Sharpe, MaxDrawdown, Return, volatilities
//
// Props received from App.js:
//   compliance   {Object|null}  compliance result
//   riskMetrics  {Object|null}  risk metrics from compute_risk node
//   loading      {boolean}      true while API call is in progress

import { C } from "../constants";

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = {
  container: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap:     "24px",
    padding: "24px",
  },
  card: {
    backgroundColor: C.card,
    borderRadius:    "10px",
    padding:         "20px 24px",
    border:          `1px solid ${C.blue}22`,
  },
  cardTitle: {
    color:         C.blue,
    fontSize:      "0.80rem",
    fontWeight:    600,
    letterSpacing: "1px",
    textTransform: "uppercase",
    marginBottom:  "16px",
    fontFamily:    "Inter, sans-serif",
  },
  // Compliance status pill
  statusPill: (passed) => ({
    display:         "inline-flex",
    alignItems:      "center",
    gap:             "8px",
    padding:         "8px 16px",
    borderRadius:    "20px",
    backgroundColor: passed ? C.green + "22" : C.red + "22",
    border:          `1px solid ${passed ? C.green : C.red}`,
    color:           passed ? C.green : C.red,
    fontSize:        "0.88rem",
    fontWeight:      700,
    fontFamily:      "IBM Plex Mono, monospace",
    marginBottom:    "16px",
  }),
  rulesInfo: {
    fontSize:   "0.78rem",
    color:      C.slate,
    fontFamily: "IBM Plex Mono, monospace",
    marginBottom: "16px",
  },
  // Table shared styles
  table: {
    width:          "100%",
    borderCollapse: "collapse",
    fontSize:       "0.85rem",
  },
  th: {
    textAlign:    "left",
    color:        C.blue,
    padding:      "8px 10px",
    borderBottom: `1px solid ${C.blue}33`,
    fontSize:     "0.75rem",
    fontFamily:   "Inter, sans-serif",
    fontWeight:   600,
  },
  td: {
    padding:      "8px 10px",
    borderBottom: `1px solid #1E2D4E55`,
    fontFamily:   "IBM Plex Mono, monospace",
    fontSize:     "0.83rem",
    color:        C.white,
    verticalAlign:"top",
  },
  // Severity badge
  severityBadge: (severity) => ({
    display:         "inline-block",
    padding:         "2px 8px",
    borderRadius:    "4px",
    fontSize:        "0.70rem",
    fontWeight:      700,
    backgroundColor: severity === "hard" ? C.red + "33" : C.amber + "33",
    color:           severity === "hard" ? C.red : C.amber,
    border:          `1px solid ${severity === "hard" ? C.red : C.amber}44`,
    fontFamily:      "IBM Plex Mono, monospace",
  }),
  noViolations: {
    color:      C.green,
    fontSize:   "0.85rem",
    fontFamily: "IBM Plex Mono, monospace",
    padding:    "12px 0",
  },
  // Risk metric value colours
  metricValue: (value, metric) => {
    // Colour code specific metrics
    if (metric === "sharpe_ratio") {
      return value > 0.5 ? C.green : value > 0 ? C.amber : C.red;
    }
    if (metric === "portfolio_return") {
      return value > 0 ? C.green : C.red;
    }
    return C.white;
  },
  sectionDivider: {
    borderTop:    `1px solid ${C.blue}22`,
    marginTop:    "16px",
    paddingTop:   "16px",
  },
  emptyState: {
    color:      C.slate,
    fontSize:   "0.88rem",
    fontFamily: "Inter, sans-serif",
    textAlign:  "center",
    padding:    "40px 20px",
  },
};

// ── Helper — format metric values for display ──────────────────────────────────
function formatMetric(key, value) {
  switch (key) {
    case "cvar_95":
    case "cvar_99":
    case "var_95":
    case "var_99":
    case "max_drawdown":
    case "portfolio_return":
      return `${(value * 100).toFixed(2)}%`;
    case "sharpe_ratio":
      return value.toFixed(3);
    case "risk_free_rate":
      return `${(value * 100).toFixed(1)}%`;
    default:
      return typeof value === "number" ? value.toFixed(4) : String(value);
  }
}

// Human-readable metric names
const METRIC_LABELS = {
  cvar_95:          "CVaR 95% · Daily (historical)",
  cvar_99:          "CVaR 99% · Daily (historical)",
  var_95:           "VaR 95%",
  var_99:           "VaR 99%",
  sharpe_ratio:     "Sharpe Ratio",
  max_drawdown:     "Max Drawdown",
  portfolio_return: "Ann. Return",
  risk_free_rate:   "Risk-Free Rate",
  computation_window: "Window",
};

// Truthful label for the CVaR that actually gates compliance — driven by
// cvar_source from _select_cvar's fallback priority (garch_sim → monte_carlo
// → risk_metrics), so the label can never drift out of sync with what was
// actually computed. This is a DIFFERENT number/method from the Risk Metrics
// table's cvar_95 above (always the 1-day historical figure) whenever the
// source is a simulation.
const CVAR_SOURCE_LABELS = {
  garch_sim: {
    label: "CVaR 95% · 1-year (GARCH-simulated)",
    note:  "252-trading-day forward simulation using fitted GARCH dynamics.",
  },
  monte_carlo: {
    label: "CVaR 95% · 1-year (Monte Carlo)",
    note:  "252-trading-day forward simulation, static distribution — GARCH simulation unavailable.",
  },
  risk_metrics: {
    label: "CVaR 95% · Daily (historical) — no forward simulation available",
    note:  "⚠️ Both simulations were unavailable. This threshold check is comparing a 1-day figure " +
           "against a limit calibrated for annual risk — a breach is far less likely to fire in this " +
           "fallback state than when a simulation is available.",
  },
};

// Order metrics for display — most important first
const METRIC_ORDER = [
  "cvar_95", "cvar_99", "var_95", "var_99",
  "sharpe_ratio", "max_drawdown", "portfolio_return",
  "risk_free_rate", "computation_window",
];

// ── Main component ─────────────────────────────────────────────────────────────
export default function ComplianceCheck({ compliance, riskMetrics, loading }) {

  // ── Empty / loading state ──────────────────────────────────────────────────
  if (loading || (!compliance && !riskMetrics)) {
    return (
      <div style={styles.container}>
        <div style={styles.card}>
          <div style={styles.emptyState}>
            {loading ? "⏳ Running compliance check…" : "Run analysis to see compliance results"}
          </div>
        </div>
        <div style={styles.card}>
          <div style={styles.emptyState}>
            {loading ? "⏳ Computing risk metrics…" : "Run analysis to see risk metrics"}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>

      {/* ── Left: Compliance detail ── */}
      <div style={styles.card}>
        <div style={styles.cardTitle}>Compliance Detail</div>

        {compliance ? (
          <>
            {/* Status pill */}
            <div style={styles.statusPill(compliance.passed)}>
              {compliance.passed ? "✅ COMPLIANT" : "❌ BREACH"}
            </div>

            {/* Rules info */}
            <div style={styles.rulesInfo}>
              Profile: {compliance.rules_profile} · Version: {compliance.rules_version}
            </div>

            {/* Gating CVaR — the value actually checked against CVAR_THRESHOLD.
                Label is driven by cvar_source, never hardcoded, so it cannot
                drift out of sync with what was actually computed. */}
            {compliance.cvar_source && (
              <div style={{ ...styles.rulesInfo, marginTop: "4px" }}>
                {(CVAR_SOURCE_LABELS[compliance.cvar_source] || {}).label ||
                  `CVaR 95% (source: ${compliance.cvar_source})`}
                {": "}
                {(compliance.cvar_95 * 100).toFixed(1)}%
                {CVAR_SOURCE_LABELS[compliance.cvar_source]?.note && (
                  <div style={{ fontSize: "0.7rem", color: C.slate, marginTop: "3px", fontStyle: "italic" }}>
                    {CVAR_SOURCE_LABELS[compliance.cvar_source].note}
                  </div>
                )}
              </div>
            )}

            {/* Violations table */}
            <div style={styles.cardTitle}>Hard Violations</div>
            {compliance.violations.length === 0 ? (
              <div style={styles.noViolations}>✓ No violations</div>
            ) : (
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Rule</th>
                    <th style={styles.th}>Severity</th>
                    <th style={{ ...styles.th, textAlign: "right" }}>Value</th>
                    <th style={{ ...styles.th, textAlign: "right" }}>Limit</th>
                  </tr>
                </thead>
                <tbody>
                  {compliance.violations.map((v, i) => (
                    <tr key={i}>
                      <td style={styles.td}>
                        <div style={{ color: C.red }}>{v.rule_id}</div>
                        <div style={{ fontSize: "0.72rem", color: C.slate, marginTop: "2px" }}>
                          {v.description}
                        </div>
                      </td>
                      <td style={styles.td}>
                        <span style={styles.severityBadge(v.severity)}>
                          {v.severity}
                        </span>
                      </td>
                      <td style={{ ...styles.td, textAlign: "right", color: C.red }}>
                        {(v.value * 100).toFixed(1)}%
                      </td>
                      <td style={{ ...styles.td, textAlign: "right", color: C.slate }}>
                        {(v.limit * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {/* Warnings table */}
            <div style={{ ...styles.sectionDivider }}>
              <div style={styles.cardTitle}>Warnings</div>
              {compliance.warnings.length === 0 ? (
                <div style={styles.noViolations}>✓ No warnings</div>
              ) : (
                <table style={styles.table}>
                  <thead>
                    <tr>
                      <th style={styles.th}>Rule</th>
                      <th style={{ ...styles.th, textAlign: "right" }}>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {compliance.warnings.map((w, i) => (
                      <tr key={i}>
                        <td style={styles.td}>
                          <div style={{ color: C.amber }}>{w.rule_id}</div>
                          <div style={{ fontSize: "0.72rem", color: C.slate, marginTop: "2px" }}>
                            {w.description}
                          </div>
                        </td>
                        <td style={{ ...styles.td, textAlign: "right", color: C.amber }}>
                          {(w.value * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </>
        ) : (
          <div style={styles.emptyState}>Compliance data unavailable</div>
        )}
      </div>

      {/* ── Right: Risk metrics ── */}
      <div style={styles.card}>
        <div style={styles.cardTitle}>Risk Metrics</div>

        {riskMetrics ? (
          <>
            {/* Key metrics table */}
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>Metric</th>
                  <th style={{ ...styles.th, textAlign: "right" }}>Value</th>
                </tr>
              </thead>
              <tbody>
                {METRIC_ORDER
                  .filter(key => riskMetrics[key] !== undefined)
                  .map(key => (
                    <tr key={key}>
                      <td style={{ ...styles.td, color: C.slate }}>
                        {METRIC_LABELS[key] || key}
                      </td>
                      <td style={{
                        ...styles.td,
                        textAlign: "right",
                        color: styles.metricValue(riskMetrics[key], key),
                      }}>
                        {formatMetric(key, riskMetrics[key])}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>

            {/* Asset volatilities */}
            {riskMetrics.volatility && (
              <div style={styles.sectionDivider}>
                <div style={styles.cardTitle}>Asset Volatilities (Ann.)</div>
                <table style={styles.table}>
                  <thead>
                    <tr>
                      <th style={styles.th}>Symbol</th>
                      <th style={{ ...styles.th, textAlign: "right" }}>Volatility</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(riskMetrics.volatility).map(([symbol, vol]) => (
                      <tr key={symbol}>
                        <td style={{ ...styles.td, color: C.slate }}>
                          {symbol.replace(".NS", "")}
                        </td>
                        <td style={{ ...styles.td, textAlign: "right", color: C.blue }}>
                          {(vol * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        ) : (
          <div style={styles.emptyState}>Risk metrics unavailable</div>
        )}
      </div>
    </div>
  );
}
