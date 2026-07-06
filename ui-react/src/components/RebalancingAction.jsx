// src/components/RebalancingAction.jsx
//
// Tab 4 — ⚖️ Rebalancing Action
//
// Responsibilities:
//   1. Side-by-side horizontal bar chart: current vs optimal weights
//   2. Weight delta chart: optimal minus current
//   3. Summary table: what to buy and sell
//
// Props received from App.js:
//   optimisation  {Object|null}  optimisation result with optimal_weights
//   portfolio     {Object}       current holdings {symbol: weight}
//   loading       {boolean}      true while API call is in progress

import Plot from "react-plotly.js";
import { C, BASE_LAYOUT } from "../constants";

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = {
  container: {
    padding: "24px",
  },
  card: {
    backgroundColor: C.card,
    borderRadius:    "10px",
    padding:         "20px 24px",
    border:          `1px solid ${C.blue}22`,
    marginBottom:    "20px",
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
  emptyState: {
    color:      C.slate,
    fontSize:   "0.88rem",
    fontFamily: "Inter, sans-serif",
    textAlign:  "center",
    padding:    "60px 20px",
  },
  table: {
    width:          "100%",
    borderCollapse: "collapse",
    fontSize:       "0.85rem",
    fontFamily:     "IBM Plex Mono, monospace",
  },
  th: {
    textAlign:    "left",
    color:        C.blue,
    padding:      "8px 12px",
    borderBottom: `1px solid ${C.blue}33`,
    fontSize:     "0.75rem",
    fontFamily:   "Inter, sans-serif",
    fontWeight:   600,
  },
  td: {
    padding:       "8px 12px",
    borderBottom:  `1px solid #1E2D4E55`,
    color:         C.white,
    verticalAlign: "middle",
  },
  actionBadge: (delta) => ({
    display:         "inline-block",
    padding:         "2px 10px",
    borderRadius:    "4px",
    fontSize:        "0.75rem",
    fontWeight:      700,
    backgroundColor: delta > 1  ? C.green + "22"
                   : delta < -1 ? C.red   + "22"
                   : C.slate    + "22",
    color:           delta > 1  ? C.green
                   : delta < -1 ? C.red
                   : C.slate,
    border: `1px solid ${
      delta > 1  ? C.green + "44"
    : delta < -1 ? C.red   + "44"
    : C.slate    + "44"
    }`,
  }),
};

// ── Main component ─────────────────────────────────────────────────────────────
export default function RebalancingAction({ optimisation, portfolio, loading }) {

  // ── Empty / loading state ──────────────────────────────────────────────────
  if (loading || !optimisation) {
    return (
      <div style={styles.container}>
        <div style={{ ...styles.card, ...styles.emptyState }}>
          {loading
            ? "⏳ Running portfolio optimisation…"
            : "Run a FULL or OPTIMISATION analysis to see rebalancing recommendations"}
        </div>
      </div>
    );
  }

  // ── Prepare data ───────────────────────────────────────────────────────────
  // currentW: all symbols with non-zero current weight
  const currentW = Object.fromEntries(
    Object.entries(portfolio).filter(([, w]) => w > 0)
  );

  // optimalW: only symbols with meaningful optimal weight (>5% threshold)
  // Filters out solver artefacts like 6.68e-15 or 0.002
  const optimalW = Object.fromEntries(
    Object.entries(optimisation.optimal_weights).filter(([, w]) => w > 0.05)
  );

  // Union of symbols — include if current > 1% OR optimal > 5%
  // This shows all stocks we currently hold plus all recommended positions
  // Excludes symbols with zero/negligible weight in both
  const symbols = [...new Set([
    ...Object.keys(currentW),
    ...Object.keys(optimalW),
  ])].sort().filter(s =>
    (currentW[s] || 0) > 0.01 || (optimalW[s] || 0) > 0.05
  );

  const currentVals = symbols.map(s => (currentW[s] || 0) * 100);
  const optimalVals = symbols.map(s => (optimalW[s] || 0) * 100);
  const deltas      = symbols.map((s, i) => 
    Math.round((optimalVals[i] - currentVals[i]) * 10) /10 );
  const deltaColors = deltas.map(d => d >= 0 ? C.green : C.red);

  // Short symbol names for display (remove .NS suffix)
  const shortSymbols = symbols.map(s => s.replace(".NS", ""));

  // Dynamic x-axis range — tight fit around actual data
  const maxVal = Math.max(...currentVals, ...optimalVals);
  const xMax   = Math.ceil(maxVal / 10) * 10 + 15;  // round up to next 10, add 15pp padding for labels

  // ── Plotly traces ──────────────────────────────────────────────────────────
  const data = [
    {
      type:          "bar",
      orientation:   "h",
      name:          "Current",
      y:             shortSymbols,
      x:             currentVals,
      marker:        { color: C.red, opacity: 0.8 },
      // Only show label if value is meaningful — suppresses 0.0% on exit positions
      text:          currentVals.map(v => v >= 1 ? `${v.toFixed(1)}%` : ""),
      textposition:  "outside",
      cliponaxis:    false,
      textfont:      { color: C.white, size: 11 },
    },
    {
      type:          "bar",
      orientation:   "h",
      name:          "Optimal",
      y:             shortSymbols,
      x:             symbols.map(s => optimalW[s] ? (optimalW[s] * 100) : 0),
      marker:        { color: C.green, opacity: 0.8 },
      // Only show label if value is meaningful — suppresses 0.0% on exit positions
      text:          optimalVals.map(v => v >= 1 ? `${v.toFixed(1)}%` : ""),
      textposition:  "outside",
      cliponaxis:    false,
      textfont:      { color: C.white, size: 11 },
    },
  ];

  const deltaData = [
    {
      type:          "bar",
      orientation:   "h",
      name:          "Delta",
      y:             shortSymbols,
      x:             deltas,
      base:          0,     // <- anchor all bars at zero
      marker:        { color: deltaColors, opacity: 0.9 },
      text:          deltas.map(d => `${d > 0 ? "+" : ""}${d.toFixed(1)}%`),
      textposition:  "outside",
      cliponaxis:    false,
      textfont:      { color: C.white, size: 11 },
    },
  ];

  // ── Plotly layout ──────────────────────────────────────────────────────────
  const layout = {
    ...BASE_LAYOUT,
    title:   "Current vs Optimal Portfolio Weights",
    barmode: "group",
    height:  Math.max(350, symbols.length * 70 + 120),
    xaxis: {
      ...BASE_LAYOUT.xaxis,
      ticksuffix: "%",
      type:       "linear",
      range:      [0, xMax],
      dtick:      10,
      autorange:  false,
      fixedrange: true,   // prevent user zoom from breaking range
    },
    yaxis:  { ...BASE_LAYOUT.yaxis, type: "category", automargin: true },
    legend: { ...BASE_LAYOUT.legend, orientation: "h", y: -0.15 },
    margin: { ...BASE_LAYOUT.margin, l: 90, r: 60 },
  };

  // Delta chart x-axis: symmetric around zero, padded to fit labels
  const maxDelta  = Math.max(...deltas.map(Math.abs));
  const deltaXMax = Math.ceil(maxDelta / 10) * 10 + 15;

  const deltaLayout = {
    ...BASE_LAYOUT,
    title:   "Weight Delta (Optimal − Current)",
    height:  Math.max(300, symbols.length * 60 + 120),
    xaxis: {
      ...BASE_LAYOUT.xaxis,
      ticksuffix:    "%",
      type:          "linear",
      range:         [-deltaXMax, deltaXMax],
      dtick:         10,
      autorange:     false,
      fixedrange:    true,
      zeroline:      true,
      zerolinecolor: C.blue,
      zerolinewidth: 2,
      tickvals:      [-deltaXMax, -deltaXMax/2, 0, deltaXMax/2, deltaXMax]
    },
    yaxis:      { ...BASE_LAYOUT.yaxis, type: "category", automargin: true },
    margin:     { ...BASE_LAYOUT.margin, l: 90, r: 60 },
    showlegend: false,
  };

  const config = {
    responsive:             true,
    displayModeBar:         true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  return (
    <div style={styles.container}>

      {/* Weight comparison chart */}
      <div style={styles.card}>
        <div style={styles.cardTitle}>Portfolio Weights — Current vs Optimal</div>
        <Plot
          data={data}
          layout={layout}
          config={config}
          style={{ width: "100%" }}
          useResizeHandler={true}
        />
      </div>

      {/* Delta chart */}
      <div style={styles.card}>
        <div style={styles.cardTitle}>Rebalancing Actions Required</div>
        <Plot
          data={deltaData}
          layout={deltaLayout}
          config={config}
          style={{ width: "100%" }}
          useResizeHandler={true}
        />
      </div>

      {/* Action summary table */}
      <div style={styles.card}>
        <div style={styles.cardTitle}>Action Summary</div>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Symbol</th>
              <th style={{ ...styles.th, textAlign: "right" }}>Current</th>
              <th style={{ ...styles.th, textAlign: "right" }}>Optimal</th>
              <th style={{ ...styles.th, textAlign: "right" }}>Change</th>
              <th style={styles.th}>Action</th>
            </tr>
          </thead>
          <tbody>
            {symbols.map((symbol, i) => {
              const delta  = deltas[i];
              const action = delta > 1  ? "BUY / INCREASE"
                           : delta < -1 ? "SELL / REDUCE"
                           : "HOLD";
              return (
                <tr key={symbol}>
                  <td style={styles.td}>{symbol}</td>
                  <td style={{ ...styles.td, textAlign: "right" }}>
                    {currentVals[i].toFixed(1)}%
                  </td>
                  <td style={{ ...styles.td, textAlign: "right", color: C.green }}>
                    {optimalVals[i].toFixed(1)}%
                  </td>
                  <td style={{
                    ...styles.td,
                    textAlign: "right",
                    color: delta > 0 ? C.green : delta < 0 ? C.red : C.slate,
                  }}>
                    {delta > 0 ? "+" : ""}{delta.toFixed(1)}%
                  </td>
                  <td style={styles.td}>
                    <span style={styles.actionBadge(delta)}>{action}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        {/* Optimiser metadata */}
        <div style={{
          marginTop:  "12px",
          fontSize:   "0.75rem",
          color:      C.slate,
          fontFamily: "IBM Plex Mono, monospace",
        }}>
          Solver: {optimisation.solver_used} ·
          Expected return: {(optimisation.expected_return * 100).toFixed(2)}% ·
          Volatility: {(optimisation.portfolio_volatility * 100).toFixed(2)}% ·
          Sharpe: {optimisation.sharpe_ratio.toFixed(3)}
        </div>
      </div>

    </div>
  );
}
