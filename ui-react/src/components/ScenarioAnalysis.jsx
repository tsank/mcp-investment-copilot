// src/components/ScenarioAnalysis.jsx
//
// Tab 6 — 🔮 Scenario Analysis
//
// Three charts:
//   1. Fan chart — portfolio value paths over 252 trading days
//      current weights (red) vs optimal weights (green)
//      p10, p25, p50, p75, p90 percentile bands
//
//   2. Return distribution — 1-year terminal return density
//      current vs optimal, with VaR and CVaR vertical lines
//
//   3. CVaR comparison bars — all 4 simulation variants
//      MC current / MC optimal / GARCH current / GARCH optimal
//      with 25% compliance threshold line
//
// Props received from App.js:
//   simulation   {Object|null}  simulation result
//   totalValue   {number}       portfolio value in INR (for fan chart y-axis)
//   loading      {boolean}

import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import { C, BASE_LAYOUT } from "../constants";
import { useMemo } from "react";

const Plot = createPlotlyComponent(Plotly);

const styles = {
  container: { padding: "24px" },
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
  row: {
    display:             "grid",
    gridTemplateColumns: "1fr 1fr",
    gap:                 "20px",
    clear:               "both",
  },
  emptyState: {
    color:      C.slate,
    fontSize:   "0.88rem",
    fontFamily: "Inter, sans-serif",
    textAlign:  "center",
    padding:    "60px 20px",
  },
  regimeWarning: {
    display:         "flex",
    alignItems:      "center",
    gap:             "10px",
    padding:         "10px 16px",
    backgroundColor: C.amber + "22",
    border:          `1px solid ${C.amber}44`,
    borderRadius:    "8px",
    marginBottom:    "16px",
    color:           C.amber,
    fontSize:        "0.85rem",
    fontFamily:      "IBM Plex Mono, monospace",
  },
};

// ── Fan chart builder ──────────────────────────────────────────────────────────
// Builds Plotly traces for the portfolio value fan chart.
// Each "fan" has 5 percentile lines: p10, p25, p50, p75, p90.
// The p50 (median) is the solid line; others are dotted with varying opacity.

function buildFanTraces(simOutput, color, labelPrefix, totalValue) {
  if (!simOutput) return [];

  const days = Array.from({ length: 253 }, (_, i) => i);  // 0 to 252
  const p    = simOutput.percentiles;
  const traces = [];

  // Percentile bands: [name, value, lineWidth, opacity, dash]
  const bands = [
    ["p10", p.p10, 1,   0.15, "dot"],
    ["p25", p.p25, 1,   0.25, "dot"],
    ["p50", p.p50, 2.5, 0.80, "solid"],  // median — solid and bold
    ["p75", p.p75, 1,   0.25, "dot"],
    ["p90", p.p90, 1,   0.15, "dot"],
  ];

  bands.forEach(([pctName, pctVal, width, opacity, dash]) => {
    // Convert terminal return to daily compounded path
    // (1 + annual_return)^(1/252) - 1 = daily return
    const dailyReturn = Math.pow(1 + pctVal, 1 / 252) - 1;
    const values      = days.map(d => totalValue * Math.pow(1 + dailyReturn, d));

    traces.push({
      type:        "scatter",
      mode:        "lines",
      x:           days,
      y:           values,
      opacity:     opacity,
      line:        { color, width, dash },
      showlegend:  pctName === "p50",
      name:        pctName === "p50" ? `${labelPrefix} (median)` : undefined,
      legendgroup: labelPrefix,
      hovertemplate: pctName === "p50"
        ? `Day %{x}<br>${labelPrefix} median: ₹%{y:,.0f}<extra></extra>`
        : `<extra></extra>`,
    });
  });

  return traces;
}

// ── Return distribution builder ────────────────────────────────────────────────
// Approximates the terminal return distribution as a Gaussian
// fitted from the p10/p90 percentiles (2.56 sigma range).

function buildDistTraces(simOutput, color, fillColor, label) {
  if (!simOutput) return [];

  const p   = simOutput.percentiles;
  const std = (p.p90 - p.p10) / 2.56;   // approximate std from percentile range
  const mu  = p.p50;                      // median as approximate mean

  const xMin = p.p10 - 0.15;
  const xMax = p.p90 + 0.15;
  const x    = Array.from({ length: 400 }, (_, i) => xMin + (i / 399) * (xMax - xMin));
  const y    = x.map(xi =>
    Math.exp(-0.5 * Math.pow((xi - mu) / std, 2)) / (std * Math.sqrt(2 * Math.PI))
  );

  return [
    {
      type:          "scatter",
      mode:          "lines",
      name:          label,
      x,
      y,
      line:          { color, width: 2 },
      fill:          "tozeroy",
      fillcolor:     fillColor,
      hovertemplate: "Return: %{x:.1%}<extra>" + label + "</extra>",
    },
  ];
}

// ── Main component ─────────────────────────────────────────────────────────────
export default function ScenarioAnalysis({ simulation, totalValue, loading }) {

  // Destructure safely with optional chaining — null when simulation not yet available.
  // These must be declared before useMemo so dependencies are stable references.
  const monte_carlo         = simulation?.monte_carlo         || null;
  const monte_carlo_optimal = simulation?.monte_carlo_optimal || null;
  const garch_sim           = simulation?.garch_sim           || null;
  const garch_sim_optimal   = simulation?.garch_sim_optimal   || null;
  const regime_warning      = simulation?.regime_warning      || false;

  // ── Fan chart data ─────────────────────────────────────────────────────────
  // useMemo MUST be before any early return — React hooks rules.
  // buildFanTraces already guards on null simOutput so safe to call with nulls.
  const fanData = useMemo(() => [
    ...buildFanTraces(monte_carlo,         C.red,   "Current", totalValue),
    ...buildFanTraces(monte_carlo_optimal, C.green, "Optimal", totalValue),
  ], [monte_carlo, monte_carlo_optimal, totalValue]);

  // ── Return distribution data ───────────────────────────────────────────────
  // buildDistTraces already guards on null simOutput so safe to call with nulls.
  const distData = useMemo(() => [
    ...buildDistTraces(monte_carlo,         C.red,   "rgba(229,83,83,0.08)",  "Current (MC)"),
    ...buildDistTraces(monte_carlo_optimal, C.green, "rgba(16,185,129,0.08)", "Optimal (MC)"),
  ], [monte_carlo, monte_carlo_optimal]);

  // ── Empty / loading state ──────────────────────────────────────────────────
  if (loading || !simulation) {
    return (
      <div style={styles.container}>
        <div style={{ ...styles.card, ...styles.emptyState }}>
          {loading
            ? "⏳ Running Monte Carlo and GARCH simulations (~30 seconds)…"
            : "Run a FULL or SIMULATION analysis to see scenario analysis"}
        </div>
      </div>
    );
  }

  // ── Fan chart layout ───────────────────────────────────────────────────────
  const fanLayout = {
    ...BASE_LAYOUT,
    height: 400,
    xaxis:  { 
        ...BASE_LAYOUT.xaxis, 
        title: { text: "Trading Days" }, 
        type:  "linear",
        dtick: 50,
    },
    yaxis:  {
      ...BASE_LAYOUT.yaxis,
      title:      { text: "Portfolio Value (INR ₹)", standoff: 20 },
      type:       "linear",
      tickformat: ",.0f",
      tickprefix: "₹",
    },
    margin: {
        ...BASE_LAYOUT.margin,
        l: 150,  // room for y-axis label + title
        r: 40,   
        t: 50,   // room for chart title
        b: 70    // room for x-axis title
    },
    shapes: [{
      type: "line",
      xref: "paper", yref: "y",
      x0:   0, x1:   1,
      y0:   totalValue, y1: totalValue,
      line: { color: C.slate, width: 1, dash: "dash" },
    }],
    annotations: [{
      x:         252,
      y:         totalValue,
      xref:      "x",
      yref:      "y",
      text:      `Start: ₹${totalValue.toLocaleString("en-IN")}`,
      showarrow: false,
      font:      { color: C.slate, size: 10 },
      xanchor:   "right",
    }],
  };

  // ── Return distribution layout ─────────────────────────────────────────────
  const distShapes = [];
  if (monte_carlo) {
    distShapes.push(
      { type:"line", x0:-monte_carlo.var_95,  x1:-monte_carlo.var_95,
        y0:0, y1:1, yref:"paper",
        line:{ color:C.red, width:1, dash:"dot" } },
      { type:"line", x0:-monte_carlo.cvar_95, x1:-monte_carlo.cvar_95,
        y0:0, y1:1, yref:"paper",
        line:{ color:C.red, width:1.5, dash:"dash" } },
    );
  }
  if (monte_carlo_optimal) {
    distShapes.push(
      { type:"line", x0:-monte_carlo_optimal.cvar_95, x1:-monte_carlo_optimal.cvar_95,
        y0:0, y1:1, yref:"paper",
        line:{ color:C.green, width:1.5, dash:"dash" } },
    );
  }

  const distLayout = {
    ...BASE_LAYOUT,
    title:  "1-Year Terminal Return",
    height: 320,
    xaxis:  { ...BASE_LAYOUT.xaxis, title: "Portfolio Return", type: "linear", tickformat: ".0%" },
    yaxis:  { ...BASE_LAYOUT.yaxis, title: "Density", type: "linear" },
    legend: { ...BASE_LAYOUT.legend, orientation: "h", x: 0.5, xanchor: "center", y: -0.2 },
    shapes: distShapes,
    margin: { ...BASE_LAYOUT.margin, l: 50, t: 40, r: 20 },
  };

  // ── CVaR comparison bar data ───────────────────────────────────────────────
  const metrics = ["CVaR 95%", "CVaR 99%", "VaR 95%"];

  function getVals(sim) {
    if (!sim) return [0, 0, 0];
    return [sim.cvar_95 * 100, sim.cvar_99 * 100, sim.var_95 * 100];
  }

  const cvarData = [
    { name:"MC Current",    color:C.red,   vals:getVals(monte_carlo) },
    { name:"MC Optimal",    color:C.green, vals:getVals(monte_carlo_optimal) },
    { name:"GARCH Current", color:C.amber, vals:getVals(garch_sim) },
    { name:"GARCH Optimal", color:C.blue,  vals:getVals(garch_sim_optimal) },
  ].filter(d => d.vals.some(v => v > 0)).map(d => ({
    type:         "bar",
    name:         d.name,
    x:            metrics,
    y:            d.vals,
    marker:       { color: d.color, opacity: 0.85 },
    text:         d.vals.map(v => `${v.toFixed(0)}%`),
    textposition: "outside",
    textangle:    -90,
    textfont:     { color: d.color, size: 10 },
    hovertemplate: "%{x}: %{y:.0f}%<extra></extra>",
  }));

  const cvarLayout = {
    ...BASE_LAYOUT,
    title:   "Risk Metrics: MC vs GARCH — Current vs Optimal",
    height:  320,
    barmode: "group",
    xaxis:   { ...BASE_LAYOUT.xaxis, type: "category" },
    yaxis:   { ...BASE_LAYOUT.yaxis, title: "Risk (%)", type: "linear", ticksuffix: "%", range: [0, 60] },
    legend:  { ...BASE_LAYOUT.legend, orientation: "h", x: 0.5, xanchor: "center", y: -0.25 },
    shapes: [{
      type: "line",
      xref: "paper", yref: "y",
      x0:   0, x1:   1,
      y0:   25, y1:  25,
      line: { color: C.white, width: 1.5, dash: "dash" },
    }],
    annotations: [{
      x:         0.98,
      y:         25,
      xref:      "paper",
      yref:      "y",
      text:      "CVaR Limit (25%)",
      showarrow: false,
      font:      { color: C.white, size: 10 },
      yanchor:   "bottom",
      xanchor:   "right",
    }],
    margin: { ...BASE_LAYOUT.margin, l: 45, r: 20, t: 40 },
  };

  const config = {
    responsive:             true,
    displayModeBar:         true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  return (
    <div style={styles.container}>

      {/* Regime warning banner */}
      {regime_warning && (
        <div style={styles.regimeWarning}>
          ⚠️ VOLATILITY REGIME WARNING — GARCH CVaR diverges materially from
          Monte Carlo CVaR. Current volatility regime is elevated above the
          historical average embedded in the Monte Carlo simulation.
        </div>
      )}

      {/* Fan chart — full width */}
      <div style={{ ...styles.card, minHeight: "460px" }}>
        <div style={styles.cardTitle}>
          Portfolio Value — 1-Year Forward Scenarios
        </div>
        <Plot
          data={fanData}
          layout={fanLayout}
          config={config}
          style={{ width: "100%" }}
          useResizeHandler={true}
        />
      </div>

      {/* Distribution + CVaR bars — side by side */}
      <div style={styles.row}>
        <div style={styles.card}>
          <div style={styles.cardTitle}>1-Year Terminal Return Distribution (Monte Carlo)</div>
          <Plot
            data={distData}
            layout={distLayout}
            config={config}
            style={{ width: "100%" }}
            useResizeHandler={true}
          />
        </div>

        <div style={styles.card}>
          <div style={styles.cardTitle}>CVaR Comparison</div>
          <Plot
            data={cvarData}
            layout={cvarLayout}
            config={config}
            style={{ width: "100%" }}
            useResizeHandler={true}
          />
        </div>
      </div>

    </div>
  );
}
