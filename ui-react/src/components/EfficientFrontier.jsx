// src/components/EfficientFrontier.jsx
//
// Tab 5 — 📈 Efficient Frontier
//
// Responsibilities:
//   1. Efficient frontier curve (50 Pareto-optimal points from optimiser)
//   2. Current portfolio point (red circle)
//   3. Max Sharpe portfolio point (green star)
//   4. Risk-free rate horizontal line
//   5. Improvement vector arrow (current → optimal)
//   6. Annotations showing key metrics
//
// Props received from App.js:
//   optimisation  {Object|null}  optimisation result
//   riskMetrics   {Object|null}  risk metrics (for current portfolio position)
//   loading       {boolean}
//
// Note on the frontier curve:
//   The optimiser returns 50 frontier points in optimisation.efficient_frontier.
//   Each point has { volatility, expected_return, sharpe_ratio, weights }.
//   We plot these directly — no approximation needed unlike the Gradio version
//   which approximated a parabola because it didn't receive frontier points.
//   React gets the full API response so we use the real frontier data.

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
  emptyState: {
    color:      C.slate,
    fontSize:   "0.88rem",
    fontFamily: "Inter, sans-serif",
    textAlign:  "center",
    padding:    "60px 20px",
  },
  metricsRow: {
    display:             "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap:                 "12px",
    marginTop:           "16px",
  },
  metricCard: (color) => ({
    backgroundColor: color + "11",
    border:          `1px solid ${color}33`,
    borderRadius:    "8px",
    padding:         "12px 16px",
    textAlign:       "center",
  }),
  metricLabel: {
    color:        C.slate,
    fontSize:     "0.72rem",
    fontFamily:   "Inter, sans-serif",
    marginBottom: "4px",
  },
  metricValue: (color) => ({
    color:      color,
    fontSize:   "1.1rem",
    fontWeight: 700,
    fontFamily: "IBM Plex Mono, monospace",
  }),
};

export default function EfficientFrontier({ optimisation, riskMetrics, loading }) {

  // ── Derived values ─────────────────────────────────────────────────────────
  // useMemo MUST be before any early return — React hooks rules.
  // Only recomputes when optimisation or riskMetrics changes.
  // Equivalent to Python's @functools.lru_cache for component renders.
  const chartData = useMemo(() => {
    if (!optimisation || !riskMetrics) return null;

    const rfr = riskMetrics.risk_free_rate;

    // Current portfolio position
    // Compute weighted average volatility from per-asset volatilities
    const currVol = Math.sqrt(
      Object.entries(riskMetrics.volatility).reduce((sum, [, v]) => sum + v * v, 0)
        / Object.keys(riskMetrics.volatility).length
    );
    const currRet = riskMetrics.portfolio_return;

    // Optimal portfolio position
    const optVol = optimisation.portfolio_volatility;
    const optRet = optimisation.expected_return;

    // Efficient frontier points from optimiser
    // Each point: { volatility, expected_return, sharpe_ratio }
    const frontier = optimisation.efficient_frontier || [];
    const fVols    = frontier.map(p => p.volatility);
    const fRets    = frontier.map(p => p.expected_return);

    // Y-axis range — pad above and below for breathing room
    const allRets  = [...fRets, currRet, optRet, rfr];
    const yMin     = Math.min(...allRets) - 0.05;
    const yMax     = Math.max(...allRets) + 0.05;

    // X-axis range
    const allVols  = [...fVols, currVol, optVol];
    const xMin     = Math.max(0, Math.min(...allVols) - 0.03);
    const xMax     = Math.max(...allVols) + 0.05;

    return { rfr, currVol, currRet, optVol, optRet, fVols, fRets, yMin, yMax, xMin, xMax };
  }, [optimisation, riskMetrics]);

  // ── Empty / loading state ──────────────────────────────────────────────────
  if (loading || !optimisation || !riskMetrics || !chartData) {
    return (
      <div style={styles.container}>
        <div style={{ ...styles.card, ...styles.emptyState }}>
          {loading
            ? "⏳ Computing efficient frontier…"
            : "Run a FULL or OPTIMISATION analysis to see the efficient frontier"}
        </div>
      </div>
    );
  }

  const { rfr, currVol, currRet, optVol, optRet, fVols, fRets, yMin, yMax, xMin, xMax } = chartData;

  // ── Plotly traces ──────────────────────────────────────────────────────────

  const data = [
    // Risk-free rate — horizontal dashed line
    {
      type: "scatter",
      mode: "lines",
      name: `Risk-free rate (${(rfr * 100).toFixed(1)}%)`,
      x:    [xMin, xMax],
      y:    [rfr, rfr],
      line: { color: C.slate, dash: "dash", width: 1 },
      hovertemplate: `Risk-free rate: ${(rfr * 100).toFixed(1)}%<extra></extra>`,
    },

    // Efficient frontier curve
    {
      type: "scatter",
      mode: "lines",
      name: "Efficient Frontier",
      x:    fVols,
      y:    fRets,
      line: { color: C.blue, width: 2.5 },
      hovertemplate:
        "Volatility: %{x:.1%}<br>Return: %{y:.1%}<extra>Frontier</extra>",
    },

    // Current portfolio — red circle
    {
      type:   "scatter",
      mode:   "markers+text",
      name:   `Current (ret=${(currRet * 100).toFixed(1)}%, vol=${(currVol * 100).toFixed(1)}%)`,
      x:      [currVol],
      y:      [currRet],
      marker: {
        color:  C.red,
        size:   16,
        symbol: "circle",
        line:   { color: C.white, width: 2 },
      },
      text:          ["Current"],
      textposition:  "top right",
      textfont:      { color: C.red, size: 12 },
      hovertemplate: `Current Portfolio<br>Return: ${(currRet * 100).toFixed(2)}%<br>Vol: ${(currVol * 100).toFixed(2)}%<extra></extra>`,
    },

    // Max Sharpe portfolio — green star
    {
      type:   "scatter",
      mode:   "markers+text",
      name:   `Max Sharpe (ret=${(optRet * 100).toFixed(1)}%, vol=${(optVol * 100).toFixed(1)}%)`,
      x:      [optVol],
      y:      [optRet],
      marker: {
        color:  C.green,
        size:   20,
        symbol: "star",
        line:   { color: C.white, width: 2 },
      },
      text:          ["Max Sharpe"],
      textposition:  "top right",
      textfont:      { color: C.green, size: 12 },
      hovertemplate: `Max Sharpe Portfolio<br>Return: ${(optRet * 100).toFixed(2)}%<br>Vol: ${(optVol * 100).toFixed(2)}%<br>Sharpe: ${optimisation.sharpe_ratio.toFixed(3)}<extra></extra>`,
    },
  ];

  // ── Layout ─────────────────────────────────────────────────────────────────
  const layout = {
    ...BASE_LAYOUT,
    title:  "Efficient Frontier — Risk-Return Space",
    height: 560,
    xaxis: {
      ...BASE_LAYOUT.xaxis,
      title:      "Annualised Volatility",
      type:       "linear",
      tickformat: ".0%",
      range:      [xMin, xMax],
    },
    yaxis: {
      ...BASE_LAYOUT.yaxis,
      title:      "Annualised Expected Return",
      type:       "linear",
      tickformat: ".1%",
      range:      [yMin, yMax],
    },
    // Arrow annotation — improvement vector from current to optimal
    annotations: [
      {
        x:          optVol,
        y:          optRet,
        ax:         currVol,
        ay:         currRet,
        xref:       "x",
        yref:       "y",
        axref:      "x",
        ayref:      "y",
        arrowhead:  2,
        arrowsize:  1.3,
        arrowcolor: C.amber,
        arrowwidth: 2,
        showarrow:  true,
        text:       "",
      },
    ],
  };

  const config = {
    responsive:             true,
    displayModeBar:         true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  // ── Key metrics comparison ─────────────────────────────────────────────────
  const sharpeImprovement = optimisation.sharpe_ratio - (riskMetrics.sharpe_ratio);
  const returnImprovement = optRet - currRet;
  const volChange         = optVol - currVol;

  return (
    <div style={styles.container}>

      {/* Main chart */}
      <div style={styles.card}>
        <div style={styles.cardTitle}>Risk-Return Space — Where do you sit?</div>
        <Plot
          data={data}
          layout={layout}
          config={config}
          style={{ width: "100%" }}
          useResizeHandler={true}
        />
      </div>

      {/* Key metrics comparison */}
      <div style={styles.card}>
        <div style={styles.cardTitle}>Improvement from Current → Max Sharpe</div>
        <div style={styles.metricsRow}>

          <div style={styles.metricCard(C.blue)}>
            <div style={styles.metricLabel}>Sharpe Improvement</div>
            <div style={styles.metricValue(sharpeImprovement > 0 ? C.green : C.red)}>
              {sharpeImprovement > 0 ? "+" : ""}{sharpeImprovement.toFixed(3)}
            </div>
          </div>

          <div style={styles.metricCard(C.blue)}>
            <div style={styles.metricLabel}>Return Change</div>
            <div style={styles.metricValue(returnImprovement > 0 ? C.green : C.red)}>
              {returnImprovement > 0 ? "+" : ""}{(returnImprovement * 100).toFixed(2)}%
            </div>
          </div>

          <div style={styles.metricCard(C.blue)}>
            <div style={styles.metricLabel}>Volatility Change</div>
            <div style={styles.metricValue(volChange < 0 ? C.green : C.amber)}>
              {volChange > 0 ? "+" : ""}{(volChange * 100).toFixed(2)}%
            </div>
          </div>

        </div>

        {/* Note about frontier */}
        <div style={{
          marginTop:  "12px",
          fontSize:   "0.75rem",
          color:      C.slate,
          fontFamily: "Inter, sans-serif",
        }}>
          Note: Frontier computed from current portfolio symbols only (v1).
          v2 will expand to full NSE universe for a true Capital Market Line.
        </div>
      </div>

    </div>
  );
}
