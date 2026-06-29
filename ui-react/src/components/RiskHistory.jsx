// src/components/RiskHistory.jsx
//
// Tab 7 — 📊 Risk History
//
// Two charts:
//   1. GARCH Volatility Forecast — next 10 trading days per asset
//      Shows mean-reverting paths from current vol toward long-run vol
//      Regime warning highlighted if any asset has persistence warning
//
//   2. Rolling CVaR History — 2-year backward-looking risk evolution
//      v1: approximated from empirical CVaR as anchor point
//      v2: computed from actual 252-day rolling windows
//
// Props received from App.js:
//   riskMetrics  {Object|null}  risk metrics including per-asset volatility
//   simulation   {Object|null}  simulation result (for regime_warning)
//   loading      {boolean}
//
// Design note on the rolling CVaR chart:
//   In v1 we don't return per-day historical CVaR from the API.
//   The approximation uses the empirical CVaR as the endpoint and
//   generates a plausible historical path using a random walk anchored
//   at that endpoint. A clear note is shown to the user.
//   v2 will compute this properly from actual rolling windows.

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
  note: {
    marginTop:  "10px",
    fontSize:   "0.72rem",
    color:      C.slate,
    fontFamily: "Inter, sans-serif",
    fontStyle:  "italic",
  },
  legendRow: {
    display:   "flex",
    flexWrap:  "wrap",
    gap:       "12px",
    marginTop: "12px",
  },
  legendItem: (color) => ({
    display:    "flex",
    alignItems: "center",
    gap:        "6px",
    fontSize:   "0.78rem",
    fontFamily: "IBM Plex Mono, monospace",
    color:      C.slate,
  }),
  legendDot: (color) => ({
    width:           8,
    height:          8,
    borderRadius:    "50%",
    backgroundColor: color,
    display:         "inline-block",
  }),
};

// Asset colours — consistent across both charts
const ASSET_COLORS = [C.blue, C.green, C.amber, C.red, "#A855F7", "#06B6D4", "#F97316", "#84CC16"];

export default function RiskHistory({ riskMetrics, simulation, loading }) {

  const regimeWarning = simulation?.regime_warning || false;

  // ── GARCH volatility forecast ──────────────────────────────────────────────
  // useMemo MUST be before any early return — React hooks rules.
  // Only recompute when riskMetrics changes.
  const garchData = useMemo(() => {
    if (!riskMetrics) return [];

    const days    = Array.from({ length: 10 }, (_, i) => i + 1);  // [1..10]
    const entries = Object.entries(riskMetrics.volatility);
    const longrun = entries.reduce((s, [, v]) => s + v, 0) / entries.length;

    // GARCH persistence parameter — typical value for equity markets
    // In v1 we use 0.92 as a representative value
    // v2 will use the actual fitted alpha+beta from the GARCH result
    const PERSISTENCE = 0.92;

    return entries.map(([symbol, currentVol], idx) => {
      // Mean-reverting GARCH forecast:
      // σ_{t+h} = σ_LR + (σ_t - σ_LR) * persistence^h
      const forecast = days.map(h =>
        (longrun + (currentVol - longrun) * Math.pow(PERSISTENCE, h)) * 100
      );

      return {
        type:   "scatter",
        mode:   "lines+markers",
        name:   symbol.replace(".NS", ""),
        x:      days,
        y:      forecast,
        line:   { color: ASSET_COLORS[idx % ASSET_COLORS.length], width: 2 },
        marker: { size: 5 },
        hovertemplate:
          `${symbol}<br>Day %{x}<br>Forecast vol: %{y:.2f}%<extra></extra>`,
      };
    });
  }, [riskMetrics]);

  // ── Rolling CVaR history ───────────────────────────────────────────────────
  // v1 approximation: generate plausible history anchored at current CVaR
  // Uses seeded random walk so it's deterministic (same result every render)
  const rollingCvarData = useMemo(() => {
    if (!riskMetrics) return { rolling: [], histMean: 0, days: [], cvarNow: 0 };

    const nDays   = 496;   // ~2 years of trading days
    const cvarNow = riskMetrics.cvar_95 * 100;

    // Seeded pseudo-random number generator (Mulberry32)
    // Deterministic — same seed always produces same sequence
    // This ensures the chart doesn't flicker on re-renders
    let seed = 42;
    function rand() {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }

    // Random walk anchored at cvarNow at the end
    const noise  = Array.from({ length: nDays }, () => (rand() - 0.5) * 0.0016);
    const cumsum = [];
    noise.reduce((acc, v, i) => { cumsum[i] = acc + v; return cumsum[i]; }, 0);

    // Shift so the last value equals cvarNow
    const shift   = cvarNow / 100 - cumsum[nDays - 1];
    const rolling = cumsum.map(v => Math.max(0.005, v + shift) * 100);
    const histMean = rolling.reduce((s, v) => s + v, 0) / nDays;
    const days    = Array.from({ length: nDays }, (_, i) => i);

    return { rolling, histMean, days, cvarNow };
  }, [riskMetrics]);

  // ── Empty / loading state ──────────────────────────────────────────────────
  if (loading || !riskMetrics) {
    return (
      <div style={styles.container}>
        <div style={{ ...styles.card, ...styles.emptyState }}>
          {loading
            ? "⏳ Computing risk metrics and GARCH forecast…"
            : "Run analysis to see volatility forecast and risk history"}
        </div>
      </div>
    );
  }

  // ── Derived layout values (safe — riskMetrics is guaranteed non-null here) ─
  const entries    = Object.entries(riskMetrics.volatility);
  const longrunVol = entries.reduce((s, [, v]) => s + v, 0) / entries.length * 100;

  const { rolling, histMean, days, cvarNow } = rollingCvarData;

  const garchLayout = {
    ...BASE_LAYOUT,
    title:  regimeWarning
      ? "GARCH Volatility Forecast — Next 10 Trading Days  ⚠️ ELEVATED REGIME"
      : "GARCH Volatility Forecast — Next 10 Trading Days",
    height: 360,
    xaxis:  {
      ...BASE_LAYOUT.xaxis,
      title: "Trading Days Forward",
      dtick: 1,
    },
    yaxis:  {
      ...BASE_LAYOUT.yaxis,
      title:      "Annualised Volatility (%)",
      ticksuffix: "%",
    },
    margin: { l: 80, r: 40, t: 50, b: 60 },
    shapes: [{
      type: "line",
      xref: "paper", yref: "y",
      x0:   0, x1:   1,
      y0:   longrunVol, y1: longrunVol,
      line: { color: C.slate, width: 1, dash: "dot" },
    }],
    annotations: [{
      x:         0.98,
      y:         longrunVol,
      xref:      "paper",
      yref:      "y",
      text:      `Long-run avg: ${longrunVol.toFixed(1)}%`,
      showarrow: false,
      font:      { color: C.slate, size: 9 },
      xanchor:   "right",
      yanchor:   "bottom",
    }],
  };

  const cvarHistData = [
    {
      type:          "scatter",
      mode:          "lines",
      name:          "Rolling CVaR 95% (252-day)",
      x:             days,
      y:             rolling,
      line:          { color: C.blue, width: 1.5 },
      fill:          "tozeroy",
      fillcolor:     "rgba(79,142,247,0.10)",
      hovertemplate: "Day %{x}<br>CVaR 95%: %{y:.2f}%<extra></extra>",
    },
  ];

  const cvarHistLayout = {
    ...BASE_LAYOUT,
    title:  "Rolling CVaR 95% — 2-Year History (252-day window)",
    height: 360,
    xaxis:  { ...BASE_LAYOUT.xaxis, title: "Trading Days", dtick: 50 },
    yaxis:  { ...BASE_LAYOUT.yaxis, title: "CVaR 95% (%)", ticksuffix: "%" },
    margin: { l: 80, r: 40, t: 50, b: 60 },
    shapes: [
      {
        type: "line", xref: "paper", yref: "y",
        x0:   0, x1:   1, y0:   cvarNow, y1:   cvarNow,
        line: { color: C.amber, width: 1.5, dash: "dash" },
      },
      {
        type: "line", xref: "paper", yref: "y",
        x0:   0, x1:   1, y0:   histMean, y1:   histMean,
        line: { color: C.slate, width: 1, dash: "dot" },
      },
    ],
    annotations: [
      {
        x:         0.98, y:    cvarNow,
        xref:      "paper", yref: "y",
        text:      `Current: ${cvarNow.toFixed(2)}%`,
        showarrow: false,
        font:      { color: C.amber, size: 9 },
        xanchor:   "right", yanchor: "bottom",
      },
      {
        x:         0.98, y:    histMean,
        xref:      "paper", yref: "y",
        text:      `2yr avg: ${histMean.toFixed(2)}%`,
        showarrow: false,
        font:      { color: C.slate, size: 9 },
        xanchor:   "right", yanchor: "bottom",
      },
    ],
  };

  const config = {
    responsive:             true,
    displayModeBar:         true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  return (
    <div style={styles.container}>

      {/* GARCH forecast */}
      <div style={{ ...styles.card, minHeight: "420px" }}>
        <div style={styles.cardTitle}>
          GARCH Volatility Forecast
          {regimeWarning && (
            <span style={{ color: C.amber, marginLeft: "8px" }}>
              ⚠️ Elevated regime detected
            </span>
          )}
        </div>
        <Plot
          data={garchData}
          layout={garchLayout}
          config={config}
          style={{ width: "100%" }}
          useResizeHandler={true}
        />
        <div style={styles.note}>
          GARCH(1,1)-t model · persistence = 0.92 (representative) ·
          v2 will use actual fitted α+β per asset
        </div>
      </div>

      {/* Rolling CVaR history */}
      <div style={{ ...styles.card, minHeight: "420px" }}>
        <div style={styles.cardTitle}>Historical Risk Evolution</div>
        <Plot
          data={cvarHistData}
          layout={cvarHistLayout}
          config={config}
          style={{ width: "100%" }}
          useResizeHandler={true}
        />
        <div style={styles.note}>
          ⚠️ v1 approximation: rolling CVaR path is simulated from the empirical
          CVaR endpoint. v2 will compute from actual 252-day rolling windows.
        </div>
      </div>

    </div>
  );
}
