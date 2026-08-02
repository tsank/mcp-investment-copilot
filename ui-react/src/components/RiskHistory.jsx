// src/components/RiskHistory.jsx
//
// Tab 7 — 📊 Risk
//
// Three sections:
//   1. GARCH Volatility Forecast — next N trading days per asset, plotting the
//      backend's real per-asset forecast (each asset's own fitted α+β).
//
//   2. Rolling Risk Evolution — real rolling CVaR over a trailing window,
//      window-configurable (1M / 3M / 1Y), current + optimal portfolios
//      overlaid. Each point is a real trailing-window CVaR computed in the
//      risk engine — no fabrication.
//
//   3. Risk Posture strip — CVaR, vol, max drawdown, Sharpe as current +
//      optimal; CVaR and vol carry a trend arrow vs their rolling mean.
//
// Props received from App.js:
//   riskMetrics         {Object|null}  risk metrics incl. per-asset volatility
//   garchForecast       {Object|null}  real per-asset GARCH forecast
//   rollingRiskCurrent  {Object|null}  rolling CVaR/vol series, current weights
//   rollingRiskOptimal  {Object|null}  rolling CVaR/vol series, optimal weights
//   optimisation        {Object|null}  optimisation result (optimal vol/Sharpe)
//   simulation          {Object|null}  simulation result (for regime_warning)
//   loading             {boolean}

import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import { C, BASE_LAYOUT } from "../constants";
import { useMemo, useState } from "react";

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
  selectorRow: {
    display:        "flex",
    justifyContent: "space-between",
    alignItems:     "center",
    marginBottom:   "12px",
  },
  segGroup: {
    display:      "inline-flex",
    border:       `1px solid ${"#4F8EF7"}44`,
    borderRadius: "8px",
    overflow:     "hidden",
  },
  segBtn: (active) => ({
    padding:        "5px 14px",
    fontSize:       "0.78rem",
    fontFamily:     "IBM Plex Mono, monospace",
    cursor:         "pointer",
    border:         "none",
    borderLeft:     `1px solid #4F8EF722`,
    backgroundColor: active ? "#4F8EF7" : "transparent",
    color:          active ? "#0A0F1E" : "#94A3B8",
  }),
  postureGrid: {
    display:             "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
    gap:                 "12px",
    marginTop:           "4px",
  },
  postureCard: {
    backgroundColor: "#1E2D4E",
    borderRadius:    "8px",
    padding:         "12px 14px",
    fontFamily:      "IBM Plex Mono, monospace",
  },
  postureLabel: { fontSize: "0.72rem", color: "#94A3B8", marginBottom: "4px" },
  postureValue: { fontSize: "1.35rem", color: "#E2E8F0" },
  postureOpt:   { fontSize: "0.74rem", color: "#10B981", marginTop: "4px" },
  postureHeading: {
    color: "#94A3B8", fontSize: "0.78rem", fontWeight: 600,
    letterSpacing: "1px", textTransform: "uppercase",
    fontFamily: "Inter, sans-serif", margin: "4px 0 10px",
  },
};

// Asset colours — consistent across both charts
const ASSET_COLORS = [C.blue, C.green, C.amber, C.red, "#A855F7", "#06B6D4", "#F97316", "#84CC16"];

// Simple average of per-asset annualised vols — a portfolio-level vol proxy
// for the posture card. (Not covariance-adjusted; a display summary only.)
function avgVol(volDict) {
  if (!volDict) return null;
  const vals = Object.values(volDict);
  return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
}

export default function RiskHistory({ riskMetrics, garchForecast, rollingRiskCurrent, rollingRiskOptimal, optimisation, simulation, loading }) {

  // Window selector for the rolling chart: "21" (1M) / "63" (3M) / "252" (1Y).
  // Default 1M — shortest window, most sensitive to volatility clusters.
  const [rollWindow, setRollWindow] = useState("21");

  // Regime badge: prefer the real per-asset GARCH regime/persistence signal;
  // fall back to the simulation-level flag if the forecast is unavailable.
  const anyElevated = garchForecast
    ? Object.values(garchForecast.per_asset).some(
        a => a.regime === "elevated" || a.persistence_warning
      )
    : false;
  const regimeWarning = anyElevated || simulation?.regime_warning || false;

  // ── GARCH volatility forecast ──────────────────────────────────────────────
  // useMemo MUST be before any early return — React hooks rules.
  // Plots the backend's real per-asset forecast directly. Each asset's
  // vol_forecast σ_{T+1}..σ_{T+H} was computed with that asset's OWN fitted
  // persistence (α+β) via _compute_vol_forecast in garch_forecast.py.
  // No recomputation, no shared persistence constant, no frontend GARCH math.
  const garchData = useMemo(() => {
    if (!garchForecast) return [];

    return Object.entries(garchForecast.per_asset).map(([symbol, asset], idx) => {
      const forecast = asset.vol_forecast;              // already annualised (fraction)
      const days = Array.from({ length: forecast.length }, (_, i) => i + 1);

      return {
        type:   "scatter",
        mode:   "lines+markers",
        name:   symbol.replace(".NS", ""),
        x:      days,
        y:      forecast.map(v => v * 100),             // fraction → percent for display
        line:   { color: ASSET_COLORS[idx % ASSET_COLORS.length], width: 2 },
        marker: { size: 5 },
        hovertemplate:
          `${symbol}<br>Day %{x}<br>Forecast vol: %{y:.2f}%` +
          `<br>α+β = ${asset.alpha_plus_beta.toFixed(3)}<extra></extra>`,
      };
    });
  }, [garchForecast]);

  // ── Rolling risk series (real) ─────────────────────────────────────────────
  // Selects the current + optimal rolling CVaR/vol series for the chosen
  // window from the backend payload. No fabrication, no random walk — each
  // point is a real trailing-window CVaR/vol computed in the risk engine.
  // Optimal is present only when optimise ran (full/optimisation queries).
  const rollData = useMemo(() => {
    const cur = rollingRiskCurrent?.windows?.[rollWindow] || null;
    const opt = rollingRiskOptimal?.windows?.[rollWindow] || null;
    return { cur, opt };
  }, [rollingRiskCurrent, rollingRiskOptimal, rollWindow]);

  // Posture-strip trend helper: is the current value above/below its own
  // rolling mean? Returns "up" | "down" | null. Used for CVaR and Vol only.
  const trendVs = (value, mean) => {
    if (value == null || mean == null) return null;
    if (value > mean * 1.02) return "up";
    if (value < mean * 0.98) return "down";
    return "flat";
  };

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
  // Long-run reference: average of the REAL per-asset unconditional vols
  // (each asset reverts to its own longrun_vol; this line is an average
  // reference only). null when the GARCH forecast is unavailable.
  const longrunVol = garchForecast
    ? Object.values(garchForecast.per_asset)
        .reduce((s, a) => s + a.longrun_vol, 0)
        / Object.keys(garchForecast.per_asset).length * 100
    : null;

  const horizonDays = garchForecast?.horizon_days ?? 10;

  const garchLayout = {
    ...BASE_LAYOUT,
    title:  regimeWarning
      ? `GARCH Volatility Forecast — Next ${horizonDays} Trading Days  ⚠️ ELEVATED REGIME`
      : `GARCH Volatility Forecast — Next ${horizonDays} Trading Days`,
    height: 360,
    xaxis:  {
      ...BASE_LAYOUT.xaxis,
      title: { text: "Trading Days Forward" },
      type:  "linear",
      dtick: 1,
    },
    yaxis:  {
      ...BASE_LAYOUT.yaxis,
      title:      { text: "Annualised Volatility (%)" },
      type:       "linear",
      ticksuffix: "%",
    },
    margin: { l: 80, r: 40, t: 50, b: 70 },
    shapes: longrunVol !== null ? [{
      type: "line",
      xref: "paper", yref: "y",
      x0:   0, x1:   1,
      y0:   longrunVol, y1: longrunVol,
      line: { color: C.slate, width: 1, dash: "dot" },
    }] : [],
    annotations: longrunVol !== null ? [{
      x:         0.98,
      y:         longrunVol,
      xref:      "paper",
      yref:      "y",
      text:      `Avg long-run vol: ${longrunVol.toFixed(1)}%`,
      showarrow: false,
      font:      { color: C.slate, size: 9 },
      xanchor:   "right",
      yanchor:   "bottom",
    }] : [],
  };

  // Real rolling CVaR chart: current + optimal lines at the selected window.
  const WINDOW_LABEL = { "21": "1M (21-day)", "63": "3M (63-day)", "252": "1Y (252-day)" };
  const cvarHistData = [];
  if (rollData.cur) {
    cvarHistData.push({
      type: "scatter", mode: "lines",
      name: "Current portfolio",
      x: rollData.cur.window_end,
      y: rollData.cur.rolling_cvar.map(v => v * 100),
      line: { color: C.blue, width: 1.8 },
      hovertemplate: "End-day %{x}<br>Current CVaR 95%: %{y:.2f}%<extra></extra>",
    });
  }
  if (rollData.opt) {
    cvarHistData.push({
      type: "scatter", mode: "lines",
      name: "Optimal portfolio",
      x: rollData.opt.window_end,
      y: rollData.opt.rolling_cvar.map(v => v * 100),
      line: { color: C.green, width: 1.8, dash: "dot" },
      hovertemplate: "End-day %{x}<br>Optimal CVaR 95%: %{y:.2f}%<extra></extra>",
    });
  }

  // Full-window reported CVaR as a dashed reference. Different window than the
  // rolling series by design (full ~2y vs trailing window) — labelled as such.
  const fullWindowCvar = riskMetrics ? riskMetrics.cvar_95 * 100 : null;

  const cvarHistLayout = {
    ...BASE_LAYOUT,
    title:  `Rolling Daily CVaR — Historical · ${WINDOW_LABEL[rollWindow]} lookback`,
    height: 360,
    xaxis:  { ...BASE_LAYOUT.xaxis, title: { text: "Window end-day (trading days)" }, type: "linear", dtick: 50 },
    yaxis:  { ...BASE_LAYOUT.yaxis, title: { text: "Daily CVaR 95% (%)" }, type: "linear", ticksuffix: "%" },
    margin: { l: 80, r: 40, t: 50, b: 70 },
    shapes: fullWindowCvar !== null ? [{
      type: "line", xref: "paper", yref: "y",
      x0: 0, x1: 1, y0: fullWindowCvar, y1: fullWindowCvar,
      line: { color: C.slate, width: 1, dash: "dash" },
    }] : [],
    annotations: fullWindowCvar !== null ? [{
      x: 0.98, y: fullWindowCvar, xref: "paper", yref: "y",
      text: `Full-window CVaR: ${fullWindowCvar.toFixed(2)}%`,
      showarrow: false, font: { color: C.slate, size: 9 },
      xanchor: "right", yanchor: "bottom",
    }] : [],
  };

  const config = {
    responsive:             true,
    displayModeBar:         true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  // ── Posture strip metrics ──────────────────────────────────────────────────
  // CVaR + Vol carry trend arrows (current value vs the selected window's
  // rolling mean). Max drawdown + Sharpe show current + optimal values only —
  // a rolling trend for those isn't statistically clean on short windows.
  const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(2)}%`);
  const num = (v) => (v == null ? "—" : v.toFixed(2));

  const curCvar  = riskMetrics?.cvar_95 ?? null;
  const curVol   = riskMetrics ? avgVol(riskMetrics.volatility) : null;
  const optCvar  = rollData.opt ? rollData.opt.rolling_cvar.at(-1) : null;
  const optVol   = optimisation?.portfolio_volatility ?? null;

  const cvarTrend = trendVs(curCvar, rollData.cur?.mean_cvar);
  const volTrend  = trendVs(curVol,  rollData.cur?.mean_vol);

  const posture = [
    { label: "CVaR 95% · Daily", value: pct(curCvar),                     trend: cvarTrend,
      optLabel: optCvar != null ? `optimal ${pct(optCvar)}` : null },
    { label: "Portfolio vol", value: pct(curVol),                     trend: volTrend,
      optLabel: optVol != null ? `optimal ${pct(optVol)}` : null },
    { label: "Max drawdown",  value: pct(riskMetrics?.max_drawdown),  trend: null,
      optLabel: null },
    { label: "Sharpe ratio",  value: num(riskMetrics?.sharpe_ratio),  trend: null,
      optLabel: optimisation ? `optimal ${num(optimisation.sharpe_ratio)}` : null },
  ];

  const trendArrow = (t) =>
    t === "up"   ? { ch: "↑", col: C.red }
  : t === "down" ? { ch: "↓", col: C.green }
  : t === "flat" ? { ch: "→", col: C.slate }
  : null;

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
          {garchForecast ? (() => {
            const abs = Object.values(garchForecast.per_asset)
              .map(a => a.alpha_plus_beta);
            const lo = Math.min(...abs).toFixed(3);
            const hi = Math.max(...abs).toFixed(3);
            return `GARCH(1,1)-t model · fitted persistence α+β ${lo}–${hi} across assets · ` +
              `deterministic expected volatility path (not a stochastic simulation)`;
          })() : "GARCH forecast unavailable — run analysis to compute."}
        </div>
      </div>

      {/* Rolling risk evolution — real, window-configurable, current + optimal */}
      <div style={{ ...styles.card, minHeight: "420px" }}>
        <div style={styles.selectorRow}>
          <div style={styles.cardTitle}>Rolling Daily CVaR · Historical</div>
          <div style={styles.segGroup} role="group" aria-label="Rolling window">
            {[["21", "1M"], ["63", "3M"], ["252", "1Y"]].map(([w, lbl]) => (
              <button
                key={w}
                style={styles.segBtn(rollWindow === w)}
                onClick={() => setRollWindow(w)}
              >
                {lbl}
              </button>
            ))}
          </div>
        </div>

        {cvarHistData.length > 0 ? (
          <>
            <Plot
              key={rollWindow}
              data={cvarHistData}
              layout={cvarHistLayout}
              config={config}
              style={{ width: "100%" }}
              useResizeHandler={true}
            />
            <div style={styles.legendRow}>
              <span style={styles.legendItem()}>
                <span style={styles.legendDot(C.blue)} /> Current portfolio
              </span>
              {rollData.opt && (
                <span style={styles.legendItem()}>
                  <span style={styles.legendDot(C.green)} /> Optimal portfolio
                </span>
              )}
            </div>
            <div style={styles.note}>
              Each point is a 1-day CVaR estimated from a trailing {WINDOW_LABEL[rollWindow]}{" "}
              window of history — not the 1-year forward-simulated CVaR shown on the
              Compliance tab. The endpoint differs from the full-window CVaR by design;
              they measure different lookback lengths.
              {!rollData.opt && " Optimal overlay appears on full/optimisation queries."}
            </div>
          </>
        ) : (
          <div style={{ ...styles.emptyState, padding: "40px 20px" }}>
            Rolling risk unavailable — history too short for this window.
          </div>
        )}
      </div>

      {/* Risk posture strip */}
      <div style={styles.postureHeading}>Risk Posture</div>
      <div style={styles.postureGrid}>
        {posture.map((p) => {
          const a = trendArrow(p.trend);
          return (
            <div key={p.label} style={styles.postureCard}>
              <div style={styles.postureLabel}>{p.label}</div>
              <div style={styles.postureValue}>
                {p.value}
                {a && (
                  <span style={{ color: a.col, fontSize: "0.9rem", marginLeft: "6px" }}>
                    {a.ch}<span style={{ fontSize: "0.62rem", color: C.slate }}> vs avg</span>
                  </span>
                )}
              </div>
              {p.optLabel && <div style={styles.postureOpt}>{p.optLabel}</div>}
            </div>
          );
        })}
      </div>

    </div>
  );
}
