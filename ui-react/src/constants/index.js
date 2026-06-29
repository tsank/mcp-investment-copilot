// src/constants/index.js
//
// Central configuration — colours, symbols, defaults.
// All other files import from here.
// Equivalent to the C = {...} dict in the Gradio app.

// ── API ───────────────────────────────────────────────────────────────────────
// process.env.REACT_APP_API_URL is set in .env.local for local dev
// and in .env.production for AWS deployment.
// Falls back to localhost:8900 if not set.
export const API_URL =
  process.env.REACT_APP_API_URL || "http://localhost:8900";

// ── Colour palette ────────────────────────────────────────────────────────────
export const C = {
  bg:     "#0A0F1E",   // deep navy — page background
  card:   "#1E2D4E",   // card background
  blue:   "#4F8EF7",   // primary accent — actions, links
  green:  "#10B981",   // positive — compliant, optimal
  red:    "#E55353",   // negative — breach, current underperformance
  amber:  "#F59E0B",   // warning — regime signal, GARCH
  purple: "#7C3AED",   // optimise node colour
  slate:  "#94A3B8",   // muted — labels, secondary text
  white:  "#E2E8F0",   // primary text
};

// ── Plotly base layout ────────────────────────────────────────────────────────
// Applied to every chart. Individual charts extend this.
export const BASE_LAYOUT = {
  paper_bgcolor: "#0A0F1E",
  plot_bgcolor:  "#0D1628",
  font:          { color: "#E2E8F0", family: "IBM Plex Mono, monospace" },
  legend:        { bgcolor: "#1E2D4E", bordercolor: "#4F8EF7", borderwidth: 1 },
  xaxis:         { gridcolor: "#1E2D4E", zerolinecolor: "#4F8EF7" },
  yaxis:         { gridcolor: "#1E2D4E", zerolinecolor: "#4F8EF7" },
  margin:        { l: 150, r: 40, t: 60, b: 80 },
};

// ── Available NSE symbols (fixture universe) ──────────────────────────────────
export const AVAILABLE_SYMBOLS = [
  "RELIANCE.NS",
  "TCS.NS",
  "INFY.NS",
  "HDFCBANK.NS",
  "ICICIBANK.NS",
  "ADANIENT.NS",
  "BAJFINANCE.NS",
  "BHARTIARTL.NS",
  "SBIN.NS",
  "LT.NS",
];

// ── Default portfolio ─────────────────────────────────────────────────────────
// Loaded on first visit before localStorage has any data.
export const DEFAULT_HOLDINGS = {
  "RELIANCE.NS":  0.25,
  "TCS.NS":       0.20,
  "INFY.NS":      0.20,
  "HDFCBANK.NS":  0.20,
  "ICICIBANK.NS": 0.15,
};

export const DEFAULT_TOTAL_VALUE = 1000000;

export const DEFAULT_QUERY =
  "Give me a complete portfolio analysis with optimisation and simulation";

// ── localStorage key ──────────────────────────────────────────────────────────
// Single key for portfolio persistence.
// Written only on HTTP 200 from /api/v1/analyse.
export const PORTFOLIO_STORAGE_KEY = "mcp_copilot_portfolio";