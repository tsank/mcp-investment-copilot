// src/components/PortfolioInput.jsx
//
// Tab 1 — 📋 My Portfolio
//
// Responsibilities:
//   1. Query text input
//   2. Total portfolio value input
//   3. Holdings table — editable symbol/weight rows
//   4. Add symbol dropdown
//   5. Weight sum indicator (green if =1.0, red otherwise)
//   6. Analyse button (disabled if weights don't sum to 1.0)
//
// Data flow:
//   All state lives in usePortfolio() hook in App.js.
//   This component receives state as props and calls setters via props.
//   It owns NO state — it is a "controlled component".
//
// What is a controlled component?
//   A component where the displayed value always comes from props,
//   and every change fires a callback to update the parent's state.
//   Equivalent to a Python function that takes arguments and returns
//   results — it never modifies global state directly.
//
// Props received from App.js:
//   holdings       {Object}   symbol → weight mapping
//   totalValue     {number}   total portfolio value in INR
//   query          {string}   analysis query text
//   weightSum      {number}   sum of all weights (derived)
//   loading        {boolean}  true while API call is in progress
//   onQueryChange  {function} called when query text changes
//   onValueChange  {function} called when total value changes
//   onWeightChange {function} called when a weight is edited
//   onAddSymbol    {function} called when a symbol is added
//   onRemoveSymbol {function} called when a symbol is removed
//   onAnalyse      {function} called when Analyse button is clicked

import { useState } from "react";
import { AVAILABLE_SYMBOLS, C } from "../constants";

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = {
  container: {
    display:  "grid",
    gridTemplateColumns: "1fr 1fr",
    gap:      "24px",
    padding:  "24px",
  },
  card: {
    backgroundColor: C.card,
    borderRadius:    "10px",
    padding:         "20px",
    border:          `1px solid ${C.blue}22`,
  },
  label: {
    display:      "block",
    color:        C.slate,
    fontSize:     "0.82rem",
    marginBottom: "6px",
    fontFamily:   "Inter, sans-serif",
  },
  input: {
    width:           "100%",
    backgroundColor: "#0D1628",
    color:           C.white,
    border:          `1px solid ${C.blue}44`,
    borderRadius:    "6px",
    padding:         "10px 12px",
    fontSize:        "0.9rem",
    fontFamily:      "Inter, sans-serif",
    boxSizing:       "border-box",
    outline:         "none",
  },
  textarea: {
    width:           "100%",
    backgroundColor: "#0D1628",
    color:           C.white,
    border:          `1px solid ${C.blue}44`,
    borderRadius:    "6px",
    padding:         "10px 12px",
    fontSize:        "0.9rem",
    fontFamily:      "Inter, sans-serif",
    boxSizing:       "border-box",
    resize:          "vertical",
    minHeight:       "80px",
    outline:         "none",
  },
  table: {
    width:          "100%",
    borderCollapse: "collapse",
    fontSize:       "0.88rem",
    fontFamily:     "IBM Plex Mono, monospace",
  },
  th: {
    textAlign:     "left",
    color:         C.blue,
    padding:       "8px 10px",
    borderBottom:  `1px solid ${C.card}`,
    fontSize:      "0.80rem",
  },
  td: {
    padding:      "6px 10px",
    color:        C.white,
    borderBottom: `1px solid #1E2D4E44`,
  },
  weightInput: {
    backgroundColor: "transparent",
    border:          "none",
    borderBottom:    `1px solid ${C.blue}44`,
    color:           C.white,
    fontFamily:      "IBM Plex Mono, monospace",
    fontSize:        "0.88rem",
    width:           "70px",
    textAlign:       "right",
    outline:         "none",
    padding:         "2px 4px",
  },
  removeBtn: {
    background:  "none",
    border:      "none",
    color:       C.red,
    cursor:      "pointer",
    fontSize:    "1rem",
    padding:     "0 4px",
    lineHeight:  1,
  },
  addRow: {
    display:    "flex",
    gap:        "8px",
    marginTop:  "12px",
    alignItems: "center",
  },
  select: {
    flex:            1,
    backgroundColor: "#0D1628",
    color:           C.white,
    border:          `1px solid ${C.blue}44`,
    borderRadius:    "6px",
    padding:         "8px 10px",
    fontSize:        "0.85rem",
    fontFamily:      "IBM Plex Mono, monospace",
    outline:         "none",
  },
  addBtn: {
    backgroundColor: C.blue + "22",
    border:          `1px solid ${C.blue}`,
    color:           C.blue,
    borderRadius:    "6px",
    padding:         "8px 16px",
    cursor:          "pointer",
    fontSize:        "0.85rem",
    fontFamily:      "IBM Plex Mono, monospace",
  },
  weightSum: (valid) => ({
    marginTop:  "10px",
    fontSize:   "0.85rem",
    fontFamily: "IBM Plex Mono, monospace",
    color:      valid ? C.green : C.red,
    fontWeight: 600,
  }),
  analyseBtn: (disabled) => ({
    width:           "100%",
    padding:         "14px",
    marginTop:       "20px",
    backgroundColor: disabled ? C.card : C.blue,
    color:           disabled ? C.slate : C.white,
    border:          `1px solid ${disabled ? C.slate : C.blue}`,
    borderRadius:    "8px",
    fontSize:        "1rem",
    fontWeight:      700,
    fontFamily:      "Inter, sans-serif",
    cursor:          disabled ? "not-allowed" : "pointer",
    letterSpacing:   "1px",
    transition:      "all 0.2s ease",
  }),
  section: {
    marginBottom: "18px",
  },
  sectionTitle: {
    color:        C.white,
    fontSize:     "0.9rem",
    fontWeight:   600,
    marginBottom: "12px",
    fontFamily:   "Inter, sans-serif",
  },
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function PortfolioInput({
  holdings,
  totalValue,
  query,
  weightSum,
  loading,
  onQueryChange,
  onValueChange,
  onWeightChange,
  onAddSymbol,
  onRemoveSymbol,
  onAnalyse,
}) {

  // selectedSymbol is local state — only needed inside this component
  // It doesn't need to live in App.js because no other component cares about it
  const [selectedSymbol, setSelectedSymbol] = useState("");

  // Weight sum is valid when it equals 1.0 within a small tolerance
  const weightValid = Math.abs(weightSum - 1.0) < 0.01;

  // Analyse button is disabled when weights don't sum to 1.0 or API is loading
  const analyseDisabled = !weightValid || loading;

  // Handle add symbol button click
  function handleAdd() {
    if (!selectedSymbol) return;
    onAddSymbol(selectedSymbol);
    setSelectedSymbol("");  // reset dropdown after adding
  }

  // Available symbols not yet in holdings
  const symbolsNotInPortfolio = AVAILABLE_SYMBOLS.filter(
    s => !Object.keys(holdings).includes(s)
  );

  return (
    <div style={styles.container}>

      {/* ── Left column: Query + Value + Analyse button ── */}
      <div style={styles.card}>

        <div style={styles.section}>
          <div style={styles.sectionTitle}>Analysis Query</div>
          <label style={styles.label}>
            What would you like to analyse?
          </label>
          {/* textarea — multi-line text input */}
          {/* value={query} makes this a controlled input — value always from props */}
          {/* onChange fires on every keystroke — calls onQueryChange with new value */}
          <textarea
            style={styles.textarea}
            value={query}
            onChange={e => onQueryChange(e.target.value)}
            rows={4}
            placeholder="e.g. Analyse my portfolio risk and suggest rebalancing"
          />
        </div>

        <div style={styles.section}>
          <label style={styles.label}>Total Portfolio Value (INR ₹)</label>
          <input
            style={styles.input}
            type="number"
            value={totalValue}
            onChange={e => onValueChange(Number(e.target.value))}
            min={1000}
            step={10000}
          />
        </div>

        {/* Analyse button */}
        <button
          style={styles.analyseBtn(analyseDisabled)}
          onClick={onAnalyse}
          disabled={analyseDisabled}
        >
          {loading ? "⏳ Analysing… (~30s)" : "⚡ Analyse Portfolio"}
        </button>

        {/* Helper text */}
        <div style={{
          marginTop: "10px",
          fontSize:  "0.75rem",
          color:     C.slate,
          fontFamily:"Inter, sans-serif",
        }}>
          {loading
            ? "Running LangGraph pipeline — GARCH simulation takes ~30 seconds"
            : "Holdings are saved automatically after a successful analysis"}
        </div>

      </div>

      {/* ── Right column: Holdings table ── */}
      <div style={styles.card}>

        <div style={styles.sectionTitle}>Holdings</div>
        <div style={{ fontSize: "0.75rem", color: C.slate, marginBottom: "12px", fontFamily: "Inter, sans-serif" }}>
          Edit weights inline · must sum to 1.000
        </div>

        {/* Holdings table */}
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Symbol</th>
              <th style={{ ...styles.th, textAlign: "right" }}>Weight</th>
              <th style={{ ...styles.th, textAlign: "right" }}>% of Portfolio</th>
              <th style={styles.th}></th>
            </tr>
          </thead>
          <tbody>
            {/* Object.entries converts {symbol: weight} to [[symbol, weight], ...] */}
            {/* Equivalent to holdings.items() in Python */}
            {Object.entries(holdings).map(([symbol, weight]) => (
              <tr key={symbol}>
                <td style={styles.td}>
                  <span style={{ color: C.blue }}>{symbol.replace(".NS", "")}</span>
                  <span style={{ color: C.slate, fontSize: "0.75rem" }}>.NS</span>
                </td>
                <td style={{ ...styles.td, textAlign: "right" }}>
                  {/* Controlled input — value always from holdings prop */}
                  <input
                    style={styles.weightInput}
                    type="number"
                    value={weight}
                    step={0.01}
                    min={0}
                    max={1}
                    onChange={e => onWeightChange(symbol, parseFloat(e.target.value))}
                  />
                </td>
                <td style={{ ...styles.td, textAlign: "right", color: C.slate }}>
                  {(weight * 100).toFixed(1)}%
                </td>
                <td style={styles.td}>
                  <button
                    style={styles.removeBtn}
                    onClick={() => onRemoveSymbol(symbol)}
                    title={`Remove ${symbol}`}
                  >
                    ✕
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Weight sum indicator */}
        <div style={styles.weightSum(weightValid)}>
          Weight sum: {weightSum.toFixed(3)}
          {weightValid ? " ✓" : ` (need ${(1 - weightSum).toFixed(3)} more)`}
        </div>

        {/* Add symbol row */}
        <div style={styles.addRow}>
          <select
            style={styles.select}
            value={selectedSymbol}
            onChange={e => setSelectedSymbol(e.target.value)}
          >
            <option value="">Add symbol…</option>
            {symbolsNotInPortfolio.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <button
            style={styles.addBtn}
            onClick={handleAdd}
            disabled={!selectedSymbol}
          >
            ＋ Add
          </button>
        </div>

      </div>
    </div>
  );
}
