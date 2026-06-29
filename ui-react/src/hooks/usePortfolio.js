// src/hooks/usePortfolio.js
//
// Custom React hook — encapsulates all portfolio state and localStorage logic.
//
// What is a custom hook?
//   A function that starts with "use" and contains React state/effect logic.
//   It extracts stateful logic out of components so it can be reused and tested
//   independently. Equivalent to a Python utility module that components import.
//
// What this hook does:
//   1. Loads portfolio from localStorage on first render (or defaults if empty)
//   2. Exposes portfolio state and setters to the component that uses it
//   3. Provides a savePortfolio() function that writes to localStorage
//      — called by App.js only after HTTP 200 from /api/v1/analyse
//
// Usage in App.js:
//   const { holdings, totalValue, query,
//           setHoldings, setTotalValue, setQuery,
//           savePortfolio } = usePortfolio();
//
// Single source of truth:
//   localStorage → loaded into useState on mount
//   useState     → live editing in browser memory
//   localStorage → written back only on HTTP 200

import { useState, useEffect } from "react";
import {
  DEFAULT_HOLDINGS,
  DEFAULT_TOTAL_VALUE,
  DEFAULT_QUERY,
  PORTFOLIO_STORAGE_KEY,
} from "../constants";

export function usePortfolio() {

  // ── Initialise state from localStorage ─────────────────────────────────────
  // useState(initialiserFn) — the function runs once on mount.
  // If localStorage has saved data, use it. Otherwise use defaults.
  // This is equivalent to load_portfolio_state() in the Gradio app.

  const [holdings, setHoldings] = useState(() => {
    try {
      const saved = localStorage.getItem(PORTFOLIO_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        // Validate — must have holdings object with at least one entry
        if (parsed.holdings && Object.keys(parsed.holdings).length > 0) {
          return parsed.holdings;
        }
      }
    } catch (err) {
      console.warn("usePortfolio: failed to load holdings from localStorage:", err);
    }
    return DEFAULT_HOLDINGS;
  });

  const [totalValue, setTotalValue] = useState(() => {
    try {
      const saved = localStorage.getItem(PORTFOLIO_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.total_value && parsed.total_value > 0) {
          return parsed.total_value;
        }
      }
    } catch (err) {
      console.warn("usePortfolio: failed to load total_value from localStorage:", err);
    }
    return DEFAULT_TOTAL_VALUE;
  });

  const [query, setQuery] = useState(DEFAULT_QUERY);

  // ── Weight sum (derived value — not stored) ─────────────────────────────────
  // Computed from holdings on every render.
  // No useState needed — it's a pure calculation.
  // Equivalent to sum(holdings.values()) in Python.
  const weightSum = Object.values(holdings).reduce((acc, w) => acc + w, 0);

  // ── savePortfolio — write to localStorage ───────────────────────────────────
  // Called by App.js ONLY after HTTP 200 from /api/v1/analyse.
  // Never called during editing — localStorage only reflects known-good state.
  //
  // Architectural decision: this function is passed down to App.js,
  // which calls it after a successful API response. The hook owns the
  // write logic — App.js doesn't need to know about localStorage keys
  // or JSON serialisation.

  function savePortfolio() {
    try {
      const data = {
        holdings,
        total_value: totalValue,
        saved_at: new Date().toISOString(),
      };
      localStorage.setItem(PORTFOLIO_STORAGE_KEY, JSON.stringify(data));
    } catch (err) {
      console.warn("usePortfolio: failed to save to localStorage:", err);
    }
  }

  // ── addSymbol — add a new symbol with zero weight ───────────────────────────
  // Called when user selects a symbol from the dropdown and clicks Add.
  // Does nothing if symbol already exists in holdings.

  function addSymbol(symbol) {
    if (!symbol || holdings[symbol] !== undefined) return;
    setHoldings(prev => ({ ...prev, [symbol]: 0.0 }));
  }

  // ── removeSymbol — remove a symbol from holdings ────────────────────────────
  // Called when user clicks the remove button on a holdings row.

  function removeSymbol(symbol) {
    setHoldings(prev => {
      const next = { ...prev };
      delete next[symbol];
      return next;
    });
  }

  // ── updateWeight — update a single symbol's weight ──────────────────────────
  // Called on every keystroke in the weight input field.
  // Does not write to localStorage — only updates useState.

  function updateWeight(symbol, weight) {
    const w = parseFloat(weight);
    if (isNaN(w)) return;
    setHoldings(prev => ({ ...prev, [symbol]: w }));
  }

  // ── Return everything the component needs ───────────────────────────────────
  // App.js destructures this return value:
  //   const { holdings, totalValue, ... } = usePortfolio();

  return {
    // State values
    holdings,
    totalValue,
    query,
    weightSum,

    // State setters
    setHoldings,
    setTotalValue,
    setQuery,

    // Actions
    savePortfolio,
    addSymbol,
    removeSymbol,
    updateWeight,
  };
}