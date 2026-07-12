// src/components/Header.jsx
//
// Top bar — always visible regardless of active tab.
//
// Responsibilities:
//   1. Display app title
//   2. Show API health (is the backend reachable?)
//
// Data flow:
//   Header owns its own health state via useState.
//   It calls api.js functions directly — no props needed from App.js.
//
// v2 note: AWS ECS start/stop controls (present in v1) were removed
// here — Lambda scales to zero natively, so there's no "running" or
// "stopped" state to control or display. The concept doesn't map
// onto serverless architecture at all.

import { useState, useEffect, useCallback } from "react";
import { checkHealth } from "../services/api";
import { C } from "../constants";

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = {
  header: {
    display:         "flex",
    alignItems:      "center",
    justifyContent:  "space-between",
    padding:         "14px 28px",
    backgroundColor: C.card,
    borderBottom:    `1px solid ${C.blue}44`,
    position:        "sticky",
    top:             0,
    zIndex:          100,
  },
  title: {
    fontSize:   "1.35rem",
    fontWeight: 700,
    color:      C.white,
    fontFamily: "Inter, sans-serif",
    margin:     0,
  },
  subtitle: {
    fontSize:   "0.78rem",
    color:      C.slate,
    fontFamily: "Inter, sans-serif",
    marginTop:  2,
  },
  controls: {
    display:    "flex",
    alignItems: "center",
    gap:        "16px",
  },
  indicator: {
    display:    "flex",
    alignItems: "center",
    gap:        "6px",
    fontSize:   "0.82rem",
    fontFamily: "IBM Plex Mono, monospace",
    color:      C.slate,
  },
  dot: (color) => ({
    width:        8,
    height:       8,
    borderRadius: "50%",
    backgroundColor: color,
    display:      "inline-block",
  }),
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function Header() {

  // ── Local state ─────────────────────────────────────────────────────────────
  const [apiHealth, setApiHealth] = useState("checking"); // "ok" | "error" | "checking"

  // ── fetchHealth — check backend reachability ─────────────────────────────────
  const fetchHealth = useCallback(async () => {
    try {
      await checkHealth();
      setApiHealth("ok");
    } catch {
      setApiHealth("error");
    }
  }, []);

  // ── useEffect — run on mount ──────────────────────────────────────────────────
  useEffect(() => {
    fetchHealth();
  }, [fetchHealth]);

  // ── Derived display values ───────────────────────────────────────────────────
  const apiColor = apiHealth === "ok" ? C.green : apiHealth === "error" ? C.red : C.amber;
  const apiLabel = apiHealth === "ok" ? "API ✓" : apiHealth === "error" ? "API ✗" : "API …";

  // ── JSX — what the component renders ────────────────────────────────────────
  return (
    <header style={styles.header}>

      {/* Left side — title */}
      <div>
        <h1 style={styles.title}>⚡ Portfolio Copilot</h1>
        <div style={styles.subtitle}>
          Agentic portfolio analysis · LangGraph orchestration · 5 MCP servers
        </div>
      </div>

      {/* Right side — API health indicator */}
      <div style={styles.controls}>
        <div style={styles.indicator}>
          <span style={styles.dot(apiColor)} />
          <span style={{ color: apiColor }}>{apiLabel}</span>
        </div>
      </div>
    </header>
  );
}