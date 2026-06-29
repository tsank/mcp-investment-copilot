// src/components/Header.jsx
//
// Top bar — always visible regardless of active tab.
//
// Responsibilities:
//   1. Display app title
//   2. Show API health (is FastAPI on :8900 reachable?)
//   3. Show AWS ECS status (Running / Stopped)
//   4. Start / Stop AWS buttons
//
// Data flow:
//   Header owns its own AWS + health state via useState.
//   It calls api.js functions directly — no props needed from App.js.
//   useEffect polls AWS status every 30 seconds while mounted.
//
// Why Header owns its own state (not App.js)?
//   AWS status is not needed by any other component.
//   Keeping it here follows the principle: state lives as close
//   to where it's used as possible.

import { useState, useEffect, useCallback } from "react";
import { getAwsStatus, startAws, stopAws, checkHealth } from "../services/api";
import { C } from "../constants";

// ── Styles ────────────────────────────────────────────────────────────────────
// Inline styles as JavaScript objects.
// Keys are camelCase (backgroundColor not background-color).
// Values are strings (same as CSS values).
// Equivalent to the CSS dict in the Gradio app.

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
  btn: (color, disabled) => ({
    padding:         "5px 14px",
    borderRadius:    "6px",
    border:          `1px solid ${color}`,
    backgroundColor: disabled ? C.card : color + "22",
    color:           disabled ? C.slate : color,
    fontSize:        "0.80rem",
    fontFamily:      "IBM Plex Mono, monospace",
    cursor:          disabled ? "not-allowed" : "pointer",
    transition:      "all 0.15s ease",
  }),
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function Header() {

  // ── Local state ─────────────────────────────────────────────────────────────
  const [apiHealth,  setApiHealth]  = useState("checking"); // "ok" | "error" | "checking"
  const [awsStatus,  setAwsStatus]  = useState("unknown");  // "running" | "stopped" | "unknown"
  const [awsLoading, setAwsLoading] = useState(false);      // spinner during start/stop

  // ── fetchAwsStatus — poll ECS task count ────────────────────────────────────
  // useCallback memoises the function so useEffect's dependency array is stable.
  // Without useCallback, a new function reference is created on every render,
  // causing useEffect to re-run infinitely.

  const fetchAwsStatus = useCallback(async () => {
    try {
      const data = await getAwsStatus();
      setAwsStatus(data.running_count > 0 ? "running" : "stopped");
    } catch {
      setAwsStatus("unknown");
    }
  }, []);

  // ── fetchHealth — check FastAPI reachability ─────────────────────────────────
  const fetchHealth = useCallback(async () => {
    try {
      await checkHealth();
      setApiHealth("ok");
    } catch {
      setApiHealth("error");
    }
  }, []);

  // ── useEffect — run on mount and set up polling ──────────────────────────────
  // useEffect(fn, [deps]) — equivalent to Python's __init__ side effects.
  // Empty dependency array [] means: run once when component mounts.
  // The returned cleanup function runs when component unmounts.

  useEffect(() => {
    // Run immediately on mount
    fetchHealth();
    fetchAwsStatus();

    // Poll AWS status every 30 seconds
    const interval = setInterval(fetchAwsStatus, 30000);

    // Cleanup — clear interval when Header unmounts
    // Equivalent to cancelling a background thread in Python
    return () => clearInterval(interval);
  }, [fetchHealth, fetchAwsStatus]);

  // ── handleStart — start AWS ECS tasks ───────────────────────────────────────
  async function handleStart() {
    setAwsLoading(true);
    try {
      await startAws();
      // Poll status after a short delay to let ECS update
      setTimeout(fetchAwsStatus, 3000);
    } catch (err) {
      console.error("Failed to start AWS:", err);
    } finally {
      setAwsLoading(false);
    }
  }

  // ── handleStop — stop AWS ECS tasks ─────────────────────────────────────────
  async function handleStop() {
    if (!window.confirm("Stop AWS ECS tasks? Billing will pause.")) return;
    setAwsLoading(true);
    try {
      await stopAws();
      setTimeout(fetchAwsStatus, 3000);
    } catch (err) {
      console.error("Failed to stop AWS:", err);
    } finally {
      setAwsLoading(false);
    }
  }

  // ── Derived display values ───────────────────────────────────────────────────
  const apiColor = apiHealth === "ok" ? C.green : apiHealth === "error" ? C.red : C.amber;
  const apiLabel = apiHealth === "ok" ? "API ✓" : apiHealth === "error" ? "API ✗" : "API …";

  const awsColor = awsStatus === "running" ? C.green : awsStatus === "stopped" ? C.red : C.amber;
  const awsLabel = awsStatus === "running" ? "AWS: Running" : awsStatus === "stopped" ? "AWS: Stopped" : "AWS: Unknown";

  // ── JSX — what the component renders ────────────────────────────────────────
  // JSX looks like HTML but it's JavaScript.
  // className instead of class (class is a reserved word in JS).
  // Style is a JavaScript object, not a CSS string.
  // {expression} embeds JavaScript inside JSX.

  return (
    <header style={styles.header}>

      {/* Left side — title */}
      <div>
        <h1 style={styles.title}>⚡ MCP Investment Copilot</h1>
        <div style={styles.subtitle}>
          Agentic portfolio analysis · LangGraph orchestration · 5 MCP servers
        </div>
      </div>

      {/* Right side — indicators and buttons */}
      <div style={styles.controls}>

        {/* API health indicator */}
        <div style={styles.indicator}>
          <span style={styles.dot(apiColor)} />
          <span style={{ color: apiColor }}>{apiLabel}</span>
        </div>

        {/* AWS status indicator */}
        <div style={styles.indicator}>
          <span style={styles.dot(awsColor)} />
          <span style={{ color: awsColor }}>{awsLabel}</span>
        </div>

        {/* Start button — only shown when stopped or unknown */}
        {awsStatus !== "running" && (
          <button
            style={styles.btn(C.green, awsLoading)}
            onClick={handleStart}
            disabled={awsLoading}
          >
            {awsLoading ? "Starting…" : "▶ Start AWS"}
          </button>
        )}

        {/* Stop button — only shown when running */}
        {awsStatus === "running" && (
          <button
            style={styles.btn(C.red, awsLoading)}
            onClick={handleStop}
            disabled={awsLoading}
          >
            {awsLoading ? "Stopping…" : "■ Stop AWS"}
          </button>
        )}

      </div>
    </header>
  );
}
