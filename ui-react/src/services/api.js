// src/services/api.js
//
// All HTTP calls to the FastAPI backend live here.
// Components never call fetch() directly — they import functions from here.
//
// Equivalent to the MCP client factory in the Python backend:
//   - one place where all external calls are made
//   - components don't know how the API is called, they just get data back
//
// Three API concerns:
//   1. analysePortfolio()  — the main analysis call
//   2. getAwsStatus()      — check if ECS tasks are running
//   3. startAws()          — set desired-count to 1
//   4. stopAws()           — set desired-count to 0

import axios from "axios";
import { API_URL } from "../constants";

// ── Axios instance ────────────────────────────────────────────────────────────
// Create a configured axios instance instead of using axios directly.
// This sets the base URL and default headers once — all calls inherit them.
// Equivalent to creating a requests.Session() in Python with default headers.

const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 180000, // 180 seconds — GARCH simulations take ~30s each
});

// ── Main analysis call ────────────────────────────────────────────────────────

/**
 * Call POST /api/v1/analyse with the user's query and portfolio.
 *
 * @param {string} query        - Natural language query
 * @param {Object} holdings     - Symbol → weight mapping e.g. {"RELIANCE.NS": 0.25}
 * @param {number} totalValue   - Total portfolio value in INR
 * @returns {Promise<Object>}   - AnalyseResponse from FastAPI
 *
 * Throws on network error or non-2xx HTTP status.
 * The caller (App.js handleAnalyse) catches errors and sets error state.
 */
export async function analysePortfolio(query, holdings, totalValue) {
  const payload = {
    query,
    portfolio: {
      holdings,
      total_value: totalValue,
    },
  };

  const response = await api.post("/api/v1/analyse", payload);

  // axios automatically parses JSON and throws on non-2xx status
  // response.data is the parsed AnalyseResponse object
  return response.data;
}

// ── AWS status and control ────────────────────────────────────────────────────

/**
 * GET /api/v1/aws/status
 * Returns current ECS task status.
 *
 * @returns {Promise<{status: string, running_count: number, desired_count: number}>}
 */
export async function getAwsStatus() {
  const response = await api.get("/api/v1/aws/status");
  return response.data;
}

/**
 * POST /api/v1/aws/start
 * Sets ECS desired-count to 1 — starts both API and UI tasks.
 *
 * @returns {Promise<{message: string}>}
 */
export async function startAws() {
  const response = await api.post("/api/v1/aws/start");
  return response.data;
}

/**
 * POST /api/v1/aws/stop
 * Sets ECS desired-count to 0 — stops both tasks, billing pauses.
 *
 * @returns {Promise<{message: string}>}
 */
export async function stopAws() {
  const response = await api.post("/api/v1/aws/stop");
  return response.data;
}

// ── Health check ──────────────────────────────────────────────────────────────

/**
 * GET /health
 * Simple liveness check — used by Header to show API connectivity.
 *
 * @returns {Promise<{status: string, service: string}>}
 */
export async function checkHealth() {
  const response = await api.get("/health");
  return response.data;
}