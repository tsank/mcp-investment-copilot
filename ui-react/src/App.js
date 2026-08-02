// src/App.js
//
// Root component — owns all application state and orchestrates all components.
//
// Responsibilities:
//   1. Own the analysis result state (from /api/v1/analyse)
//   2. Own the loading and error states
//   3. Own the active tab state
//   4. Call analysePortfolio() when user clicks Analyse
//   5. Write to localStorage only on HTTP 200
//   6. Pass data down to tab components as props
//
// State owned here vs in components:
//   App.js owns:   result, loading, error, activeTab
//   usePortfolio:  holdings, totalValue, query, weightSum
//   Header:        awsStatus, apiHealth (self-contained)
//   Components:    only local UI state (e.g. showTrace toggle)
//
// Data flow (unidirectional):
//   State (App.js)
//     → props → Components (display only)
//     ← events ← Components (call handlers defined here)

import { useState } from "react";
import { usePortfolio } from "./hooks/usePortfolio";
import { analysePortfolio } from "./services/api";

import Header             from "./components/Header";
import PortfolioInput     from "./components/PortfolioInput";
import Recommendation     from "./components/Recommendation";
import ComplianceCheck    from "./components/ComplianceCheck";
import RebalancingAction  from "./components/RebalancingAction";
import EfficientFrontier  from "./components/EfficientFrontier";
import ScenarioAnalysis   from "./components/ScenarioAnalysis";
import RiskHistory        from "./components/RiskHistory";

import "./App.css";

// ── Tab definitions ────────────────────────────────────────────────────────────
// Centralised list — tab labels and their component keys.
// Order here matches the design spec.
const TABS = [
  { id: "portfolio",  label: "📋 My Portfolio" },
  { id: "recommendation", label: "💡 AI based Recommendation" },
  { id: "compliance", label: "✅ Compliance" },
  { id: "rebalancing", label: "⚖️ Rebalance" },
  { id: "frontier",   label: "📈 Efficient Frontier" },
  { id: "scenario",   label: "🔮 Scenarios" },
  { id: "risk",       label: "📊 Risk" },
];

// ── Main App component ─────────────────────────────────────────────────────────
export default function App() {

  // ── Portfolio state (from custom hook) ────────────────────────────────────
  const {
    holdings,
    totalValue,
    query,
    weightSum,
    setTotalValue,
    setQuery,
    savePortfolio,
    addSymbol,
    removeSymbol,
    updateWeight,
  } = usePortfolio();

  // ── Analysis result state ──────────────────────────────────────────────────
  const [result,    setResult]    = useState(null);   // AnalyseResponse from API
  const [loading,   setLoading]   = useState(false);  // true during API call
  const [error,     setError]     = useState(null);   // error message string
  const [activeTab, setActiveTab] = useState("portfolio");

  // ── handleAnalyse — the main action ───────────────────────────────────────
  // Called when user clicks "⚡ Analyse Portfolio" in PortfolioInput.
  // Follows the locked architectural decision:
  //   - localStorage written ONLY after HTTP 200
  //   - error clears previous error
  //   - loading state shown during entire API call
  //   - on success: save portfolio, store result, switch to recommendation tab

  async function handleAnalyse() {
    setLoading(true);
    setError(null);

    try {
      const data = await analysePortfolio(query, holdings, totalValue);

      // HTTP 200 confirmed — safe to persist portfolio
      savePortfolio();

      // Store result — triggers re-render of all tab components
      setResult(data);

      // Auto-navigate to recommendation tab to show results
      setActiveTab("recommendation");

    } catch (err) {
      // Network error, timeout, or non-2xx HTTP status
      // axios throws automatically on non-2xx — no manual status check needed
      const message = err.response?.data?.detail
        || err.message
        || "Analysis failed — please check the API server is running";
      setError(message);

    } finally {
      // Always clear loading state — whether success or error
      setLoading(false);
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="app">

      {/* Header — always visible, manages its own AWS/health state */}
      <Header />

      {/* Error banner — shown when handleAnalyse catches an error */}
      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)} className="error-dismiss">✕</button>
        </div>
      )}

      {/* Tab navigation */}
      <nav className="tab-nav">
        {TABS.map(tab => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? "tab-btn--active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Tab content — only the active tab renders */}
      <main className="tab-content">

        {activeTab === "portfolio" && (
          <PortfolioInput
            holdings={holdings}
            totalValue={totalValue}
            query={query}
            weightSum={weightSum}
            loading={loading}
            onQueryChange={setQuery}
            onValueChange={setTotalValue}
            onWeightChange={updateWeight}
            onAddSymbol={addSymbol}
            onRemoveSymbol={removeSymbol}
            onAnalyse={handleAnalyse}
          />
        )}

        {activeTab === "recommendation" && (
          <Recommendation
            recommendation={result?.recommendation}
            compliance={result?.compliance}
            executionTrace={result?.execution_trace}
            loading={loading}
          />
        )}

        {activeTab === "compliance" && (
          <ComplianceCheck
            compliance={result?.compliance}
            riskMetrics={result?.risk_metrics}
            loading={loading}
          />
        )}

        {activeTab === "rebalancing" && (
          <RebalancingAction
            optimisation={result?.optimisation}
            portfolio={holdings}
            loading={loading}
          />
        )}

        {activeTab === "frontier" && (
          <EfficientFrontier
            optimisation={result?.optimisation}
            riskMetrics={result?.risk_metrics}
            loading={loading}
          />
        )}

        {activeTab === "scenario" && (
          <ScenarioAnalysis
            simulation={result?.simulation}
            totalValue={totalValue}
            loading={loading}
          />
        )}

        {activeTab === "risk" && (
          <RiskHistory
            riskMetrics={result?.risk_metrics}
            simulation={result?.simulation}
            loading={loading}
          />
        )}

      </main>

      {/* Global disclaimer footer — visible on every tab, independent of any
          analysis having run. Static text: this is a portfolio-analysis
          demonstration, not regulated financial advice. */}
      <footer className="disclaimer-footer">
        For demonstration and educational use only. This tool performs
        quantitative portfolio analysis and does not constitute financial,
        investment, or trading advice. Figures are model estimates, not
        guarantees. Consult a SEBI-registered investment adviser before making
        any investment decision.
      </footer>

    </div>
  );
}
