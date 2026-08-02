# Changelog

All notable changes to Portfolio Copilot are documented here.

---

## [v1] — 2026-07-08 — AWS ECS Fargate

**Status:** Complete, tagged `v1-fargate`, permanently recoverable via `git checkout v1-fargate`.

### Added
- FastAPI backend, LangGraph orchestrator (7 nodes), 5 independent MCP servers (Market Data, Risk Engine, Portfolio Optimiser, Scenario Simulation, Compliance)
- React frontend, 7-tab dashboard (My Portfolio, AI Recommendation, Compliance, Rebalance, Efficient Frontier, Scenarios, Risk)
- GARCH(1,1)-t volatility forecasting, VaR/CVaR/Sharpe/drawdown risk metrics, SLSQP mean-variance optimisation with efficient frontier, Monte Carlo + GARCH-based scenario simulation, configurable YAML compliance ruleset
- 242 tests across all 5 MCP servers and the orchestration layer
- Deployed end-to-end on AWS ECS Fargate (`ap-south-1`) — ECR, Secrets Manager, CloudWatch Logs
- `README.md` and `ARCHITECTURE.md`

### Fixed
- Plotly.js chart corruption on tab revisit — two root causes: Plotly's axis auto-type detection giving a different (wrong) answer on repeated mounts within one browser session, and a Plotly.js v3 breaking change requiring `title: { text: "..." }` instead of a plain string
- Docker image platform mismatch (ARM64 built on Apple Silicon vs. required AMD64 for Fargate)

### Renamed
- App display name: "MCP Investment Copilot" → "Portfolio Copilot" (user-facing surfaces only — repo name and internal code comments intentionally kept referencing MCP as accurate technical context)

### Known limitations
- No Application Load Balancer — Fargate public IPs change on every service restart, requiring a manual UI rebuild with the new backend IP. Deliberate cost trade-off (see `Portfolio_Copilot_Reference.docx` for the full cost comparison). **Resolved in v2.**

---

## [v2] — 2026-07-12 — Serverless re-platform

**Branch:** `v2-serverless`. Core infrastructure complete and verified end-to-end, including on iPhone.

### Added
- Backend: AWS Lambda (container image, ARM64/Graviton2, 1024MB, 120s timeout) + API Gateway, replacing ECS Fargate — permanent URL `https://701eexyejj.execute-api.ap-south-1.amazonaws.com`
- Frontend: S3 + CloudFront, replacing the Fargate UI container and nginx — permanent HTTPS URL `https://d2jlcue9iriq3l.cloudfront.net`
- MCP servers refactored from subprocess-spawned to direct in-process function calls in the orchestrator (Option B), for lower Lambda cold-start latency
- PWA support: `manifest.json` (Portfolio Copilot branding, dark navy theme), iOS-specific meta tags, custom amber lightning-bolt icon set. Installable via "Add to Home Screen" (iPhone Safari) and "Add to Dock" (macOS Safari)
- Removes v1's IP-churn limitation entirely — both frontend and backend URLs are now permanent across redeployments

### Fixed
- Docker's default buildx output includes an attestation/SBOM manifest that Lambda's container image support rejects outright — fixed by building with `--provenance=false --sbom=false`
- `logging.basicConfig()` is a silent no-op on Lambda (runtime pre-configures the root logger) — fixed with `force=True`; this was the precondition for diagnosing the bug below, since it had made every `logger.info()` call invisible in CloudWatch
- Scenario simulation node taking ~30s/call (~60s per full pipeline run): diagnosed as a single-threaded, unvectorised 10,000-simulation × 252-day for-loop in `run_garch_simulation`, not a Lambda resource constraint (confirmed by doubling Lambda memory/vCPU with no effect). Pragmatic fix: reduced simulation count 10,000 → 1,000 (pipeline now ~17–26s, demo-viable). Proper numpy vectorisation deliberately deferred to v3 — out of v2's infra-only scope
- iOS Safari blocks/restricts plain HTTP more aggressively than desktop browsers — the S3 HTTP-only website endpoint worked on desktop but failed on iPhone Safari; resolved by requiring the CloudFront HTTPS URL

### Removed
- Dead ECS start/stop controls from the UI header — Lambda has no "running/stopped" state to toggle, the concept doesn't map onto serverless

### Kept (deliberate)
- v1's Fargate-specific files (Dockerfiles, ECS task definitions, `api/routes/aws.py`) remain visible in the repo on `main` rather than being deleted at merge time, alongside the `v1-fargate` tag — shows both deployment approaches as a single, still-real portfolio signal

### Cost
- Target ~$0.31–0.35/month at typical demo-driven usage, down from v1's $0.75–4/month depending on hours run. Real-usage check-in still pending.

---

## [v3-guardrails] — 2026-08-02 — Guardrails hardening + Risk tab overhaul

**Branch:** `v3-guardrails`. Guardrails 1–5 and the Risk tab overhaul (item 8) complete; items 6–7 (API Gateway rate limiting, in-process MCP state-leak check) outstanding before merge to `main`.

### Added — guardrails
- Input validation: unknown-ticker allowlist + max-holdings complexity cap (`api/schemas/request.py`)
- `parse_query` hardening: `OUT_OF_SCOPE` classification with graph-level short-circuit, anti-prompt-injection system prompt instructions
- Skip-messaging consistency: `ScenarioAnalysis.jsx` banner explaining missing optimal-weights comparison data on SIMULATION-only queries
- Deterministic compliance-breach alert: `synthesise.py` prepends a compliance breach warning independent of LLM prose, guaranteeing correctness even if the LLM omits the violation
- "Not financial advice" disclaimer: static global footer, visible on every tab regardless of analysis state

### Fixed — `parse_query` false positive on keyword-free queries
- The `OUT_OF_SCOPE` classifier's categories were keyword-anchored (each defined by naming its own trigger words — VaR/CVaR, rebalance, Monte Carlo, etc.), causing legitimate but casually-phrased portfolio queries ("How should my portfolio look one year from now?") to be misclassified as out-of-scope
- Confirmed via a live eval harness (`scripts/eval_classification_accuracy.py`) that gpt-4o inherited the identical failure on the same prompt — a prompt-design gap, not a gpt-4o-mini capability limit, so no model upgrade was made (would have doubled per-query cost for no fix)
- Rewrote `_SYSTEM_PROMPT` as a two-stage decision (is this portfolio-related at all? then which specific area) with few-shot examples anchoring the boundary; true-negative controls (off-topic queries, prompt injection attempts) confirmed unaffected — 10/10 on the eval harness across both models

### Added — GARCH volatility forecast now plots real per-asset data
- The GARCH Volatility Forecast chart previously recomputed every asset's decay path in the frontend from one hardcoded shared persistence constant (`PERSISTENCE = 0.92`) and a fake shared long-run target, because the real per-asset forecast — already computed correctly in `garch_forecast.py` with each asset's own fitted α+β — was never serialized into the API response
- Added `GARCHForecastResponse` schema, serialized `state.garch_result` in `analyse.py`; frontend now plots `per_asset.vol_forecast` directly with no recomputation. Long-run reference line and elevated-regime badge now use real fitted values per asset

### Added — Rolling Risk Evolution (real rolling CVaR/vol, replacing a fabricated chart)
- The "Historical Risk Evolution" chart was a seeded random walk anchored only at the current empirical CVaR endpoint — disclosed via an in-app note, but fabricated. Replaced with genuine rolling-window CVaR + volatility, computed via a new `compute_rolling_risk` tool (`servers/risk_engine/tools/rolling_cvar.py`) that reuses `risk_metrics.py`'s exact CVaR/VaR primitives so sign convention and method match the reported `cvar_95` by construction (verified: hand-computed trailing-window CVaR matches the rolling series endpoint to 1e-12 on real fixtures)
- Precomputes three windows (21/63/252 trading days — 1M/3M/1Y) in one call so the frontend selector switches client-side with no re-fetch; shorter windows confirmed empirically more reactive to volatility clusters (21-day CVaR range 0.49–3.29% vs. 252-day 1.20–1.69% on the same fixture data)
- Computed for both current portfolio weights (`compute_risk` node, every risk analysis) and optimal weights (`optimise` node, only when it runs — `optimal_weights` doesn't exist before then). Both calls non-fatal and isolated: a rolling-chart calculation failure can never null out `risk_metrics` (which feeds compliance) or `optimisation_result`
- Added a 4-card Risk Posture strip below the chart (CVaR, portfolio vol, max drawdown, Sharpe) showing current + optimal values; CVaR and vol carry a trend arrow against their own rolling mean (drawdown/Sharpe show values only — a rolling trend isn't statistically clean on short windows for either)

### Fixed — CVaR labelling (two figures, identical label, incompatible meaning)
- Two CVaR₉₅ figures were rendered under the identical "CVaR 95%" label across tabs: the 1-day historical figure (`risk_metrics.cvar_95`, ~2.5%) and the 1-year forward-simulated figure that actually gates compliance (`_select_cvar`'s `garch_sim → monte_carlo → risk_metrics` fallback, ~28–39%) — same formula, ~15× different values, looked like a contradiction without a horizon qualifier
- `cvar_source` was already computed by `_select_cvar` but discarded after the log line; threaded it through `ComplianceResult` → API → frontend so every CVaR label states its horizon and, on the Compliance tab, its actual source — including the fallback state where no simulation ran and the 25%-annual-calibrated threshold is effectively being checked against a 1-day number
- Rolling chart title/axis/note updated to state explicitly that every point is a 1-day CVaR estimated from a trailing lookback window (the window selector controls estimation lookback, not loss horizon), cross-referencing the Compliance tab's figure so the two can't be conflated
- Full detail and the underlying formulas: [`COMPUTATIONS.md`](COMPUTATIONS.md)

### Fixed — Efficient Frontier chart
- Chart and axis titles used bare-string form (`title: "..."`), which current Plotly.js silently drops rather than rendering — fixed to object form (`title: { text: "..." }`). **Note:** this is the same bug class the `[v1]` entry above already documents fixing once; it had regressed on this chart, and the three other chart components (`RebalancingAction.jsx`, `ScenarioAnalysis.jsx`, the GARCH chart in `RiskHistory.jsx`) still use the bare-string form and have not yet been re-verified
- Chart intermittently rendered at a squeezed width, especially after switching away from and back to the tab (a browser zoom — which fires a native resize event — would temporarily fix it, tab-switch would break it again). Root cause: layout had a fixed `height` but no `autosize: true`, so Plotly locked in whatever width it measured on initial draw — sometimes before the tab's layout had settled — with two competing resize mechanisms (`config.responsive` and `useResizeHandler`) both present. Fixed by adding `autosize: true` and removing the redundant `config.responsive`, keeping `useResizeHandler` as the single resize path. Two earlier attempted fixes (a `key` prop alone; a manual `Plotly.Plots.resize()` call in `onInitialized`) did not resolve it and the second actively introduced new intermittent corruption — removed once identified as the wrong mechanism

### Added — documentation
- `COMPUTATIONS.md` — detailed methodology reference: the CVaR two-horizon distinction, GARCH forecasting (including a non-stationarity / FIGARCH discussion), rolling risk evolution mechanics, optimisation, and compliance CVaR provenance, with illustrative generated charts (`docs/images/`)
- `ARCHITECTURE.md` — added a "Data flow notes" section covering the `compute_risk`/`optimise` rolling-risk node split and `cvar_source` provenance threading

### Outstanding before merge to `main`
- API Gateway rate limiting (item 6)
- In-process MCP state-leak check for mixed-intent requests back-to-back, relevant given v2's in-process MCP refactor (item 7)
- Re-verify the bare-string-title / missing-`autosize` Plotly pattern on the three chart components not yet re-checked (see Efficient Frontier fix above)
- Minor housekeeping: stray unreferenced `masktable512.png` in `ui-react/public/`

---

## [v3] — Planned, after P3

- Sector-wide symbol universe expansion, live market data
- Buy/sell-new-ticker suggestions (not just reweighting held positions)
- Differential Evolution solver for the larger, non-smooth search space
- Compliance rules extended to evaluate proposed (not just held) tickers