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

## [v2] — In progress — Serverless re-platform

**Branch:** `v2-serverless`

### Planned
- Backend: AWS Lambda (FastAPI via Mangum) + API Gateway, replacing ECS Fargate
- Frontend: S3 + CloudFront, replacing the Fargate UI container and nginx
- MCP servers refactored from subprocess-spawned to direct in-process function calls in the orchestrator (Option B — see `ARCHITECTURE.md`), for lower Lambda cold-start latency
- Permanent, stable URL — removes the v1 IP-churn limitation entirely
- Target cost: ~$0.31–0.35/month, down from v1's $0.75–4/month depending on usage
- PWA support (manifest, icons, service worker) as a tail-end addition once the core migration is verified

---

## [v3] — Planned, after P3

- Sector-wide symbol universe expansion, live market data
- Buy/sell-new-ticker suggestions (not just reweighting held positions)
- Differential Evolution solver for the larger, non-smooth search space
- Compliance rules extended to evaluate proposed (not just held) tickers