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

## [v3] — Planned, after P3

- Sector-wide symbol universe expansion, live market data
- Buy/sell-new-ticker suggestions (not just reweighting held positions)
- Differential Evolution solver for the larger, non-smooth search space
- Compliance rules extended to evaluate proposed (not just held) tickers