# Portfolio Copilot

**Agentic portfolio analysis powered by LangGraph orchestration and a fleet of specialist MCP servers.**

Portfolio Copilot takes a natural-language investment query and a set of portfolio holdings, then runs a full multi-agent analysis pipeline — market data retrieval, risk computation, portfolio optimisation, Monte Carlo/GARCH scenario simulation, and compliance checking — before synthesising everything into a single AI-generated recommendation.

**🔗 Live app:** [d2jlcue9iriq3l.cloudfront.net](https://d2jlcue9iriq3l.cloudfront.net) · **API:** [701eexyejj.execute-api.ap-south-1.amazonaws.com](https://701eexyejj.execute-api.ap-south-1.amazonaws.com) · Installable as a PWA on iOS and macOS ("Add to Home Screen" / "Add to Dock")

---

## What it does

Given a portfolio (e.g. `RELIANCE.NS: 25%, TCS.NS: 20%, ...`) and a natural-language query, Portfolio Copilot:

1. Parses intent and identifies the type of analysis requested
2. Fetches historical price data for the held securities
3. Computes risk metrics — volatility, Sharpe ratio, VaR/CVaR, maximum drawdown — plus a real rolling CVaR/volatility evolution view (1M/3M/1Y windows, current vs. optimal portfolio)
4. Runs a GARCH(1,1)-t volatility forecast for the next 10 trading days
5. Optimises the portfolio (mean-variance, SLSQP solver) and computes the efficient frontier
6. Simulates 1-year forward scenarios via Monte Carlo and GARCH-based path generation
7. Checks the portfolio against a configurable compliance ruleset (e.g. sector concentration limits, CVaR thresholds)
8. Synthesises all of the above into a single, coherent investment recommendation using an LLM

All of this is visible in a 7-tab dashboard: **My Portfolio**, **AI Recommendation**, **Compliance**, **Rebalance**, **Efficient Frontier**, **Scenarios**, and **Risk**.

242 tests passing across all 5 MCP servers and the orchestration layer.

**Deep dive:** for exactly how each risk and optimisation figure is computed — including why CVaR is reported at two genuinely different horizons (a 1-day historical figure and a 1-year forward-simulated one) and how that's labelled so the two are never confused — see [`COMPUTATIONS.md`](COMPUTATIONS.md).

<!-- 🎥 Demo video: add Loom link here once recorded -->

---

## Architecture

```
React UI  →  FastAPI (on Lambda)  →  LangGraph Orchestrator (7 nodes)  →  5 MCP Servers
                                                                            ├── Market Data
                                                                            ├── Risk Engine
                                                                            ├── Portfolio Optimiser
                                                                            ├── Scenario Simulation
                                                                            └── Compliance
                                        LangGraph also calls → OpenAI (GPT-4o) for final synthesis
```

Each of the 5 MCP servers is a genuine, independent [Model Context Protocol](https://modelcontextprotocol.io/) server — not a set of local function calls dressed up as "agents." The LangGraph orchestrator coordinates calls across all five, handles the data flow between them, and hands the aggregated result to an LLM for the final narrative synthesis.

**Two deployment generations, same orchestration core**, kept side by side in this repo on purpose (see [Deployment history](#deployment-history) below):

- **v1 — AWS ECS Fargate**, MCP servers spawned as stdio subprocesses. Tagged [`v1-fargate`](../../tree/v1-fargate).
- **v2 — AWS Lambda + API Gateway + S3/CloudFront** (current), MCP servers called as direct in-process functions for Lambda-friendly cold starts.

Full request-lifecycle and deployment detail is in [`ARCHITECTURE.md`](ARCHITECTURE.md). Full computation methodology — every formula, every model, and their honest limitations — is in [`COMPUTATIONS.md`](COMPUTATIONS.md).

### Why MCP?

Splitting the domain logic into standalone MCP servers (rather than one monolithic backend) keeps each piece of financial logic — data retrieval, risk modelling, optimisation, simulation, compliance — independently testable, independently deployable, and cleanly separated by concern. It also demonstrates a genuinely multi-agent architecture rather than a single LLM call with tool use bolted on.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React, Plotly.js, Axios, installable PWA |
| Backend API | FastAPI (Mangum adapter on Lambda) |
| Orchestration | LangGraph |
| Agent protocol | Model Context Protocol (MCP) — direct in-process calls in v2, stdio subprocesses in v1 |
| LLM | OpenAI GPT-4o |
| Risk modelling | `arch` (GARCH), `scipy`, `numpy`, `pandas` |
| Optimisation | `scipy` (SLSQP solver) |
| Market data | `yfinance` (CSV fixtures in v1/v2; live data planned for v4) |
| Deployment (v2, current) | AWS Lambda (ARM64/Graviton2), API Gateway, S3, CloudFront |
| Deployment (v1, previous) | AWS ECS Fargate, ECR, Secrets Manager, CloudWatch |

---

## Running locally

**Prerequisites:** Python 3.10+, Node 18+, an OpenAI API key.

```bash
# Backend — install each service's dependencies
cd api && python -m pip install -r requirements.txt && cd ..
cd orchestrator && python -m pip install -r requirements.txt && cd ..
for server in market_data risk_engine portfolio_optimiser scenario_simulation compliance; do
  cd servers/$server && python -m pip install -r requirements.txt && cd ../..
done

# Add your OpenAI key
echo "OPENAI_API_KEY=your-key-here" > .env

# Run the API
uvicorn api.main:app --reload --port 8900
```

```bash
# Frontend
cd ui-react
npm install
npm start
```

Visit `http://localhost:3000`.

---

## AWS deployment (v2, current)

Portfolio Copilot runs entirely serverless in `ap-south-1`:

- **Backend** — AWS Lambda (container image, **ARM64/Graviton2**, 1024MB, 120s timeout) behind **API Gateway** (HTTP API), giving a permanent URL that never changes across redeployments. FastAPI is wrapped for Lambda via the **Mangum** adapter.
- **Frontend** — **S3** static hosting behind **CloudFront**, which provides free HTTPS via an ACM certificate with no custom domain required.
- **MCP servers** — all 5 servers were refactored from subprocess-spawned (v1's model) to **direct in-process function calls**, trading away process isolation for materially lower Lambda cold-start latency.
- **Secrets** — the OpenAI API key lives in Secrets Manager and is set as a Lambda environment variable at configuration time.

Full resource IDs and redeployment commands: [`infra/LAMBDA_DEPLOYMENT.md`](infra/LAMBDA_DEPLOYMENT.md), [`infra/FRONTEND_DEPLOYMENT.md`](infra/FRONTEND_DEPLOYMENT.md).

**Target cost:** ~$0.31–0.35/month at typical demo-driven usage — down from v1's $0.75–4/month, and without v1's IP-churn limitation (see below).

### Debugging notes worth knowing about

A few real issues came up building this that are worth flagging, since they're the kind of thing that only shows up once you actually deploy rather than just design on paper:

- **ARM64/Graviton2 for the Lambda function.** Unlike ECS Fargate in this region (which required `linux/amd64` images for v1), Lambda's container image support runs ARM64/Graviton2 natively — cheaper per millisecond of compute and no architecture mismatch to work around on an Apple Silicon dev machine.
- **Docker's attestation manifest breaks Lambda deploys.** Buildx's default output includes an attestation/SBOM manifest that Lambda's container image support rejects outright (`InvalidParameterValueException: image manifest ... is not supported`). Every build needs `--provenance=false --sbom=false` explicitly.
- **`logging.basicConfig()` is a silent no-op on Lambda.** Lambda's Python runtime pre-configures the root logger before application code runs, so a plain `basicConfig()` call does nothing — per Python's own documented behaviour, but easy to miss. Every `logger.info()` in the orchestrator was invisible in CloudWatch until this was set with `force=True`, which is what actually made the next bug diagnosable.
- **The real GARCH bottleneck wasn't infra — it was an unvectorised loop.** The scenario simulation node was taking ~30s per call (~60s for a full current + optimal-weights run). Doubling Lambda memory (and the vCPUs that come with it) had no effect, which confirmed the bottleneck was single-threaded Python, not available compute — a nested 10,000-simulation × 252-day for-loop in `run_garch_simulation`, never vectorised. The pragmatic v2 fix was reducing simulation count 10,000 → 1,000 (real pipeline time now ~17–26s, demo-viable); a proper numpy-vectorised rewrite is deliberately deferred to v4, since v2's scope was infra only, not modeling changes.
- **iOS Safari enforces HTTPS more strictly than desktop browsers.** The S3 website endpoint (plain HTTP) loaded fine on desktop Chrome and Safari but failed on iPhone Safari specifically. This is what made CloudFront's HTTPS a genuine functional requirement for the stated goal of a working iPhone demo, not just a nice-to-have.

## Deployment history

### v1 — AWS ECS Fargate (previous, still in the repo)

The original deployment — two ECS Fargate services (API, UI), Docker images on ECR, secrets via Secrets Manager, logs via CloudWatch. Fully preserved and runnable via `git checkout v1-fargate`; the Dockerfiles and ECS task definitions are also kept visible on `main` (`infra/docker/Dockerfile.api`, `infra/docker/Dockerfile.ui`, `infra/ecs/task_definitions/`) rather than deleted, as a second, still-real deployment path alongside v2.

**Known v1 limitation (resolved in v2):** no Application Load Balancer — each Fargate service got a new public IP on every restart, requiring the frontend to be rebuilt with the updated backend IP. A deliberate cost trade-off at the time (an always-on ALB runs ~$18–20/month regardless of usage); v2's Lambda + API Gateway gives a permanent URL at near-zero idle cost instead.

---

## Roadmap

Full version-by-version history — what shipped in v1, v2, v3, and what's planned for v4 — is in [`CHANGELOG.md`](CHANGELOG.md).

- **v4** (next, after Document Intelligence project) — sector-wide symbol universe, buy/sell-new-ticker suggestions (not just reweighting held positions), a Differential Evolution solver for the larger search space, live market data, compliance rules extended to proposed tickers, and the deferred GARCH vectorisation fix noted above. Deliberately sequenced after infra (v2) and guardrails/accuracy hardening (v3) so infra risk, guardrail risk, and modeling risk stay isolated from each other.

---

## Project structure

```
.
├── api/                          FastAPI application
├── orchestrator/                 LangGraph orchestration layer
├── servers/                      5 independent MCP servers
│   ├── market_data/
│   ├── risk_engine/
│   ├── portfolio_optimiser/
│   ├── scenario_simulation/
│   └── compliance/
├── ui-react/                     React frontend (PWA-enabled)
├── data/fixtures/                Sample NSE price data
├── docs/images/                  Illustrative charts referenced from COMPUTATIONS.md
└── infra/
    ├── docker/
    │   ├── Dockerfile.api         v1 — Fargate API image
    │   ├── Dockerfile.ui          v1 — Fargate UI image
    │   └── Dockerfile.lambda      v2 — Lambda container image
    ├── ecs/task_definitions/     v1 — ECS task definitions
    ├── s3/bucket-policy.json     v2 — frontend bucket policy
    ├── LAMBDA_DEPLOYMENT.md      v2 — backend deployment reference
    └── FRONTEND_DEPLOYMENT.md    v2 — frontend deployment reference
```

---

## License

MIT