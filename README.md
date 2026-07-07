# Portfolio Copilot

**Agentic portfolio analysis powered by LangGraph orchestration and a fleet of specialist MCP servers.**

Portfolio Copilot takes a natural-language investment query and a set of portfolio holdings, then runs a full multi-agent analysis pipeline — market data retrieval, risk computation, portfolio optimisation, Monte Carlo/GARCH scenario simulation, and compliance checking — before synthesising everything into a single AI-generated recommendation.

---

## What it does

Given a portfolio (e.g. `RELIANCE.NS: 25%, TCS.NS: 20%, ...`) and a natural-language query, Portfolio Copilot:

1. Parses intent and identifies the type of analysis requested
2. Fetches historical price data for the held securities
3. Computes risk metrics — volatility, Sharpe ratio, VaR/CVaR, maximum drawdown
4. Runs a GARCH(1,1)-t volatility forecast for the next 10 trading days
5. Optimises the portfolio (mean-variance, SLSQP solver) and computes the efficient frontier
6. Simulates 1-year forward scenarios via Monte Carlo and GARCH-based path generation
7. Checks the portfolio against a configurable compliance ruleset (e.g. sector concentration limits, CVaR thresholds)
8. Synthesises all of the above into a single, coherent investment recommendation using an LLM

All of this is visible in a 7-tab dashboard: **My Portfolio**, **AI Recommendation**, **Compliance**, **Rebalance**, **Efficient Frontier**, **Scenarios**, and **Risk**.

242 tests passing across all 5 MCP servers and the orchestration layer.

<!-- 🎥 Demo video: add Loom link here once recorded -->

---

## Architecture

```
React UI  →  FastAPI  →  LangGraph Orchestrator (7 nodes)  →  5 MCP Servers
                                                                  ├── Market Data
                                                                  ├── Risk Engine
                                                                  ├── Portfolio Optimiser
                                                                  ├── Scenario Simulation
                                                                  └── Compliance
                              LangGraph also calls → OpenAI (GPT-4o) for final synthesis
```

Each of the 5 MCP servers is a genuine, independent [Model Context Protocol](https://modelcontextprotocol.io/) server, spoken to over stdio transport — not a set of local function calls dressed up as "agents." The LangGraph orchestrator coordinates calls across all five, handles the data flow between them, and hands the aggregated result to an LLM for the final narrative synthesis.

### Why MCP?

Splitting the domain logic into standalone MCP servers (rather than one monolithic backend) keeps each piece of financial logic — data retrieval, risk modelling, optimisation, simulation, compliance — independently testable, independently deployable, and cleanly separated by concern. It also demonstrates a genuinely multi-agent architecture rather than a single LLM call with tool use bolted on.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React, Plotly.js, Axios |
| Backend API | FastAPI, Uvicorn |
| Orchestration | LangGraph |
| Agent protocol | Model Context Protocol (MCP), stdio transport |
| LLM | OpenAI GPT-4o |
| Risk modelling | `arch` (GARCH), `scipy`, `numpy`, `pandas` |
| Optimisation | `scipy` (SLSQP solver) |
| Market data | `yfinance` (CSV fixtures in v1; live data planned for v2/v3) |
| Deployment | AWS ECS Fargate, ECR, Secrets Manager, CloudWatch |

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

## AWS deployment

Portfolio Copilot is deployed on **AWS ECS Fargate** (region: `ap-south-1`), with:

- Two Fargate services (API, UI), 0.5 vCPU / 1GB each
- Images built for `linux/amd64` and pushed to **ECR**
- The OpenAI API key stored in **AWS Secrets Manager**, injected into the container at runtime
- **CloudWatch Logs** for both services
- A minimal IAM setup: an ECS task execution role scoped to ECR pull, Secrets Manager read, and CloudWatch log write

**Known v1 limitation:** there is no Application Load Balancer in front of the Fargate services. Each service is reachable via a direct public IP, which **changes every time the service restarts**. This is a deliberate cost trade-off — an ALB running continuously costs roughly $18–20/month regardless of usage, while direct Fargate IPs cost only for the hours actually running (typically under $2/month for occasional use). The trade-off is a manual redeploy step (rebuilding the frontend with the new backend IP) after each restart. v2 replaces this entirely with AWS Lambda + API Gateway, which provides a permanent URL at near-zero idle cost.

A live demo is available on request.

---

## Roadmap

- **v2** — serverless re-platform: AWS Lambda (FastAPI via Mangum) + API Gateway, S3 + CloudFront for the frontend. Removes the IP-churn limitation above; targets near-zero idle cost.
- **v3** — sector-wide symbol universe, buy/sell-new-ticker suggestions (not just reweighting held positions), a Differential Evolution solver for the larger search space, live market data, and compliance rules extended to proposed tickers.

---

## Project structure

```
.
├── api/                    FastAPI application
├── orchestrator/           LangGraph orchestration layer
├── servers/                5 independent MCP servers
│   ├── market_data/
│   ├── risk_engine/
│   ├── portfolio_optimiser/
│   ├── scenario_simulation/
│   └── compliance/
├── ui-react/                React frontend
├── data/fixtures/          Sample NSE price data (v1)
└── infra/
    ├── docker/              Dockerfiles for API and UI
    └── ecs/                 ECS task definitions
```

---

## License

MIT