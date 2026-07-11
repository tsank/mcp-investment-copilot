# infra/LAMBDA_DEPLOYMENT.md
#
# v2 Lambda + API Gateway deployment reference.
# Resource IDs and setup steps for this specific deployment — useful
# for redeployment, debugging, or eventual ARCHITECTURE.md updates.

## Live endpoint

    https://701eexyejj.execute-api.ap-south-1.amazonaws.com

- GET  /health           — liveness check
- POST /api/v1/analyse   — main analysis endpoint

## AWS resources

| Resource | Identifier |
|---|---|
| ECR repo | `442421142920.dkr.ecr.ap-south-1.amazonaws.com/portfolio-copilot-lambda` |
| Lambda function | `portfolio-copilot-api` (arm64, 1024MB, 120s timeout) |
| Lambda execution role | `arn:aws:iam::442421142920:role/portfolio-copilot-lambda-role` |
| API Gateway (HTTP API) | `701eexyejj` |

## Redeployment (after a code change)

```bash
docker build --provenance=false --sbom=false \
  -f infra/docker/Dockerfile.lambda \
  -t portfolio-copilot-lambda:local .

docker tag portfolio-copilot-lambda:local \
  442421142920.dkr.ecr.ap-south-1.amazonaws.com/portfolio-copilot-lambda:v1

docker push 442421142920.dkr.ecr.ap-south-1.amazonaws.com/portfolio-copilot-lambda:v1

aws lambda update-function-code \
  --function-name portfolio-copilot-api \
  --image-uri 442421142920.dkr.ecr.ap-south-1.amazonaws.com/portfolio-copilot-lambda:v1 \
  --region ap-south-1 \
  --query 'LastUpdateStatus'
```

**Important:** `--provenance=false --sbom=false` is required on every build.
Docker's default buildx behaviour includes an attestation manifest that
Lambda's container image support rejects outright
(`InvalidParameterValueException: image manifest ... is not supported`).

The URL above never changes across redeployments — this is the whole
point of the v2 migration, unlike v1's Fargate public IPs which changed
on every restart.

## Known gotchas hit during setup (for future reference)

1. **Attestation manifest** — see above. Always build with
   `--provenance=false --sbom=false`.

2. **IAM user needs explicit Lambda + API Gateway permissions** —
   `mcp-copilot-deplot`'s original 5 policies (from v1) didn't include
   Lambda or API Gateway access. Added `AWSLambda_FullAccess` and
   `AmazonAPIGatewayAdministrator` (attached via root, same pattern as
   every other new-service addition in this project).

3. **Lambda execution role** (`portfolio-copilot-lambda-role`) is
   separate from v1's `ecsTaskExecutionRole` — created fresh via root,
   with `AWSLambdaBasicExecutionRole` (CloudWatch Logs) and
   `SecretsManagerReadWrite` attached.

4. **`logging.basicConfig()` needs `force=True`** in Lambda specifically
   — Lambda's Python runtime pre-configures the root logger before
   application code runs, making a plain `basicConfig()` call a silent
   no-op (per Python's own documented behaviour). Without this, every
   `logger.info()` call across the orchestrator is invisible in
   CloudWatch — this blocked real diagnosis of the timeout issue below
   until fixed.

5. **GARCH simulation performance** — `run_garch_simulation`'s nested
   Python for-loop (10,000 sims x 252 days, not vectorised) took ~30s
   per call on Lambda, ~60s total for both current+optimal weight runs.
   Doubling Lambda memory (1024MB -> 2048MB, which proportionally
   increases vCPUs) had no meaningful effect — confirms the bottleneck
   is single-threaded Python execution, not available compute. Reduced
   `_N_SIMULATIONS` from 10,000 to 1,000 as a pragmatic v2 fix (real
   pipeline time now ~17-26s). Proper fix (vectorising the loop with
   numpy) deferred to v3 — see Decision note in
   orchestrator/nodes/simulate.py's commit history for full reasoning.

6. **API Gateway permission** — `--target` on `apigatewayv2 create-api`
   mostly wires the integration automatically, but explicitly running
   `aws lambda add-permission` (principal `apigateway.amazonaws.com`,
   source ARN scoped to this API's ID) was still needed to actually
   grant invoke access.

## Secrets

`OPENAI_API_KEY` is set as a Lambda environment variable, sourced from
Secrets Manager (`mcp-copilot/openai-api-key`) at configuration time —
not injected automatically like ECS task definitions do. If the key is
ever rotated, re-run:

```bash
aws lambda update-function-configuration \
  --function-name portfolio-copilot-api \
  --environment "Variables={OPENAI_API_KEY=$(aws secretsmanager get-secret-value --secret-id mcp-copilot/openai-api-key --region ap-south-1 --query 'SecretString' --output text)}" \
  --region ap-south-1 \
  --query 'LastUpdateStatus'
```

Note the `--query 'LastUpdateStatus'` — without it, the full command
output (including the raw key value) prints to the terminal.