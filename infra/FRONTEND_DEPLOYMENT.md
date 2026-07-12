# infra/FRONTEND_DEPLOYMENT.md
#
# v2 S3 + CloudFront frontend deployment reference.

## Live endpoint

    https://d2jlcue9iriq3l.cloudfront.net

Free CloudFront-issued HTTPS certificate (Option A — no custom domain).
Verified working on desktop Chrome, desktop Safari, and iPhone Safari.

## AWS resources

| Resource | Identifier |
|---|---|
| S3 bucket | `portfolio-copilot-frontend-442421142920` (ap-south-1) |
| CloudFront distribution | `ERLUNS2JUNQW` |
| CloudFront domain | `d2jlcue9iriq3l.cloudfront.net` |

## Redeployment (after a frontend code change)

```bash
cd ui-react
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 18
npm run build
cd ..

aws s3 sync ui-react/build/ s3://portfolio-copilot-frontend-442421142920/ --region ap-south-1

# CloudFront caches aggressively (DefaultTTL 86400s = 24h) — invalidate
# to force it to pick up the new files immediately rather than waiting
# for the cache to naturally expire:
aws cloudfront create-invalidation \
  --distribution-id ERLUNS2JUNQW \
  --paths "/*"
```

The CloudFront URL above never changes across redeployments — same
principle as the Lambda/API Gateway URL. `ui-react/.env.production`
should only ever need to change if the *backend* URL changes, which
it won't unless the API Gateway itself is torn down and recreated.

## Setup steps taken (for reference / rebuild)

1. **S3 bucket**, static website hosting enabled
   (`index.html` as both index and error document — necessary for
   React Router's client-side routing to work correctly on refresh
   or direct navigation to a sub-path).
2. **Public access block disabled** and a bucket policy added granting
   public `s3:GetObject` on all objects — required for a public static
   site; see `infra/s3/bucket-policy.json`.
3. **CloudFront distribution** created with the S3 *website* endpoint
   (not the S3 REST API endpoint) as a custom origin — this matters,
   using the wrong origin type breaks SPA routing.
4. **ViewerProtocolPolicy** set to `redirect-to-https` (not the default
   `allow-all`) — plain HTTP requests are automatically upgraded rather
   than merely permitted alongside HTTPS.

## Known gotchas hit during setup

1. **IAM user needed S3 + CloudFront permissions added** — same pattern
   as every other new AWS service in this project. Attached
   `AmazonS3FullAccess` and `CloudFrontFullAccess` via root.

2. **iOS Safari blocks/restricts plain HTTP more aggressively than
   desktop browsers** — the S3 website endpoint alone (`http://...`)
   loaded fine in desktop Chrome and Safari, but failed to load
   correctly on iPhone Safari specifically. This was the direct,
   concrete reason CloudFront (for free HTTPS) was necessary, not just
   a nice-to-have — confirmed working on iPhone only after switching to
   the CloudFront HTTPS URL.

3. **CloudFront distributions take several minutes to deploy globally**
   (`Status: InProgress` → `Deployed`), unlike Lambda/API Gateway which
   are near-instant. Any config change (e.g. the ViewerProtocolPolicy
   fix above) re-triggers this multi-minute deployment cycle — plan
   for the wait when making changes.

4. **CloudFront caches aggressively by default** (24h TTL) — a code
   change pushed to S3 will not appear on the live CloudFront URL
   until either the cache naturally expires or an explicit
   invalidation is run (see the redeployment command above).

## Full stack summary (v2, complete)