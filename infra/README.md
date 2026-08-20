# Deployment

AWS CDK in Python. `Lambda (container image) + EventBridge + DynamoDB + S3 + CloudFront`.

## Dependencies are deliberately separate

`infra/requirements.txt` is **not** merged into the root `pyproject.toml`. No new
dependencies without asking, and none of `aws-cdk-lib`, `boto3`, or `mangum` is needed to
run the system, the backtest, or the test suite — only to deploy it.

```bash
npm install -g aws-cdk                              # CDK CLI (Node)
python3 -m venv infra/.venv
infra/.venv/bin/pip install -r infra/requirements.txt
cd web && npm install && npm run build && cd ..

cd infra
../infra/.venv/bin/cdk bootstrap aws://ACCOUNT-ID/REGION   # once per account/region

# Optional: source real Gemini/EDGAR/Alpaca/FRED credentials first, so the
# scheduled cycle runs on real data instead of a synthetic fallback.
set -a && source ../.env && set +a
../infra/.venv/bin/cdk deploy
```

## What gets created

| Resource | Why |
|---|---|
| Lambda `CycleFunction` | One decision cycle, 15-minute timeout, 3008MB |
| EventBridge rule | 21:30 UTC weekdays — **after** the US close, so closing prices exist |
| DynamoDB table | Single table, `pk`, on-demand (one write a day) |
| Lambda `ApiFunction` | The read-only dashboard API, 90s timeout, 3008MB |
| S3 + CloudFront | Static dashboard, origin access control, no public bucket |

## Container images, not zip

`numpy` + `scipy` + `pandas` + `scikit-learn` alone run past the 250MB zip-based Lambda
limit, and both functions need the full set — `ApiFunction`'s `routes.py` imports
`src.cfa.portfolio`, which uses `sklearn`, so even the read-only API needs the whole
scientific stack. Both Lambdas build from the root [`Dockerfile`](../Dockerfile) instead,
which supports up to 10GB. `architecture=ARM_64` on both functions must match what Docker
builds natively for on the machine running `cdk deploy` — a mismatch produces
`Runtime.InvalidEntrypoint` at cold start, not a synth-time error.

`data/cache` — the output of `make backfill` — is bundled into the image too
(`COPY data/cache ./data/cache` in the Dockerfile), so the deployed Lambdas run on real
recorded prices and SEC filings, not the synthetic fallback, without ever fetching live at
runtime.

## Snapshot caching: the API never replays the backtest live

A live backtest over the real ~28-symbol universe takes seconds to a minute, not
milliseconds — too slow to compute per HTTP request, and CloudFront's own origin timeout
(60s, the hard ceiling without an AWS support request) makes a live fallback through the
CDN actively dangerous: it would 504 for the requesting client even though the Lambda
itself succeeds.

So `CycleFunction` pre-renders every dashboard route once, after each cycle, and persists
it: the 9 backtest-summary routes plus `/api/cycles` as one DynamoDB item
(`SNAPSHOT#latest`), and `/api/research/{symbol}` as one item per symbol
(`RESEARCH#<symbol>` — kept separate because 400 price points across ~28 symbols is
~700KB combined, over DynamoDB's 400KB single-item limit). `ApiFunction` only ever reads
these; no route it serves ever recomputes the backtest.

The one exception: if nothing has been recorded yet (a fresh deploy, before the first
scheduled cycle), the *first* request to hit `ApiFunction` computes once and persists the
result — so it is the only request that pays that cost, and only if it lands before
`CycleFunction`'s first run. **Prefer priming the cache directly instead of relying on
this**, since that first request still 504s at the CloudFront layer even though it
succeeds and saves its result server-side:

```bash
aws lambda invoke --function-name <CycleFunction name> --cli-read-timeout 920 /tmp/out.json
```

## Least privilege

The cycle function can read and write the whole table. The API function can read it, plus
two narrow `dynamodb:PutItem` grants — `SNAPSHOT#latest` exactly, and anything matching
`RESEARCH#*` — scoped via `dynamodb:LeadingKeys` conditions so the bootstrap-compute path
above can save what it produces. It cannot write a `CYCLE#` record or an audit entry: a bug
in a request handler still cannot alter a recorded decision, which is the property this
scoping exists to keep. Neither function is granted any credential that could reach a live
broker — there is no live endpoint in the codebase.

## LLM agents

`CycleFunction` gets `GEMINI_API_KEY` (read from the deploying shell's environment, never
committed) and `LLM_PROVIDER=gemini` only when that key is actually present — otherwise
both fall back to `null`, matching reality rather than claiming a provider that isn't
wired in. `ApiFunction` never receives either: its emergency live-fallback stays on
`NoViews` deliberately, so it can't add ~500 real network calls on top of its own 90s
timeout. Research stays `NEUTRAL` regardless of provider — no headline data source exists
in this codebase, and fabricating one would be worse than leaving it unset.

## At-least-once delivery

EventBridge guarantees **at least** once, not exactly once, and a Lambda retry after a
timeout is indistinguishable from a fresh invocation. Three independent defences, because
no one of them is sufficient:

1. **Content-hashed mandate ids** — a repeated decision produces the same id.
2. **Broker-side idempotency** — a duplicate `client_order_id` is rejected.
3. **`put_item` on the mandate id** — a replay overwrites rather than appending.

## Cost

Comfortably within light usage on the free tier: one Lambda invocation per weekday
(3008MB, up to a few minutes with real LLM calls), DynamoDB on-demand at roughly one write
a day plus one per symbol, and a CloudFront distribution serving a static bundle plus API
responses.
