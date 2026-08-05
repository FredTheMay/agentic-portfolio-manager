# Deployment (SPEC §10, M10)

AWS CDK in Python. `Lambda + EventBridge + DynamoDB + S3 + CloudFront`.

## Dependencies are deliberately separate

`infra/requirements.txt` is **not** merged into the root `pyproject.toml`. SPEC §12 says no
new dependencies without asking, and none of `aws-cdk-lib`, `boto3`, or `mangum` is needed
to run the system, the backtest, or the test suite — only to deploy it.

```bash
python3 -m venv infra/.venv
infra/.venv/bin/pip install -r infra/requirements.txt
cd web && npm install && npm run build && cd ..
cd infra && ../infra/.venv/bin/cdk deploy
```

## What gets created

| Resource | Why |
|---|---|
| Lambda `CycleFunction` | One decision cycle, 5-minute timeout |
| EventBridge rule | 21:30 UTC weekdays — **after** the US close, so closing prices exist |
| DynamoDB table | Single table, `pk`, on-demand (one write a day) |
| Lambda `ApiFunction` | The read-only dashboard API |
| S3 + CloudFront | Static dashboard, origin access control, no public bucket |

## At-least-once delivery

EventBridge guarantees **at least** once, not exactly once, and a Lambda retry after a
timeout is indistinguishable from a fresh invocation. Three independent defences, because
no one of them is sufficient:

1. **Content-hashed mandate ids** (M4) — a repeated decision produces the same id.
2. **Broker-side idempotency** (M8) — a duplicate `client_order_id` is rejected.
3. **`put_item` on the mandate id** (M10) — a replay overwrites rather than appending.

## What is in the bundle

Only `src/` and `config/`. The handler previously imported its data generator from
`tests/`, which made the test suite a deployment dependency — packaging without it raised
`ModuleNotFoundError` at cold start, verified by packaging `src` + `config` alone and
running the handler both ways. The generator now lives in
[`src/data/synthetic.py`](../src/data/synthetic.py), and
`tests/test_layer_isolation.py::test_production_code_never_imports_the_test_suite` fails
the build if any module under `src/` imports from `tests/` again.

## Least privilege

The cycle function can read and write the table. The API function can only **read** it, so
a bug in a request handler cannot alter a recorded decision. Neither is granted any
credential that could reach a live broker — there is no live endpoint in the codebase
(SPEC §1).

## Cost

Within or near the AWS free tier: one Lambda invocation per weekday, DynamoDB on-demand at
roughly one write a day, and a CloudFront distribution serving a static bundle.
