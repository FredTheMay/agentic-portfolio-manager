# Lambda container image.
#
# numpy + scipy + pandas + scikit-learn alone exceed the 250MB unzipped limit
# for zip-based Lambdas. Container images support up to 10GB, so both the
# CycleFunction and ApiFunction build from this image with a different `cmd`
# override in infra/stack.py rather than each fighting the zip size limit.
FROM public.ecr.aws/lambda/python:3.12

COPY pyproject.toml ./
COPY src ./src
COPY config ./config
# Recorded market data (`make backfill`) — bundled so the deployed Lambdas
# read real prices/fundamentals from disk rather than falling back to
# synthetic data. Read-only at runtime; nothing here is refetched or written.
COPY data/cache ./data/cache

RUN pip install . "mangum>=0.17.0" --no-cache-dir
