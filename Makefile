.PHONY: install proto test coverage coverage-gate typecheck lint-imports check results serve web clean

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

PROTO_SRC := proto/execution.proto
PROTO_OUT := src/execution/proto_gen

$(VENV):
	python3.12 -m venv $(VENV)

install: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

## Regenerate protobuf stubs. Output is gitignored — regenerate, never commit.
proto:
	mkdir -p $(PROTO_OUT)
	touch $(PROTO_OUT)/__init__.py
	$(PY) -m grpc_tools.protoc -Iproto \
		--python_out=$(PROTO_OUT) \
		--pyi_out=$(PROTO_OUT) \
		--grpc_python_out=$(PROTO_OUT) \
		$(PROTO_SRC)

test:
	$(VENV)/bin/pytest

coverage:
	$(VENV)/bin/pytest --cov --cov-report=term-missing

## SPEC §11 — >=90% on the packages where a silent arithmetic error would
## propagate into every downstream number.
coverage-gate:
	$(VENV)/bin/pytest --cov=src.cfa --cov=src.risk --cov-report=term-missing --cov-fail-under=90

## Serve the read-only dashboard API on :8000.
serve:
	PYTHONPATH=. $(VENV)/bin/uvicorn --factory src.api.routes:app_from_environment --port 8000

## Build the React dashboard (requires npm).
web:
	cd web && npm install && npm run build

## Regenerate the numbers in RESULTS.md (SPEC §11).
results:
	PYTHONPATH=. $(PY) scripts/results.py

typecheck:
	$(VENV)/bin/mypy src tests

## SPEC §2.2 — the decision layer must not import execution.
lint-imports:
	$(VENV)/bin/lint-imports

check: test typecheck lint-imports coverage-gate

clean:
	rm -rf $(PROTO_OUT) .pytest_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
