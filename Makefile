.PHONY: install proto test coverage coverage-gate typecheck lint-imports check clean

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
## propagate into every downstream number. Add `--cov=src.risk` at M3, once the
## risk engine exists; coverage warns on a package with no code.
coverage-gate:
	$(VENV)/bin/pytest --cov=src.cfa --cov-report=term-missing --cov-fail-under=90

typecheck:
	$(VENV)/bin/mypy src tests

## SPEC §2.2 — the decision layer must not import execution.
lint-imports:
	$(VENV)/bin/lint-imports

check: test typecheck lint-imports coverage-gate

clean:
	rm -rf $(PROTO_OUT) .pytest_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
