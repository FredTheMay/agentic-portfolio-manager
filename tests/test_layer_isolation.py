"""Structural test 2 — the decision layer knows nothing about execution.

The portfolio manager decides *what to hold*. It emits a ``RebalanceMandate``
of target weights and receives an ``ExecutionReport``. That is the entire
surface. Orders, venues, slicing, fills, and brokers live below the boundary.

This test duplicates the CI import-linter contract on purpose: the contract
catches it in CI, this catches it in the editor.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DECISION = SRC / "decision"
EXECUTION = SRC / "execution"

#: Order-placement surface. Any of these above the boundary means some module
#: has learned about orders, venues, or fills.
#:
#: Note this is deliberately *not* a ban on the string "alpaca". Alpaca is both
#: a market data vendor and a broker, and EDGAR/FRED/Alpaca all sit behind
#: MarketDataSource in the data layer. Reading price bars from a vendor that
#: also happens to broker trades leaks no execution semantics; calling its
#: trading API does. The hosts differ, so the check can be precise.
ORDER_SURFACE_TOKENS = (
    "api.alpaca.markets",  # the *trading* host; data.alpaca.markets is fine
    "submit_order",
    "market_order",
    "limit_order",
    "time_in_force",
    "/v2/orders",
    "ib_insync",
    "ibapi",
)

#: The decision layer is held to the stricter rule: it may not name a broker at
#: all, since it has no business knowing one exists.
BROKER_NAMES = ("alpaca", "ib_insync", "ibapi", "interactive brokers")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — resolve against the package
                parts = path.relative_to(ROOT).parent.parts
                base = ".".join(parts[: len(parts) - node.level + 1])
                modules.add(f"{base}.{node.module}" if node.module else base)
            elif node.module:
                modules.add(node.module)
    return modules


def test_decision_layer_does_not_import_execution() -> None:
    violations: list[str] = []
    for path in sorted(DECISION.rglob("*.py")):
        for module in _imported_modules(path):
            if module == "src.execution" or module.startswith("src.execution."):
                violations.append(f"{path.relative_to(ROOT)} imports {module}")

    assert not violations, (
        "decision layer imported execution:\n" + "\n".join(violations)
    )


def test_order_placement_appears_only_below_the_boundary() -> None:
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if EXECUTION in path.parents or "proto_gen" in path.parts:
            continue
        lowered = path.read_text().lower()
        for token in ORDER_SURFACE_TOKENS:
            if token in lowered:
                violations.append(f"{path.relative_to(ROOT)} mentions {token!r}")

    assert not violations, (
        "order-placement surface outside src/execution/:\n" + "\n".join(violations)
    )


def test_decision_layer_does_not_name_a_broker() -> None:
    # Stricter than the rule above: the decision layer emits target weights and
    # must not know that brokers exist at all.
    violations: list[str] = []
    for path in sorted(DECISION.rglob("*.py")):
        lowered = path.read_text().lower()
        for name in BROKER_NAMES:
            if name in lowered:
                violations.append(f"{path.relative_to(ROOT)} names {name!r}")

    assert not violations, (
        "broker named in the decision layer:\n" + "\n".join(violations)
    )


def test_market_data_uses_the_data_host_not_the_trading_host() -> None:
    # The precise form of the distinction above: src/data may talk to Alpaca's
    # market data API and must never reach its trading API.
    sources = SRC / "data" / "sources.py"
    if not sources.is_file():
        return
    text = sources.read_text().lower()
    assert "data.alpaca.markets" in text, "expected the market data host"
    assert "api.alpaca.markets" not in text, "src/data must not reach the trading API"


def test_both_layers_exist() -> None:
    # A passing isolation test over a missing directory proves nothing.
    assert (DECISION / "__init__.py").is_file()
    assert (EXECUTION / "__init__.py").is_file()


def test_production_code_never_imports_the_test_suite() -> None:
    """``src/`` must not depend on ``tests/``.

    A module under ``src/`` that imports from ``tests/`` passes every local
    check, because the tests are present whenever the tests run, and then fails
    at runtime wherever only ``src/`` is packaged.
    """
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "proto_gen" in path.parts:
            continue
        for module in _imported_modules(path):
            if module == "tests" or module.startswith("tests."):
                violations.append(f"{path.relative_to(ROOT)} imports {module}")

    assert not violations, (
        "production code depends on the test suite:\n" + "\n".join(violations)
    )
