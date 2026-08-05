"""Structural test 2 — SPEC §2.2: the decision layer knows nothing about execution.

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

#: Vendor names that would betray broker knowledge above the boundary.
BROKER_TOKENS = ("alpaca", "ib_insync", "ibapi")


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
        "decision layer imported execution (SPEC §2.2):\n" + "\n".join(violations)
    )


def test_broker_names_appear_only_below_the_boundary() -> None:
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if EXECUTION in path.parents or "proto_gen" in path.parts:
            continue
        lowered = path.read_text().lower()
        for token in BROKER_TOKENS:
            if token in lowered:
                violations.append(f"{path.relative_to(ROOT)} mentions {token!r}")

    assert not violations, (
        "broker reference outside src/execution/ (SPEC §2.2):\n" + "\n".join(violations)
    )


def test_both_layers_exist() -> None:
    # A passing isolation test over a missing directory proves nothing.
    assert (DECISION / "__init__.py").is_file()
    assert (EXECUTION / "__init__.py").is_file()
