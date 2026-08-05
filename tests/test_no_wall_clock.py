"""Structural test 1 — SPEC §4.1: no module may read the wall clock directly.

Everything takes a :class:`~src.time.clock.Clock`. This is what lets the
identical code path run a three-year backtest and a live daily cycle; a stray
``datetime.now()`` deep in an agent would make the backtest quietly read the
present.

Duration measurement (``time.perf_counter``, ``time.monotonic``) is permitted
everywhere: those read elapsed time, not the current instant.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

#: The one module allowed to read real time.
ALLOWED = {SRC / "time" / "clock.py"}

FORBIDDEN = {
    "datetime.now(": "use Clock.now()",
    "datetime.utcnow(": "use Clock.now()",
    "date.today(": "use Clock.now().date()",
    "time.time(": "use Clock.now(); perf_counter is fine for durations",
}


def _python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "proto_gen" not in p.parts)


def test_source_tree_is_not_empty() -> None:
    # Guards against the scan silently passing because it found nothing.
    assert _python_files(), "no Python sources found under src/"


def test_no_direct_wall_clock_reads() -> None:
    violations: list[str] = []

    for path in _python_files():
        if path in ALLOWED:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            code = re.sub(r"#.*$", "", line)
            for pattern, remedy in FORBIDDEN.items():
                if pattern in code:
                    rel = path.relative_to(SRC.parent)
                    violations.append(f"{rel}:{lineno}: {pattern} — {remedy}")

    assert not violations, "wall-clock reads outside src/time/ (SPEC §4.1):\n" + "\n".join(
        violations
    )


def test_clock_module_is_the_one_exception() -> None:
    # If clock.py stops calling datetime.now(), the allowlist above is stale.
    source = (SRC / "time" / "clock.py").read_text()
    assert "datetime.now(" in source
