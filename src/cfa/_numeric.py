"""The one place ``Decimal`` meets ``float``.

Matrix inversion, least-squares regression and iterative root-finding have no
exact-decimal implementation, so they run in float64 and convert back here.
Confining both directions to one module leaves a single place to audit rather
than a float leaking quietly into a cash calculation.

``float`` to ``Decimal`` goes through ``repr``, which round-trips a float64
exactly.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


class NumericError(ValueError):
    """Raised on a non-finite result or a malformed numeric input."""


def to_float(value: Decimal) -> float:
    """Widen a ``Decimal`` for a numpy/scipy computation."""
    result = float(value)
    if not np.isfinite(result):
        raise NumericError(f"non-finite value {value!r}")
    return result


def to_decimal(value: float) -> Decimal:
    """Narrow a float64 result back to ``Decimal`` at the public boundary."""
    if not np.isfinite(value):
        raise NumericError(f"non-finite result {value!r}")
    try:
        return Decimal(repr(float(value)))
    except InvalidOperation as exc:  # pragma: no cover - defensive
        raise NumericError(f"cannot represent {value!r} as Decimal") from exc


def to_float_array(values: Iterable[Decimal]) -> NDArray[np.float64]:
    """Convert a sequence of ``Decimal`` to a float64 vector."""
    array = np.array([to_float(v) for v in values], dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise NumericError("non-finite value in input vector")
    return array


def to_float_matrix(rows: Sequence[Sequence[Decimal]]) -> NDArray[np.float64]:
    """Convert a square matrix of ``Decimal`` to float64, checking shape."""
    if not rows:
        raise NumericError("empty matrix")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise NumericError("ragged matrix")
    if len(rows) != width:
        raise NumericError(f"matrix must be square, got {len(rows)}x{width}")
    return np.array([[to_float(v) for v in row] for row in rows], dtype=np.float64)


def to_decimal_list(array: NDArray[np.float64]) -> list[Decimal]:
    """Convert a float64 vector back to ``Decimal`` at the public boundary."""
    return [to_decimal(float(v)) for v in np.asarray(array).ravel()]


def require_same_length(name_a: str, a: Sequence[object], name_b: str, b: Sequence[object]) -> None:
    if len(a) != len(b):
        raise NumericError(f"{name_a} and {name_b} differ in length: {len(a)} vs {len(b)}")


def require_min_length(name: str, values: Sequence[object], minimum: int) -> None:
    if len(values) < minimum:
        raise NumericError(f"{name} needs at least {minimum} observations, got {len(values)}")
