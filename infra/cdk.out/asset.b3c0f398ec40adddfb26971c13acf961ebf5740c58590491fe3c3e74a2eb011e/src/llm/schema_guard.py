"""Rejects numeric fields in LLM response schemas.

The LLM contributes qualitative judgment; every number is computed in Python.
This check runs inside :meth:`LLMProvider.complete`, so the invariant cannot
decay into a convention someone quietly breaks.

Rejected: ``float``, ``Decimal``, ``complex``, ``int``, and any ``IntEnum``,
which would smuggle a number past a categorical-looking field. The one
sanctioned integer is :data:`src.llm.base.Conviction`, a 1-5 ordinal mapped to
tilts by configuration rather than used in arithmetic.
"""

from __future__ import annotations

import enum
import types
from decimal import Decimal
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel


class InvalidLLMSchemaError(TypeError):
    """Raised when an LLM response schema contains a numeric field."""


class ConvictionMarker:
    """Marks the one integer an LLM is permitted to emit."""

    def __repr__(self) -> str:
        return "ConvictionMarker()"


CONVICTION_MARKER = ConvictionMarker()

_FORBIDDEN_SCALARS: tuple[type, ...] = (float, Decimal, complex)


def _is_conviction(metadata: list[Any]) -> bool:
    return any(isinstance(m, ConvictionMarker) for m in metadata)


def _check_annotation(
    annotation: Any,
    *,
    path: str,
    conviction_ok: bool,
    seen: set[type[BaseModel]],
) -> None:
    if annotation is None or annotation is type(None):
        return

    origin = get_origin(annotation)

    # Annotated[X, ...] — metadata may carry the conviction marker.
    if origin is Annotated:
        args = get_args(annotation)
        inner, metadata = args[0], list(args[1:])
        _check_annotation(
            inner,
            path=path,
            conviction_ok=conviction_ok or _is_conviction(metadata),
            seen=seen,
        )
        return

    # Unions (including X | None) and containers: recurse into every argument.
    if origin in (Union, types.UnionType) or origin in (list, set, tuple, frozenset, dict):
        for arg in get_args(annotation):
            if arg is Ellipsis:
                continue
            _check_annotation(arg, path=path, conviction_ok=conviction_ok, seen=seen)
        return

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            _walk_model(annotation, path=path, seen=seen)
            return

        if issubclass(annotation, enum.Enum):
            # An IntEnum smuggles a number past a categorical-looking field.
            if issubclass(annotation, int):
                raise InvalidLLMSchemaError(
                    f"{path}: IntEnum {annotation.__name__} is numeric; "
                    "use a str-valued enum"
                )
            return

        if annotation is bool:
            return

        if issubclass(annotation, _FORBIDDEN_SCALARS):
            raise InvalidLLMSchemaError(
                f"{path}: field of type {annotation.__name__} is numeric. "
                "The LLM may not produce numbers — compute this in Python."
            )

        if issubclass(annotation, int):
            if conviction_ok:
                return
            raise InvalidLLMSchemaError(
                f"{path}: integer field is numeric. The only integer an LLM may "
                "emit is a conviction ordinal — annotate it with "
                "src.llm.base.Conviction."
            )


def _walk_model(model: type[BaseModel], *, path: str, seen: set[type[BaseModel]]) -> None:
    if model in seen:  # recursive schema; already checked
        return
    seen.add(model)
    for name, field in model.model_fields.items():
        _check_annotation(
            field.annotation,
            path=f"{path}.{name}",
            conviction_ok=_is_conviction(list(field.metadata)),
            seen=seen,
        )


def validate_llm_schema(schema: type[BaseModel]) -> None:
    """Raise :class:`InvalidLLMSchemaError` if ``schema`` contains a numeric field.

    Called by :meth:`src.llm.base.LLMProvider.complete` for every provider, so
    no provider can opt out.
    """
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise InvalidLLMSchemaError(
            f"LLM response schema must be a pydantic BaseModel, got {schema!r}"
        )
    _walk_model(schema, path=schema.__name__, seen=set())
