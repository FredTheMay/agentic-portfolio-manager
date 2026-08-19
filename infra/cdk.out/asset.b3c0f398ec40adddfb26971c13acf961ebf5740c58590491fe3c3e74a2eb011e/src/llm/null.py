"""Provider that returns NEUTRAL for everything.

Makes "the system runs with the LLM disabled" a tested fact rather than an
aspiration, and serves as the fallback when a live provider returns output that
fails validation twice.

It constructs a neutral instance of whatever schema it is handed, so a new
agent cannot forget to register a default.
"""

from __future__ import annotations

import enum
from typing import Annotated, Any, TypeVar, Union, get_args, get_origin
import types

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from src.llm.base import LLMError, LLMProvider
from src.llm.schema_guard import ConvictionMarker

M = TypeVar("M", bound=BaseModel)

#: Recorded verbatim in rationale fields so the audit log shows *why* a view is
#: neutral, rather than presenting an empty string as considered judgment.
NULL_RATIONALE = "LLM disabled (NullProvider): no qualitative view; defaulted to NEUTRAL."

#: Lowest conviction — a neutral view carries no weight.
NULL_CONVICTION = 1


class NullProviderError(LLMError):
    """Raised when no neutral value can be derived for a required field."""


def _neutral_for(annotation: Any, *, path: str, conviction: bool) -> Any:
    origin = get_origin(annotation)

    if origin is Annotated:
        args = get_args(annotation)
        inner, metadata = args[0], args[1:]
        is_conviction = conviction or any(isinstance(m, ConvictionMarker) for m in metadata)
        return _neutral_for(inner, path=path, conviction=is_conviction)

    # Optional / union: None is the most neutral answer available.
    if origin in (Union, types.UnionType):
        args = get_args(annotation)
        if type(None) in args:
            return None
        return _neutral_for(args[0], path=path, conviction=conviction)

    if origin in (list, set, frozenset):
        return []
    if origin is dict:
        return {}
    if origin is tuple:
        args = get_args(annotation)
        # A variadic tuple[X, ...] may be empty; a fixed-length tuple[X, Y] may
        # not, so each slot gets its own neutral value.
        if not args or (len(args) == 2 and args[1] is Ellipsis):
            return ()
        return tuple(
            _neutral_for(arg, path=f"{path}[{i}]", conviction=conviction)
            for i, arg in enumerate(args)
        )

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return _neutral_payload(annotation, path=path)
        if issubclass(annotation, enum.Enum):
            for member in annotation:
                if member.name == "NEUTRAL":
                    return member
            raise NullProviderError(
                f"{path}: enum {annotation.__name__} has no NEUTRAL member, so "
                "NullProvider cannot answer neutrally. Add one."
            )
        if annotation is bool:
            return False
        if annotation is str:
            return NULL_RATIONALE
        if issubclass(annotation, int):
            if conviction:
                return NULL_CONVICTION
            raise NullProviderError(f"{path}: unexpected numeric field")

    raise NullProviderError(
        f"{path}: no neutral value for annotation {annotation!r}; "
        "give the field an explicit default"
    )


def _neutral_payload(schema: type[BaseModel], *, path: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        if not field.is_required():
            continue  # the schema's own default is the intended neutral value
        payload[name] = _field_neutral(field, path=f"{path}.{name}")
    return payload


def _field_neutral(field: FieldInfo, *, path: str) -> Any:
    conviction = any(isinstance(m, ConvictionMarker) for m in field.metadata)
    return _neutral_for(field.annotation, path=path, conviction=conviction)


class NullProvider(LLMProvider):
    """Returns a neutral, schema-valid response without calling any model."""

    name = "null"

    def _complete(self, system: str, user: str, schema: type[M]) -> M:
        payload = _neutral_payload(schema, path=schema.__name__)
        return schema.model_validate(payload)
