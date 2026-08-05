"""Structural test 4 — SPEC §2.1: "any numeric field in an LLM response schema is a bug."

The guard runs inside :meth:`LLMProvider.complete`, so no provider can opt out
and no future agent can introduce an LLM-produced number by accident.
"""

from __future__ import annotations

import enum
from decimal import Decimal

import pytest
from pydantic import BaseModel

from src.llm.base import Conviction, Stance
from src.llm.null import NullProvider
from src.llm.schema_guard import InvalidLLMSchemaError, validate_llm_schema


class GoodView(BaseModel):
    ticker: str
    stance: Stance
    conviction: Conviction
    rationale: str
    is_speculative: bool = False


class FloatTarget(BaseModel):
    stance: Stance
    price_target: float


class DecimalTarget(BaseModel):
    stance: Stance
    fair_value: Decimal


class BareInt(BaseModel):
    stance: Stance
    upside_percent: int


class NumericStance(int, enum.Enum):
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1


class IntEnumView(BaseModel):
    stance: NumericStance


class Inner(BaseModel):
    weight: float


class NestedFloat(BaseModel):
    stance: Stance
    detail: Inner


class ListOfFloats(BaseModel):
    stance: Stance
    scores: list[float]


class OptionalFloat(BaseModel):
    stance: Stance
    target: float | None = None


def test_categorical_schema_is_accepted() -> None:
    validate_llm_schema(GoodView)


@pytest.mark.parametrize(
    "schema",
    [FloatTarget, DecimalTarget, BareInt, IntEnumView, NestedFloat, ListOfFloats, OptionalFloat],
    ids=["float", "decimal", "bare-int", "int-enum", "nested", "list", "optional"],
)
def test_numeric_schemas_are_rejected(schema: type[BaseModel]) -> None:
    with pytest.raises(InvalidLLMSchemaError):
        validate_llm_schema(schema)


def test_conviction_is_the_only_permitted_integer() -> None:
    validate_llm_schema(GoodView)
    with pytest.raises(InvalidLLMSchemaError, match="conviction ordinal"):
        validate_llm_schema(BareInt)


def test_non_model_schema_is_rejected() -> None:
    with pytest.raises(InvalidLLMSchemaError):
        validate_llm_schema(dict)  # type: ignore[arg-type]


def test_guard_runs_on_every_provider_call() -> None:
    # The invariant must hold at the provider boundary, not just where someone
    # remembers to call the validator.
    with pytest.raises(InvalidLLMSchemaError):
        NullProvider().complete("system", "user", FloatTarget)


def test_conviction_bounds_are_enforced_by_pydantic() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GoodView(ticker="AAPL", stance=Stance.NEUTRAL, conviction=9, rationale="x")
