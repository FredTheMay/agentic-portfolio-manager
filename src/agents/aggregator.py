"""View Aggregator (SPEC §5.4). **Deterministic. No LLM.**

This is the module where a model's opinion becomes a number, and it does so by
table lookup in ``config/view_mapping.yaml`` — never by asking a model for a
figure, and never by a heuristic buried in an optimizer.

That matters for a reason beyond SPEC §2.1: it makes the conversion auditable.
When someone asks why a name is overweight, the answer is a row in a YAML file
and a stance from a logged agent response, both of which are identical across
runs. "The model felt strongly about it" is not an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.llm.base import Stance

ZERO = Decimal(0)

DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "view_mapping.yaml"
)


class MappingError(ValueError):
    """Raised on a malformed or incomplete view-mapping table."""


@dataclass(frozen=True, slots=True)
class ViewMapping:
    """The auditable table that turns categories into numbers."""

    tilts: Mapping[Stance, Mapping[int, Decimal]]
    agent_weights: Mapping[str, Decimal]
    minimum_citations: int
    max_absolute_tilt: Decimal

    def tilt(self, stance: Stance, conviction: int) -> Decimal:
        """Look up the tilt for one categorical view."""
        by_conviction = self.tilts.get(stance)
        if by_conviction is None:
            raise MappingError(f"no tilt row for stance {stance}")
        value = by_conviction.get(conviction)
        if value is None:
            raise MappingError(f"no tilt for {stance.value} at conviction {conviction}")
        return value

    def weight(self, agent: str) -> Decimal:
        weight = self.agent_weights.get(agent)
        if weight is None:
            raise MappingError(f"no weight configured for agent {agent!r}")
        return weight


def mapping_from_document(document: Mapping[str, Any]) -> ViewMapping:
    """Build a :class:`ViewMapping` from a parsed YAML document."""
    raw_tilts = document.get("tilts")
    if not isinstance(raw_tilts, Mapping):
        raise MappingError("view mapping has no `tilts` section")

    tilts: dict[Stance, dict[int, Decimal]] = {}
    for stance in Stance:
        row = raw_tilts.get(stance.value)
        if not isinstance(row, Mapping):
            raise MappingError(f"view mapping has no tilts for {stance.value}")
        parsed: dict[int, Decimal] = {}
        for conviction in range(1, 6):
            if conviction not in row and str(conviction) not in row:
                raise MappingError(f"{stance.value} is missing conviction {conviction}")
            value = row.get(conviction, row.get(str(conviction)))
            try:
                parsed[conviction] = Decimal(str(value))
            except InvalidOperation as exc:
                raise MappingError(f"{stance.value}/{conviction} is not numeric") from exc
        tilts[stance] = parsed

    # NEUTRAL must be exactly zero at every conviction. A neutral view that
    # nudged the portfolio would mean "no opinion" quietly had an opinion.
    for conviction, value in tilts[Stance.NEUTRAL].items():
        if value != ZERO:
            raise MappingError(
                f"NEUTRAL at conviction {conviction} must be zero, got {value}: "
                "a neutral view cannot tilt the portfolio"
            )

    raw_weights = document.get("agent_weights")
    if not isinstance(raw_weights, Mapping):
        raise MappingError("view mapping has no `agent_weights` section")
    weights = {str(k): Decimal(str(v)) for k, v in raw_weights.items()}
    total = sum(weights.values(), ZERO)
    if total != Decimal(1):
        raise MappingError(f"agent weights must sum to 1, got {total}")

    return ViewMapping(
        tilts=tilts,
        agent_weights=weights,
        minimum_citations=int(document.get("minimum_citations", 1)),
        max_absolute_tilt=Decimal(str(document.get("max_absolute_tilt", "0.02"))),
    )


def load_mapping(path: Path | None = None) -> ViewMapping:
    """Read and validate the view-mapping table."""
    source = DEFAULT_MAPPING_PATH if path is None else path
    if not source.is_file():
        raise MappingError(f"view mapping not found at {source}")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise MappingError(f"view mapping at {source} is not a YAML mapping")
    return mapping_from_document(document)


@dataclass(frozen=True, slots=True)
class AgentOpinion:
    """One agent's categorical view of one name."""

    agent: str
    symbol: str
    stance: Stance
    conviction: int


@dataclass(frozen=True, slots=True)
class AggregatedView:
    """The combined tilt for one name, with its provenance."""

    symbol: str
    tilt: Decimal
    contributions: Mapping[str, Decimal]
    opinions: tuple[AgentOpinion, ...]


def aggregate(
    opinions: Sequence[AgentOpinion],
    mapping: ViewMapping,
) -> dict[str, AggregatedView]:
    """Combine agent opinions into one numeric tilt per symbol.

    Weighted by agent, then clamped to ``max_absolute_tilt``. The clamp is belt
    and braces against a table edit that accidentally scales everything: the
    position and sector caps would still bound the damage, but a tilt large
    enough to dominate the optimizer would be asserting an edge this system
    explicitly does not claim to have.
    """
    grouped: dict[str, list[AgentOpinion]] = {}
    for opinion in opinions:
        grouped.setdefault(opinion.symbol, []).append(opinion)

    results: dict[str, AggregatedView] = {}
    for symbol in sorted(grouped):
        contributions: dict[str, Decimal] = {}
        total = ZERO
        for opinion in sorted(grouped[symbol], key=lambda o: o.agent):
            contribution = mapping.weight(opinion.agent) * mapping.tilt(
                opinion.stance, opinion.conviction
            )
            contributions[opinion.agent] = contribution
            total += contribution

        clamped = max(-mapping.max_absolute_tilt, min(mapping.max_absolute_tilt, total))
        results[symbol] = AggregatedView(
            symbol=symbol,
            tilt=clamped,
            contributions=contributions,
            opinions=tuple(sorted(grouped[symbol], key=lambda o: o.agent)),
        )
    return results


def tilts_for_optimizer(views: Mapping[str, AggregatedView]) -> dict[str, Decimal]:
    """Reduce to the plain ``symbol -> tilt`` map the optimizer consumes."""
    return {symbol: view.tilt for symbol, view in views.items()}
