from __future__ import annotations

"""When a continuous federation trains (spec §10): triggers as plugins.

The federated trigger, at the coordinator, decides when a round opens; the
local one, at each edge, whether the edge has an update for it. Both get a
``View`` of their level and answer with their name when they fire. Every
duration is data time.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveInt, field_validator

from onion_fl.continuum.drift import detectors
from onion_fl.continuum.pace import View
from onion_fl.core.registry import PluginError, Registry
from onion_fl.data.stream import Positive
from onion_fl.roles.policies import create

triggers = Registry("trigger")


def _known(registry: Registry, spec: Any) -> Any:
    try:
        create(registry, spec)
    except PluginError as exc:
        raise ValueError(str(exc)) from None
    return spec


class ScheduleParams(BaseModel):
    every: Positive = Field(description="Tiempo de datos desde la ronda anterior")


@triggers.register(
    "schedule",
    title="Calendario",
    description="Abre ronda cada cierto tiempo de datos desde la anterior.",
    params=ScheduleParams,
)
class Schedule:
    def __init__(self, every: float) -> None:
        self.every = every

    def fired(self, view: View) -> str | None:
        return "schedule" if view.now - view.since >= self.every - 1e-9 else None


class VolumeParams(BaseModel):
    samples: PositiveInt = Field(description="Filas entrenables nuevas que hacen falta")


@triggers.register(
    "volume",
    title="Volumen",
    description="Abre ronda cuando hay bastantes filas entrenables nuevas.",
    params=VolumeParams,
    explain="En el coordinador cuenta las de toda la federación desde la ronda "
    "anterior; en un edge, las suyas sin usar.",
)
class Volume:
    def __init__(self, samples: int) -> None:
        self.samples = samples

    def fired(self, view: View) -> str | None:
        return "volume" if view.volume >= self.samples else None


class DriftParams(BaseModel):
    kind: Literal["data", "prior", "performance"] = Field(
        description="Qué deriva: de datos P(X), de prior P(Y) o de rendimiento"
    )
    detector: str | dict[str, Any] = Field(
        "page_hinkley", description="Detector del estadístico de esa deriva"
    )

    _detector = field_validator("detector")(lambda v: _known(detectors, v))


@triggers.register(
    "drift",
    title="Deriva",
    description="Abre ronda cuando se detecta deriva de un tipo desde la anterior.",
    params=DriftParams,
    explain="Cada edge, cada zona y la federación vigilan su estadístico; "
    "basta una detección en cualquiera de ellos.",
)
class Drift:
    def __init__(
        self, kind: str, detector: str | dict[str, Any] = "page_hinkley"
    ) -> None:
        self.kind, self.detector = kind, detector

    def fired(self, view: View) -> str | None:
        return f"drift/{self.kind}" if view.drift.get(self.kind, 0) > 0 else None


class AnyParams(BaseModel):
    of: list[str | dict[str, Any]] = Field(
        min_length=1, description="Disparadores; basta con que salte uno"
    )

    _of = field_validator("of")(lambda v: [_known(triggers, t) for t in v])


@triggers.register(
    "any",
    title="Cualquiera",
    description="Salta en cuanto salta uno de sus disparadores.",
    params=AnyParams,
)
class AnyOf:
    def __init__(self, of: list[str | dict[str, Any]]) -> None:
        self.of = [create(triggers, t) for t in of]

    def fired(self, view: View) -> str | None:
        return next((name for t in self.of if (name := t.fired(view))), None)


def drift_detectors(*tracked: Any) -> dict[str, Any]:
    """The detector each kind of drift needs, from every trigger given
    (``any`` included); one kind cannot have two different detectors."""
    found: dict[str, Any] = {}

    def walk(trigger: Any) -> None:
        if isinstance(trigger, AnyOf):
            for inner in trigger.of:
                walk(inner)
        elif isinstance(trigger, Drift):
            if trigger.kind in found and found[trigger.kind] != trigger.detector:
                raise ValueError(
                    f"drift {trigger.kind!r} has two detectors, {found[trigger.kind]} "
                    f"and {trigger.detector}; give both triggers the same"
                )
            found[trigger.kind] = trigger.detector

    for trigger in tracked:
        if trigger is not None:
            walk(trigger)
    return found
