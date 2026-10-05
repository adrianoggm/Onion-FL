from __future__ import annotations

"""Streams: each edge's rows replayed in time order, with delayed labels (spec §4.1–4.3, §7).

Times here are continuum time: data time since the stream started, which the
simulation reaches as virtual seconds × ``stream.speed``.

- **History.** The rows before an edge's t₀ (its bootstrap) are there from the
  start, or from the edge's staggered start, and are never predicted.
- **Arrivals.** Later rows arrive in batches of ``batch_size``, each batch at the
  time of its last row.
- **Labels.** A seeded fraction of the rows is labelled. Each label arrives
  ``labels.delay`` after its row arrived (for history, after it was observed, so
  a label already due before t₀ is there at the start). A row becomes trainable
  once it has arrived and its label too.
"""

import math
from dataclasses import dataclass, replace
from typing import Annotated, Literal

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
)

from onion_fl.core.context import node_rng
from onion_fl.data.contract import SubjectData
from onion_fl.roles.policies import parse_duration

Seconds = Annotated[float, BeforeValidator(parse_duration)]
Positive = Annotated[Seconds, Field(gt=0)]


class Staggered(BaseModel):
    model_config = ConfigDict(extra="forbid")

    staggered: Positive = Field(
        description="Tiempo de datos entre la entrada de un edge y la del siguiente"
    )


class Samples(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: PositiveInt = Field(
        description="Filas de cada edge que forman su histórico"
    )


class StreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: Literal["timestamp"] = Field(
        "timestamp", description="Orden de llegada: el tiempo real de las filas"
    )
    batch_size: PositiveInt = Field(32, description="Filas que llegan juntas")
    speed: PositiveFloat = Field(
        60.0,
        description="Segundos de datos por segundo virtual: solo acelera la simulación",
    )
    start: Literal["aligned"] | Staggered = Field(
        "aligned",
        description="aligned: todos los edges a la vez; staggered: uno cada tanto",
    )
    horizon: Literal["session"] = Field(
        "session", description="Hasta el final de la grabación de cada sujeto"
    )
    bootstrap: Positive | Samples = Field(
        description="El histórico antes de t₀: ajusta el preprocesado y entrena v0"
    )
    round_every: Positive = Field(description="Tiempo de datos entre rondas")
    window: Positive | None = Field(
        None,
        description="Búfer de entrenamiento: lo que pasó a ser entrenable en este "
        "tiempo; por defecto, round_every",
    )


class LabelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fraction: float = Field(
        1.0, ge=0, le=1, description="Fracción de filas etiquetadas, por edge"
    )
    delay: Annotated[Seconds, Field(ge=0)] = Field(
        0.0, description="Retraso de cada etiqueta, en tiempo de datos"
    )


@dataclass(frozen=True, eq=False)
class EdgeStream:
    """One edge's rows, in time order, with when each arrives and can train."""

    data: SubjectData
    available_at: np.ndarray
    label_at: np.ndarray  # inf for a row that is never labelled
    history: np.ndarray  # before t₀: never predicted
    speed: float

    @property
    def trainable_at(self) -> np.ndarray:
        return np.maximum(self.available_at, self.label_at)

    @property
    def horizon(self) -> float:
        """When the last row arrives."""
        return float(self.available_at.max())

    def clock(self, now: float) -> float:
        """Continuum time at virtual time ``now``."""
        return now * self.speed

    def arrived(self, lo: float, hi: float) -> np.ndarray:
        return _within(self.available_at, lo, hi)

    def labelled(self, lo: float, hi: float) -> np.ndarray:
        return _within(self.label_at, lo, hi)

    def trainable(self, lo: float, hi: float) -> np.ndarray:
        return _within(self.trainable_at, lo, hi)

    def take(self, mask: np.ndarray) -> SubjectData:
        d = self.data
        return replace(d, X=d.X[mask], y=d.y[mask], t=d.t[mask])


def _within(times: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.isfinite(times) & (times > lo) & (times <= hi)


def edge_stream(
    data: SubjectData,
    stream: StreamConfig,
    labels: LabelsConfig,
    seed: int,
    offset: float = 0.0,
) -> EdgeStream:
    """The schedule of ``data`` as one edge's stream starting at ``offset``."""
    if data.t is None:
        raise ValueError(
            f"{data.dataset}/{data.subject} has no time per row: a stream needs a "
            "time step in its dataset descriptor"
        )
    order = np.argsort(data.t, kind="stable")
    data = replace(data, X=data.X[order], y=data.y[order], t=data.t[order])
    t, n = data.t, len(data.t)
    if isinstance(stream.bootstrap, Samples):
        history = np.arange(n) < stream.bootstrap.samples
        t0 = t[min(stream.bootstrap.samples, n - 1)]
    else:
        t0 = stream.bootstrap
        history = t < t0
    observed = t - t0 + offset
    arrival = observed.copy()
    live = np.flatnonzero(~history)
    if len(live):  # a batch arrives with its last row
        ends = (np.arange(len(live)) // stream.batch_size + 1) * stream.batch_size
        arrival[live] = observed[live][np.minimum(ends, len(live)) - 1]
    chosen = node_rng(seed, f"labels/{data.dataset}/{data.subject}").permutation(n)
    label_at = np.full(n, np.inf)
    labelled = chosen[: math.floor(labels.fraction * n + 0.5)]
    label_at[labelled] = arrival[labelled] + labels.delay
    return EdgeStream(
        data=data,
        available_at=np.maximum(arrival, offset),
        label_at=label_at,
        history=history,
        speed=stream.speed,
    )
