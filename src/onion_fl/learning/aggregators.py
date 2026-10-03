from __future__ import annotations

"""Per-key aggregators and server optimizers (spec §6.3, §7.3).

Aggregators work on arrays, never on models. Each key is combined only among
the contributions that carry it, and the summed sample weights travel up, so
a tree of ``fedavg`` aggregators gives the same result as one flat FedAvg.
Contributions are sorted by source before combining: the result does not
depend on the order in which they arrived.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from onion_fl.core.registry import Registry
from onion_fl.learning.model import is_aux


class AggregationError(ValueError):
    """Contributions cannot be combined."""


@dataclass(frozen=True)
class Contribution:
    """State and per-key sample weights coming from one child (or going to the parent)."""

    source: str
    state: Mapping[str, np.ndarray]
    weights: Mapping[str, float] = field(default_factory=dict)


Combine = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _combine_per_key(
    contributions: Sequence[Contribution],
    source: str,
    combine: Combine,
    needs_weights: bool,
) -> Contribution:
    if not contributions:
        raise AggregationError("no contributions to aggregate")
    ordered = sorted(contributions, key=lambda item: item.source)
    keys = sorted({key for item in ordered for key in item.state})
    state: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}
    for key in keys:
        holders = [item for item in ordered if key in item.state]
        shapes = {np.shape(item.state[key]) for item in holders}
        if len(shapes) != 1:
            raise AggregationError(
                f"{key}: contributions have different shapes {sorted(shapes)}"
            )
        missing = [item.source for item in holders if key not in item.weights]
        if needs_weights and missing:
            raise AggregationError(f"{key}: no sample weight from {missing}")
        stacked = np.stack(
            [np.asarray(item.state[key], dtype=np.float64) for item in holders]
        )
        sample_weights = np.array(
            [float(item.weights.get(key, 0.0)) for item in holders]
        )
        dtype = np.asarray(holders[0].state[key]).dtype
        state[key] = combine(stacked, sample_weights).astype(dtype)
        weights[key] = sum(item.weights.get(key, 0) for item in holders)
    return Contribution(source=source, state=state, weights=weights)


def _weighted_mean(stacked: np.ndarray, sample_weights: np.ndarray) -> np.ndarray:
    total = sample_weights.sum()
    if total <= 0:
        return stacked.mean(axis=0)
    shape = (-1,) + (1,) * (stacked.ndim - 1)
    return (stacked * sample_weights.reshape(shape)).sum(axis=0) / total


aggregators = Registry("aggregator")


@aggregators.register(
    "fedavg",
    title="FedAvg",
    description="Media ponderada por muestras, clave a clave.",
    explain="En cascada (edge → fog → cloud) da el mismo resultado que un FedAvg plano.",
)
class FedAvg:
    def aggregate(
        self, contributions: Sequence[Contribution], source: str
    ) -> Contribution:
        return _combine_per_key(
            contributions, source, _weighted_mean, needs_weights=True
        )


@aggregators.register(
    "mean",
    title="Media simple",
    description="Media sin ponderar: cada hijo cuenta lo mismo, tenga las muestras que tenga.",
)
class Mean:
    def aggregate(
        self, contributions: Sequence[Contribution], source: str
    ) -> Contribution:
        return _combine_per_key(
            contributions, source, lambda s, _w: s.mean(axis=0), False
        )


@aggregators.register(
    "median",
    title="Mediana",
    description="Mediana coordenada a coordenada.",
    explain="Robusta frente a actualizaciones extremas o maliciosas; no pondera por muestras.",
)
class Median:
    def aggregate(
        self, contributions: Sequence[Contribution], source: str
    ) -> Contribution:
        return _combine_per_key(
            contributions, source, lambda s, _w: np.median(s, axis=0), False
        )


class TrimmedMeanParams(BaseModel):
    beta: float = Field(
        0.1, ge=0, lt=0.5, description="Fracción recortada en cada extremo"
    )


@aggregators.register(
    "trimmed_mean",
    title="Media recortada",
    description="Descarta la fracción beta de valores más altos y más bajos de cada coordenada.",
    params=TrimmedMeanParams,
    explain="Con pocos hijos se conserva al menos el valor central.",
)
class TrimmedMean:
    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def _trim(self, stacked: np.ndarray, _weights: np.ndarray) -> np.ndarray:
        n = stacked.shape[0]
        cut = int(self.beta * n)  # beta < 0.5 keeps at least one value per coordinate
        ordered = np.sort(stacked, axis=0)
        return ordered[cut : n - cut].mean(axis=0)

    def aggregate(
        self, contributions: Sequence[Contribution], source: str
    ) -> Contribution:
        return _combine_per_key(contributions, source, self._trim, needs_weights=False)


# --- server optimizers -----------------------------------------------------------

server_optimizers = Registry("server_optimizer")
State = dict[str, np.ndarray]


@server_optimizers.register(
    "replace",
    title="Reemplazo (FedAvg clásico)",
    description="El modelo global pasa a ser el agregado.",
)
class Replace:
    def apply(
        self,
        global_state: Mapping[str, np.ndarray],
        aggregated: Mapping[str, np.ndarray],
        stats: Mapping[str, float] | None = None,
    ) -> State:
        return {**global_state, **aggregated}


class FedAvgMParams(BaseModel):
    server_lr: float = Field(1.0, gt=0)
    momentum: float = Field(0.9, ge=0, lt=1)


@server_optimizers.register(
    "fedavgm",
    title="FedAvgM",
    description="Momento de servidor sobre el pseudo-gradiente (agregado − global).",
    params=FedAvgMParams,
)
class FedAvgM:
    def __init__(self, server_lr: float = 1.0, momentum: float = 0.9) -> None:
        self.server_lr = server_lr
        self.momentum = momentum
        self._velocity: dict[str, np.ndarray] = {}

    def apply(
        self,
        global_state: Mapping[str, np.ndarray],
        aggregated: Mapping[str, np.ndarray],
        stats: Mapping[str, float] | None = None,
    ) -> State:
        new = dict(global_state)
        for key, value in aggregated.items():
            if is_aux(key):  # control variates and the like: replaced, never stepped
                new[key] = value
                continue
            current = np.asarray(global_state.get(key, value), dtype=np.float64)
            delta = np.asarray(value, dtype=np.float64) - current
            velocity = (
                self.momentum * self._velocity.get(key, np.zeros_like(delta)) + delta
            )
            self._velocity[key] = velocity
            new[key] = (current + self.server_lr * velocity).astype(
                np.asarray(value).dtype
            )
        return new


class FedAdamParams(BaseModel):
    server_lr: float = Field(0.01, gt=0)
    beta1: float = Field(0.9, ge=0, lt=1)
    beta2: float = Field(0.99, ge=0, lt=1)
    tau: float = Field(1e-3, gt=0, description="Término de estabilidad")


@server_optimizers.register(
    "fedadam",
    title="FedAdam",
    description="Adam de servidor sobre el pseudo-gradiente (Reddi et al., 2021).",
    params=FedAdamParams,
)
class FedAdam:
    def __init__(
        self,
        server_lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
    ) -> None:
        self.server_lr, self.beta1, self.beta2, self.tau = server_lr, beta1, beta2, tau
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}

    def apply(
        self,
        global_state: Mapping[str, np.ndarray],
        aggregated: Mapping[str, np.ndarray],
        stats: Mapping[str, float] | None = None,
    ) -> State:
        new = dict(global_state)
        for key, value in aggregated.items():
            if is_aux(key):  # control variates and the like: replaced, never stepped
                new[key] = value
                continue
            current = np.asarray(global_state.get(key, value), dtype=np.float64)
            delta = np.asarray(value, dtype=np.float64) - current
            m = (
                self.beta1 * self._m.get(key, np.zeros_like(delta))
                + (1 - self.beta1) * delta
            )
            v = (
                self.beta2 * self._v.get(key, np.zeros_like(delta))
                + (1 - self.beta2) * delta**2
            )
            self._m[key], self._v[key] = m, v
            step = self.server_lr * m / (np.sqrt(v) + self.tau)
            new[key] = (current + step).astype(np.asarray(value).dtype)
        return new
