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
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PositiveInt

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
        self,
        contributions: Sequence[Contribution],
        source: str,
        reference: Mapping[str, np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
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
        self,
        contributions: Sequence[Contribution],
        source: str,
        reference: Mapping[str, np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
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
        self,
        contributions: Sequence[Contribution],
        source: str,
        reference: Mapping[str, np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
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
        self,
        contributions: Sequence[Contribution],
        source: str,
        reference: Mapping[str, np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> Contribution:
        return _combine_per_key(contributions, source, self._trim, needs_weights=False)


# --- robust aggregation ------------------------------------------------------------


def _common_model_keys(ordered: Sequence[Contribution]) -> list[str]:
    keys = set.intersection(*(set(c.state) for c in ordered)) if ordered else set()
    return sorted(k for k in keys if not is_aux(k))


def _vectors(
    ordered: Sequence[Contribution],
    keys: Sequence[str],
    reference: Mapping[str, np.ndarray] | None,
) -> np.ndarray:
    """One flattened update per child over ``keys``: its delta against ``reference``."""
    rows = []
    for item in ordered:
        parts = []
        for key in keys:
            value = np.asarray(item.state[key], dtype=np.float64)
            if reference is not None and key in reference:
                value = value - np.asarray(reference[key], dtype=np.float64)
            parts.append(np.ravel(value))
        rows.append(np.concatenate(parts) if parts else np.zeros(0))
    return np.stack(rows)


def _krum_scores(vectors: np.ndarray, f: int) -> np.ndarray:
    n = len(vectors)
    nearest = max(n - f - 2, 1)
    distances = ((vectors[:, None, :] - vectors[None, :, :]) ** 2).sum(axis=-1)
    return np.array(
        [np.sort(np.delete(distances[i], i))[:nearest].sum() for i in range(n)]
    )


class _Selection:
    """Keeps its last selection so the collector can report it."""

    dropped: list[str]
    f_used: int

    def report(self) -> list[tuple[str, float, dict[str, Any]]]:
        return [
            (
                "aggregation.dropped",
                float(len(self.dropped)),
                {"dropped": list(self.dropped), "f_used": self.f_used},
            )
        ]

    def _keep(
        self, ordered: Sequence[Contribution], chosen: Sequence[int], source: str
    ) -> Contribution:
        keep = {int(i) for i in chosen}
        self.dropped = [c.source for i, c in enumerate(ordered) if i not in keep]
        kept = [c for i, c in enumerate(ordered) if i in keep]
        return _combine_per_key(kept, source, _weighted_mean, needs_weights=True)


class KrumParams(BaseModel):
    f: int = Field(1, ge=0, description="Hijos maliciosos que tolera")


@aggregators.register(
    "krum",
    title="Krum",
    description="Elige la actualización con menor suma de distancias a sus n − f − 2 vecinas.",
    params=KrumParams,
    explain=(
        "Puntúa sobre las claves de modelo que tienen todos los hijos y conserva una "
        "sola contribución; con menos de 2f + 3 hijos baja f (Blanchard et al., 2017)."
    ),
)
class Krum(_Selection):
    def __init__(self, f: int = 1) -> None:
        self.f = f

    def _feasible(self, n: int) -> int:
        return max(0, min(self.f, (n - 3) // 2))

    def _ranked(
        self,
        ordered: Sequence[Contribution],
        reference: Mapping[str, np.ndarray] | None,
    ) -> tuple[np.ndarray, int]:
        f = self._feasible(len(ordered))
        vectors = _vectors(ordered, _common_model_keys(ordered), reference)
        return np.argsort(_krum_scores(vectors, f), kind="stable"), f

    def aggregate(
        self,
        contributions: Sequence[Contribution],
        source: str,
        reference: Mapping[str, np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> Contribution:
        ordered = sorted(contributions, key=lambda item: item.source)
        ranked, self.f_used = self._ranked(ordered, reference)
        return self._keep(ordered, ranked[:1], source)


class MultiKrumParams(KrumParams):
    m: PositiveInt | None = Field(
        None, description="Cuántas conserva; por defecto n − f"
    )


@aggregators.register(
    "multi_krum",
    title="Multi-Krum",
    description="Promedia (FedAvg) las m actualizaciones mejor puntuadas por Krum.",
    params=MultiKrumParams,
    explain="Con menos de 2f + 3 hijos baja f (Blanchard et al., 2017).",
)
class MultiKrum(Krum):
    def __init__(self, f: int = 1, m: int | None = None) -> None:
        super().__init__(f)
        self.m = m

    def aggregate(
        self,
        contributions: Sequence[Contribution],
        source: str,
        reference: Mapping[str, np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> Contribution:
        ordered = sorted(contributions, key=lambda item: item.source)
        ranked, self.f_used = self._ranked(ordered, reference)
        keep = min(self.m or len(ordered) - self.f_used, len(ordered))
        return self._keep(ordered, ranked[:keep], source)


class GeometricMedianParams(BaseModel):
    iterations: PositiveInt = Field(10, description="Iteraciones de Weiszfeld")
    eps: float = Field(
        1e-6, gt=0, description="Distancia mínima (evita dividir por cero)"
    )


@aggregators.register(
    "geometric_median",
    title="Mediana geométrica",
    description="Punto que minimiza la suma de distancias a las actualizaciones (Weiszfeld).",
    params=GeometricMedianParams,
    explain=(
        "Pesa cada hijo por muestras / distancia a la mediana y aplica esos pesos a "
        "todas sus claves (RFA, Pillutla et al., 2022)."
    ),
)
class GeometricMedian:
    def __init__(self, iterations: int = 10, eps: float = 1e-6) -> None:
        self.iterations, self.eps = iterations, eps

    def aggregate(
        self,
        contributions: Sequence[Contribution],
        source: str,
        reference: Mapping[str, np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> Contribution:
        ordered = sorted(contributions, key=lambda item: item.source)
        keys = _common_model_keys(ordered)
        vectors = _vectors(ordered, keys, reference)
        alpha = np.array(
            [float(c.weights.get(keys[0], 1.0)) if keys else 1.0 for c in ordered]
        )
        beta = alpha / alpha.sum()
        for _ in range(self.iterations):
            z = (beta[:, None] * vectors).sum(axis=0)
            distance = np.maximum(np.linalg.norm(vectors - z, axis=1), self.eps)
            beta = alpha / distance
            beta = beta / beta.sum()
        weighted = [
            Contribution(item.source, item.state, dict.fromkeys(item.state, float(b)))
            for item, b in zip(ordered, beta, strict=True)
        ]
        out = _combine_per_key(weighted, source, _weighted_mean, needs_weights=True)
        totals = _combine_per_key(ordered, source, _weighted_mean, needs_weights=True)
        return Contribution(source, out.state, totals.weights)  # samples go up


@aggregators.register(
    "bulyan",
    title="Bulyan",
    description="Multi-Krum y después media recortada alrededor de la mediana, coordenada a coordenada.",
    params=KrumParams,
    explain="Con menos de 4f + 3 hijos baja f (El Mhamdi et al., 2018).",
)
class Bulyan(Krum):
    def _feasible(self, n: int) -> int:
        return max(0, min(self.f, (n - 3) // 4))

    def aggregate(
        self,
        contributions: Sequence[Contribution],
        source: str,
        reference: Mapping[str, np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> Contribution:
        ordered = sorted(contributions, key=lambda item: item.source)
        ranked, f = self._ranked(ordered, reference)
        self.f_used = f
        theta = len(ordered) - 2 * f
        keep = {int(i) for i in ranked[:theta]}
        selected = [c for i, c in enumerate(ordered) if i in keep]
        self.dropped = [c.source for i, c in enumerate(ordered) if i not in keep]
        beta = max(theta - 2 * f, 1)

        def around_median(stacked: np.ndarray, _weights: np.ndarray) -> np.ndarray:
            size = min(beta, stacked.shape[0])
            median = np.median(stacked, axis=0)
            order = np.argsort(np.abs(stacked - median), axis=0, kind="stable")[:size]
            return np.take_along_axis(stacked, order, axis=0).mean(axis=0)

        out = _combine_per_key(selected, source, around_median, needs_weights=False)
        totals = _combine_per_key(selected, source, _weighted_mean, needs_weights=True)
        return Contribution(source, out.state, totals.weights)


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


@server_optimizers.register(
    "fednova",
    title="FedNova",
    description="x + τ̄·d̄: la media de las actualizaciones normalizadas por la media de pasos.",
    explain="Va con el entrenador fednova; con pasos iguales coincide con FedAvg (Wang et al., 2020).",
)
class FedNovaOptimizer:
    PREFIX = "fednova/"
    STEPS = "fednova_steps/"

    def check_trainer(self, name: str, trainer: Any) -> None:
        if name != "fednova":
            raise ValueError(f"fednova needs the fednova trainer, not {name!r}")

    def apply(
        self,
        global_state: Mapping[str, np.ndarray],
        aggregated: Mapping[str, np.ndarray],
        stats: Mapping[str, float] | None = None,
    ) -> State:
        mean_steps = (stats or {}).get("train_steps")
        mine = (self.PREFIX, self.STEPS)
        new = {k: v for k, v in global_state.items() if not k.startswith(mine)}
        for key, value in aggregated.items():
            if key.startswith(mine):
                continue
            step = aggregated.get(self.PREFIX + key)
            if step is None:  # no normalised update (frozen or non-trainable): replace
                new[key] = value
                continue
            held = aggregated.get(self.STEPS + key)  # mean steps of this key's holders
            tau = float(np.ravel(held)[0]) if held is not None else mean_steps
            if not tau:
                raise ValueError("fednova needs train_steps in the round statistics")
            current = np.asarray(global_state.get(key, value), dtype=np.float64)
            new[key] = (current + tau * np.asarray(step, np.float64)).astype(
                np.asarray(value).dtype
            )
        return new


class FedDynOptimizerParams(BaseModel):
    alpha: float = Field(0.01, gt=0, description="El mismo α que el entrenador feddyn")


@server_optimizers.register(
    "feddyn",
    title="FedDyn",
    description="h ← h − α·(|P|/m)·(θ̄ − θ); θ ← θ̄ − h/α.",
    params=FedDynOptimizerParams,
    explain=(
        "Va con el entrenador feddyn y su mismo α; |P|/m sale de "
        "train_edges/edges_total, la fracción de toda la ronda, también para las "
        "claves de un solo dataset (Acar et al., 2021)."
    ),
)
class FedDynOptimizer:
    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        self._h: dict[str, np.ndarray] = {}

    def check_trainer(self, name: str, trainer: Any) -> None:
        if name != "feddyn":
            raise ValueError(f"feddyn needs the feddyn trainer, not {name!r}")
        if trainer.params.alpha != self.alpha:
            raise ValueError(
                f"alpha {self.alpha} differs from the trainer's {trainer.params.alpha}"
            )

    def apply(
        self,
        global_state: Mapping[str, np.ndarray],
        aggregated: Mapping[str, np.ndarray],
        stats: Mapping[str, float] | None = None,
    ) -> State:
        stats = stats or {}
        total = stats.get("edges_total") or 0.0
        share = stats.get("train_edges", total) / total if total else 1.0
        new = dict(global_state)
        for key, value in aggregated.items():
            if is_aux(key):
                new[key] = value
                continue
            mean = np.asarray(value, dtype=np.float64)
            current = np.asarray(global_state.get(key, value), dtype=np.float64)
            h = self._h.get(key, np.zeros_like(mean))
            h = h - self.alpha * share * (mean - current)
            self._h[key] = h
            new[key] = (mean - h / self.alpha).astype(np.asarray(value).dtype)
        return new
