from __future__ import annotations

"""Diagnostics each aggregator computes when it closes a round (spec §10.4).

A plugin gets a ``RoundView`` and returns ``(name, value, tags)`` triples,
emitted as ``diagnostic.<name>``. A child's delta is its contribution minus the
model it was sent, flattened per parameter group.

``contribution`` values each child's update with the aggregator's own view only:
no validation or test data reaches an aggregator, and scoring coalitions on the
zone evaluators would cost a message round trip per coalition.

Communication (bytes and messages per link and round) is not here: the
aggregators never see encoded sizes, so ``observability.run`` derives it from
the message events.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Any
from zlib import crc32

import numpy as np
from pydantic import BaseModel, Field

from onion_fl.core.registry import Registry
from onion_fl.learning.aggregators import Contribution
from onion_fl.learning.model import group_of

State = Mapping[str, np.ndarray]
Item = tuple[str, float, dict[str, Any]]


@dataclass(frozen=True)
class RoundView:
    round: int
    sent: State  # what this aggregator sent down
    contributions: Mapping[str, Contribution]  # fresh, non-empty updates by child
    aggregated: State  # the combined state, empty if the round failed
    previous: State | None  # the combined state of the last successful round
    received: State  # what this aggregator got from its parent (empty at the root)
    datasets: Mapping[str, str]  # child -> dataset tag, when the child declared one
    reports: Mapping[str, Mapping[str, float]]  # child -> metrics of its update
    participants: Sequence[str]
    late: int
    quorum_failed: bool
    time_to_quorum: float | None


def _by_group(state: State, keys: Sequence[str]) -> dict[str, np.ndarray]:
    groups: dict[str, list[np.ndarray]] = {}
    for key in sorted(keys):
        groups.setdefault(group_of(key), []).append(
            np.ravel(state[key]).astype(np.float64)
        )
    return {g: np.concatenate(parts) for g, parts in groups.items()}


def deltas(view: RoundView) -> dict[str, dict[str, np.ndarray]]:
    """group -> child -> flattened delta against what the child was sent."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for child, contribution in sorted(view.contributions.items()):
        keys = [k for k in contribution.state if k in view.sent]
        diff = {
            k: np.asarray(contribution.state[k], np.float64) - view.sent[k]
            for k in keys
        }
        for group, vector in _by_group(diff, keys).items():
            out.setdefault(group, {})[child] = vector
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float | None:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return None if norm == 0 else float(a @ b / norm)


def _distance(a: State, b: State) -> dict[str, float]:
    keys = [k for k in a if k in b]
    left, right = _by_group(a, keys), _by_group(b, keys)
    return {g: float(np.linalg.norm(left[g] - right[g])) for g in left}


diagnostics = Registry("diagnostic")


@diagnostics.register(
    "divergence",
    title="Divergencia",
    description="Coseno medio entre los Δ de los hijos y su distancia L2 media al Δ medio, por grupo.",
)
class Divergence:
    def compute(self, view: RoundView) -> list[Item]:
        out: list[Item] = []
        for group, by_child in deltas(view).items():
            vectors = list(by_child.values())
            if len(vectors) < 2:
                continue
            cosines = [
                c for a, b in combinations(vectors, 2) if (c := _cos(a, b)) is not None
            ]
            mean = np.mean(vectors, axis=0)
            l2 = float(np.mean([np.linalg.norm(v - mean) for v in vectors]))
            if cosines:
                out.append(
                    ("divergence_cos", float(np.mean(cosines)), {"group": group})
                )
            out.append(("divergence_l2", l2, {"group": group}))
        return out


@diagnostics.register(
    "dataset_conflict",
    title="Conflicto entre datasets",
    description="Coseno entre el Δ medio de cada par de datasets en los grupos que comparten.",
    explain="Valores negativos: los datasets empujan el grupo compartido en sentidos opuestos.",
)
class DatasetConflict:
    def compute(self, view: RoundView) -> list[Item]:
        out: list[Item] = []
        for group, by_child in deltas(view).items():
            per_dataset: dict[str, list[np.ndarray]] = {}
            for child, vector in by_child.items():
                if child in view.datasets:
                    per_dataset.setdefault(view.datasets[child], []).append(vector)
            means = {d: np.mean(v, axis=0) for d, v in sorted(per_dataset.items())}
            for a, b in combinations(means, 2):
                cos = _cos(means[a], means[b])
                if cos is not None:
                    out.append(
                        (
                            "dataset_conflict",
                            cos,
                            {"group": group, "datasets": f"{a}|{b}"},
                        )
                    )
        return out


@diagnostics.register(
    "drift",
    title="Deriva",
    description="Cambio por grupo respecto a la ronda anterior y distancia del modelo de zona al recibido.",
)
class Drift:
    def compute(self, view: RoundView) -> list[Item]:
        out: list[Item] = []
        if view.aggregated and view.previous:
            for group, value in _distance(view.aggregated, view.previous).items():
                out.append(("drift", value, {"group": group}))
        if view.aggregated and view.received:
            for group, value in _distance(view.aggregated, view.received).items():
                out.append(("zone_distance", value, {"group": group}))
        return out


@diagnostics.register(
    "participation",
    title="Participación",
    description="Fracción de seleccionados que respondió, más tardíos, quórum y tiempo hasta el quórum.",
)
class Participation:
    def compute(self, view: RoundView) -> list[Item]:
        selected = len(view.participants)
        responded = len(view.contributions)
        tags = {
            "selected": selected,
            "responded": responded,
            "late": view.late,
            "quorum_failed": view.quorum_failed,
            "time_to_quorum": view.time_to_quorum,
        }
        return [("participation", responded / selected if selected else 0.0, tags)]


@diagnostics.register(
    "fairness",
    title="Equidad",
    description="Mínimo, máximo y desviación de las métricas de los hijos, por dataset.",
)
class Fairness:
    def compute(self, view: RoundView) -> list[Item]:
        values: dict[tuple[str, str], list[float]] = {}
        for child, metrics in sorted(view.reports.items()):
            dataset = view.datasets.get(child, "*")
            for key, value in metrics.items():
                if key.startswith("eval.") and not key.endswith(".samples"):
                    values.setdefault((key[len("eval.") :], dataset), []).append(
                        float(value)
                    )
        return [
            (
                "fairness",
                float(np.std(v)),
                {"metric": m, "dataset": d, "min": min(v), "max": max(v)},
            )
            for (m, d), v in sorted(values.items())
        ]


class ContributionParams(BaseModel):
    exact_max: int = Field(
        10,
        ge=1,
        le=16,
        description="Hasta cuántos hijos se calcula el Shapley exacto (2^n coaliciones)",
    )
    permutations: int = Field(
        128, ge=1, description="Permutaciones de Monte Carlo con más hijos"
    )
    seed: int = Field(0, description="Semilla de las permutaciones")


def coherence(units: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """v(S) = ‖Σ_{i∈S} u_i‖ for each row of ``mask`` (0/1 per child)."""
    gram = units @ units.T
    quad = np.einsum("ki,ij,kj->k", mask, gram, mask)
    return np.sqrt(np.maximum(quad, 0.0))


def _exact_shapley(units: np.ndarray) -> np.ndarray:
    n = len(units)
    masks = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    values = coherence(units, masks.astype(np.float64))
    sizes = masks.sum(axis=1)
    weight = np.array(
        [factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)]
    )
    out = np.zeros(n)
    for i in range(n):
        without = np.flatnonzero(masks[:, i] == 0)
        out[i] = np.sum(
            weight[sizes[without]] * (values[without | (1 << i)] - values[without])
        )
    return out


def _sampled_shapley(units: np.ndarray, permutations: int, rng) -> np.ndarray:
    n = len(units)
    gram = units @ units.T
    out = np.zeros(n)
    for _ in range(permutations):
        order = rng.permutation(n)
        quad, inside, value = 0.0, np.zeros(n), 0.0
        for i in order:
            quad += 2 * inside @ gram[i] + gram[i, i]
            inside[i] = 1.0
            new = float(np.sqrt(max(quad, 0.0)))
            out[i] += new - value
            value = new
    return out / permutations


@diagnostics.register(
    "contribution",
    title="Contribución",
    description=(
        "Valor de cada hijo por grupo: leave-one-out y Shapley (exacto o por "
        "Monte Carlo) en el juego v(S) = ‖Σ_{i∈S} Δ_i/‖Δ_i‖‖ − √|S|."
    ),
    params=ContributionParams,
    explain=(
        "Solo usa las actualizaciones que ya ve el agregador, sin datos de "
        "validación ni de test. Un Δ que apunta como el resto puntúa alto, uno "
        "ortogonal (ruido) cerca de 0 y uno en contra, negativo."
    ),
)
class ContributionValue:
    """Leave-one-out and Shapley values of each child in a coherence game.

    Each child's delta becomes a unit vector, so neither the size of an update
    nor its samples buy value: only its direction relative to the others. The
    game is v(S) = ‖Σ_{i∈S} u_i‖ − √|S|, the coherence of S beyond that of
    mutually orthogonal vectors (independent noise in many dimensions), which
    is 0. A zero delta is a zero vector and does not count in |S|, so it scores
    0. The Shapley values add up to v(N). Leave-one-out is v(N) − v(N∖i), the
    aggregated direction with and without the child.

    The √|S| term is symmetric, so its share is exact (1/√m among the m
    non-zero children); only the norm term is sampled beyond ``exact_max``.
    """

    def __init__(
        self, exact_max: int = 10, permutations: int = 128, seed: int = 0
    ) -> None:
        self.exact_max, self.permutations, self.seed = exact_max, permutations, seed

    def compute(self, view: RoundView) -> list[Item]:
        out: list[Item] = []
        for group, by_child in deltas(view).items():
            children = list(by_child)
            n = len(children)
            if n < 2:
                continue
            vectors = np.stack([by_child[c] for c in children])
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            units = np.divide(
                vectors, norms, out=np.zeros_like(vectors), where=norms > 0
            )
            full = coherence(units, np.ones((1, n)))[0]
            loo = full - coherence(units, 1.0 - np.eye(n))
            if n <= self.exact_max:
                shapley, method = _exact_shapley(units), "exact"
            else:
                rng = np.random.default_rng(
                    [self.seed, view.round, crc32(group.encode())]
                )
                shapley = _sampled_shapley(units, self.permutations, rng)
                method = "monte_carlo"
            nonzero = (norms[:, 0] > 0).astype(np.float64)
            m = nonzero.sum()
            if m:
                shapley = shapley - nonzero / np.sqrt(m)
                loo = loo - nonzero * (np.sqrt(m) - np.sqrt(m - 1))
            for i, name in enumerate(children):
                tags: dict[str, Any] = {"group": group, "child": name, "n": n}
                if name in view.datasets:
                    tags["dataset"] = view.datasets[name]
                out.append(("contribution_loo", float(loo[i]), tags))
                out.append(
                    (
                        "contribution_shapley",
                        float(shapley[i]),
                        tags | {"method": method},
                    )
                )
        return out
