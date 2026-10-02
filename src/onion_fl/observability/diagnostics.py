from __future__ import annotations

"""Diagnostics each aggregator computes when it closes a round (spec §10.4).

A plugin gets a ``RoundView`` and returns ``(name, value, tags)`` triples,
emitted as ``diagnostic.<name>``. A child's delta is its contribution minus the
model it was sent, flattened per parameter group.

Communication (bytes and messages per link and round) is not here: the
aggregators never see encoded sizes, so ``observability.run`` derives it from
the message events.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

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
