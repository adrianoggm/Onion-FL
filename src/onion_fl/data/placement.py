from __future__ import annotations

"""Placement: which leaf aggregator gets which client and which zone evaluator (spec §8.5).

A placement plugin receives the items of one kind (training clients, or val
subjects that become zone evaluators), the leaf aggregators with their home
dataset and an ``rng``, and returns ``leaf id -> item ids``. Leaves are sorted
by id, so the same ``topology_id`` and seed always give the same scenario.

The home dataset of a leaf is its ``home`` setting, or the nearest ancestor's,
or one assigned in turns over the loaded datasets.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from onion_fl.core.context import node_rng
from onion_fl.core.registry import Registry
from onion_fl.core.topology import Topology
from onion_fl.data.contract import DataError, SubjectData, natural_key
from onion_fl.data.roles import Client, DataSplit


@dataclass(frozen=True)
class Item:
    id: str
    dataset: str
    samples: int
    class_counts: tuple[int, ...]


@dataclass(frozen=True)
class Leaf:
    id: str
    home: str


def largest_remainder(
    weights: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Integer counts summing to ``n`` in proportion to ``weights``; ties broken by ``rng``."""
    raw = weights / weights.sum() * n
    counts = np.floor(raw + 1e-9).astype(int)
    order = np.lexsort((rng.random(len(raw)), -(raw - counts)))
    counts[order[: n - counts.sum()]] += 1
    return counts


def _spread(
    items: Sequence[Item],
    leaves: Sequence[Leaf],
    weights_of: Any,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """Each dataset's items, shuffled, cut in the counts its weights give."""
    out: dict[str, list[str]] = {leaf.id: [] for leaf in leaves}
    for dataset in sorted({item.dataset for item in items}):
        ids = sorted((i.id for i in items if i.dataset == dataset), key=natural_key)
        ids = [ids[k] for k in rng.permutation(len(ids))]
        counts = largest_remainder(weights_of(dataset), len(ids), rng)
        start = 0
        for leaf, count in zip(leaves, counts, strict=True):
            out[leaf.id] += ids[start : start + count]
            start += count
    return out


placements = Registry("placement")


class MixingParams(BaseModel):
    alpha: float = Field(0.0, ge=0, le=1, description="0 segregado, 1 uniforme")


@placements.register(
    "mixing",
    title="Mezcla α",
    description="Cada dataset va a sus hojas de casa en proporción 1−α y al resto en α.",
    params=MixingParams,
    explain="w_f(d) = (1−α)·casa_f(d) + α/F. Con α=0 sale segregado; con α=1, todas las hojas igual.",
)
class Mixing:
    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def assign(self, items, leaves, rng):
        def weights(dataset: str) -> np.ndarray:
            home = np.array([leaf.home == dataset for leaf in leaves], dtype=float)
            if not home.any():
                return np.ones(len(leaves))  # no home leaf: always uniform
            return (1 - self.alpha) * home / home.sum() + self.alpha / len(leaves)

        return _spread(items, leaves, weights, rng)


class DirichletParams(BaseModel):
    beta: float = Field(1.0, gt=0, description="Pequeño: concentrado; grande: uniforme")


@placements.register(
    "dirichlet",
    title="Dirichlet β",
    description="Las proporciones de cada dataset entre hojas salen de Dir(β).",
    params=DirichletParams,
)
class Dirichlet:
    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def assign(self, items, leaves, rng):
        drawn: dict[str, np.ndarray] = {}

        def weights(dataset: str) -> np.ndarray:
            if dataset not in drawn:
                drawn[dataset] = rng.dirichlet(np.full(len(leaves), self.beta))
            return drawn[dataset]

        return _spread(items, leaves, weights, rng)


@placements.register(
    "label_skew",
    title="Sesgo de etiquetas β",
    description="Cada hoja tiene una proporción de clases objetivo ~ Dir(β); los clientes se asignan de forma voraz.",
    params=DirichletParams,
    explain="Las hojas se llenan hasta una cuota de muestras igual para todas.",
)
class LabelSkew:
    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def assign(self, items, leaves, rng):
        n_classes = max((len(i.class_counts) for i in items), default=2)
        targets = rng.dirichlet(np.full(n_classes, self.beta), size=len(leaves))
        tie = rng.permutation(len(leaves))  # fixed preference among equal scores
        quota = sum(i.samples for i in items) / len(leaves)
        load = np.zeros((len(leaves), n_classes))
        out: dict[str, list[str]] = {leaf.id: [] for leaf in leaves}
        ordered = sorted(items, key=lambda i: (-i.samples, natural_key(i.id)))
        for item in ordered:
            counts = np.zeros(n_classes)
            counts[: len(item.class_counts)] = item.class_counts
            open_ = [f for f in range(len(leaves)) if load[f].sum() < quota] or list(
                range(len(leaves))
            )

            def score(f: int, counts: np.ndarray = counts) -> tuple[float, int]:
                after = load[f] + counts
                share = after / after.sum() if after.sum() else after
                return float(np.abs(share - targets[f]).sum()), int(tie[f])

            best = min(open_, key=score)
            load[best] += counts
            out[leaves[best].id].append(item.id)
        return out


class ExplicitParams(BaseModel):
    assignment: dict[str, list[str]] = Field(
        description="Hoja -> identificadores de clientes y de evaluadores de zona"
    )


@placements.register(
    "explicit",
    title="Explícito",
    description="Listas de clientes (y evaluadores de zona) por hoja.",
    params=ExplicitParams,
)
class Explicit:
    def __init__(self, assignment: dict[str, list[str]]) -> None:
        self.assignment = assignment
        self.ids = {i for ids in assignment.values() for i in ids}

    def assign(self, items, leaves, rng):
        known = {item.id for item in items}
        return {
            leaf: [i for i in ids if i in known]
            for leaf, ids in self.assignment.items()
        }


class PooledParams(BaseModel):
    leaf: str | None = Field(
        None, description="Hoja que recibe todo; por defecto la primera"
    )


@placements.register(
    "pooled",
    title="Centralizado",
    description="Todos los datos de entrenamiento en un único edge por dataset.",
    params=PooledParams,
    explain="El baseline centralizado comparable: mismo modelo y mismo entrenador.",
)
class Pooled:
    merge = True

    def __init__(self, leaf: str | None = None) -> None:
        self.leaf = leaf

    def assign(self, items, leaves, rng):
        target = self.leaf or leaves[0].id
        return {
            leaf.id: [i.id for i in items] if leaf.id == target else []
            for leaf in leaves
        }


# --- the scenario ------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    name: str
    edges: dict[str, list[Client]]  # leaf id -> its edges
    zone_evaluators: dict[str, list[SubjectData]]
    test: list[SubjectData]  # global evaluators, at the root

    def composition(self) -> dict[str, dict[str, Any]]:
        out = {}
        for leaf, clients in self.edges.items():
            datasets: dict[str, int] = {}
            width = max((c.train.n_classes for c in clients), default=0)
            classes = np.zeros(width, dtype=int)
            for client in clients:
                datasets[client.dataset] = (
                    datasets.get(client.dataset, 0) + client.train.n_samples
                )
                counts = client.train.class_counts
                classes[: len(counts)] += counts
            total = sum(datasets.values())
            entropy = (
                -sum(n / total * math.log2(n / total) for n in datasets.values() if n)
                if total
                else 0.0
            )
            out[leaf] = {
                "clients": len(clients),
                "subjects": sum(len(c.subjects) for c in clients),
                "samples": total,
                "class_counts": classes.tolist(),
                "datasets": dict(sorted(datasets.items())),
                "entropy": abs(entropy),
                "evaluators": len(self.zone_evaluators.get(leaf, [])),
            }
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "placement": self.name,
            "edges": {leaf: [c.id for c in cs] for leaf, cs in self.edges.items()},
            "zone_evaluators": {
                leaf: [f"{e.dataset}-{e.subject}" for e in es]
                for leaf, es in self.zone_evaluators.items()
            },
            "composition": self.composition(),
        }


def _leaves(topology: Topology, datasets: list[str]) -> list[Leaf]:
    specs = sorted(topology.leaves(), key=lambda n: natural_key(n.id))
    declared: dict[str, str | None] = {}
    for spec in specs:
        node, home = spec, None
        while node is not None and home is None:
            home = node.settings.get("home")
            node = topology.parent(node.id)
        declared[spec.id] = home
    unknown = sorted({h for h in declared.values() if h} - set(datasets))
    if unknown:
        raise DataError(f"home datasets {unknown} are not loaded; loaded: {datasets}")
    turns = iter([datasets[k % len(datasets)] for k in range(len(specs))])
    return [Leaf(s.id, declared[s.id] or next(turns)) for s in specs]


def _check(assigned: Mapping[str, list[str]], items: Sequence[Item], leaves) -> None:
    leaf_ids = {leaf.id for leaf in leaves}
    stray = sorted(set(assigned) - leaf_ids)
    if stray:
        raise DataError(f"unknown leaf aggregators {stray}; leaves: {sorted(leaf_ids)}")
    seen = [i for ids in assigned.values() for i in ids]
    twice = sorted({i for i in seen if seen.count(i) > 1})
    if twice:
        raise DataError(f"assigned twice: {twice}")
    missing = sorted({i.id for i in items} - set(seen), key=natural_key)
    if missing:
        raise DataError(f"not assigned to any leaf: {missing}")


def _join(parts: list[SubjectData], subject: str) -> SubjectData:
    first = parts[0]
    return SubjectData(
        X=np.concatenate([p.X for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        dataset=first.dataset,
        subject=subject,
        task=first.task,
        n_classes=first.n_classes,
        feature_names=first.feature_names,
    )


def _pool(clients: list[Client]) -> list[Client]:
    """One edge per dataset holding every client of that dataset."""
    merged = []
    for dataset in sorted({c.dataset for c in clients}):
        group = [c for c in clients if c.dataset == dataset]
        client_id = f"{dataset}-pooled"
        tails = [c.local_val for c in group if c.local_val is not None]
        merged.append(
            Client(
                id=client_id,
                dataset=dataset,
                subjects=tuple(s for c in group for s in c.subjects),
                train=_join([c.train for c in group], client_id),
                local_val=_join(tails, client_id) if tails else None,
            )
        )
    return merged


def place(
    split: DataSplit,
    topology: Topology,
    placement: str = "mixing",
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> Placement:
    """Build the scenario: edges and zone evaluators under each leaf aggregator."""
    plugin = placements.create(placement, params)
    datasets = sorted(
        {c.dataset for c in split.clients} | {v.dataset for v in split.val}
    )
    leaves = _leaves(topology, datasets)
    clients = {c.id: c for c in split.clients}
    evaluators = {f"{v.dataset}-{v.subject}": v for v in split.val}
    unknown = sorted(getattr(plugin, "ids", set()) - set(clients) - set(evaluators))
    if unknown:
        raise DataError(f"unknown clients or evaluators {unknown}")

    def run(pool: Mapping[str, SubjectData], stream: str) -> dict[str, list[str]]:
        items = [
            Item(key, d.dataset, d.n_samples, tuple(d.class_counts))
            for key, d in pool.items()
        ]
        if not items:
            return {leaf.id: [] for leaf in leaves}
        assigned = plugin.assign(items, leaves, node_rng(seed, f"placement/{stream}"))
        _check(assigned, items, leaves)
        return {
            leaf.id: sorted(assigned.get(leaf.id, []), key=natural_key)
            for leaf in leaves
        }

    edges = {
        leaf: [clients[i] for i in ids]
        for leaf, ids in run(
            {k: c.train for k, c in clients.items()}, "clients"
        ).items()
    }
    if getattr(plugin, "merge", False):
        edges = {leaf: _pool(cs) if cs else [] for leaf, cs in edges.items()}
    zone = {
        leaf: [evaluators[i] for i in ids]
        for leaf, ids in run(evaluators, "val").items()
    }
    return Placement(
        name=placement, edges=edges, zone_evaluators=zone, test=list(split.test)
    )
