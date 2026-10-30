from __future__ import annotations

"""Build the nodes and links of a topology on a runtime (spec §6, §9).

Each aggregation node reads its round settings from the topology::

    aggregator: fedavg | {name: trimmed_mean, beta: 0.2}
    server_optimizer: replace        # root only
    quorum: 1.0 | 2                  # float = fraction, int = count
    close_at_quorum: false           # true: close on K updates, late ones included (FedBuff-style)
    deadline: 30s
    participation: all | {name: fraction, p: 0.5}
    staleness: drop | {name: next_round, weighting: {name: polynomial, a: 0.5}}
    register_timeout: 10s
    hello_retry: 5s                  # repeat the hello until acknowledged (null: never)
    eval: {every: 1, aggregate_children: true, holdout: true}

The edge template takes ``eval: {every: 1, models: [received, local]}``.
Edges hang from the leaf aggregators and use the topology's edge link; the
root takes evaluators only: the ``test`` subjects, or the validation ones with
``evaluation.global.subjects: val`` (hyperparameter selection).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from onion_fl.core.topology import Topology
from onion_fl.learning.aggregators import aggregators, server_optimizers
from onion_fl.learning.metrics import evaluate as score
from onion_fl.learning.sharing import SharingPolicy
from onion_fl.learning.sharing import sharing as sharing_presets
from onion_fl.learning.trainers import trainers as trainer_plugins
from onion_fl.observability.diagnostics import diagnostics as diagnostic_plugins
from onion_fl.roles.nodes import Aggregator, Coordinator, Edge, Evaluate
from onion_fl.roles.policies import (
    create,
    parse_duration,
    participations,
    stalenesses,
)
from onion_fl.runtime.sim import SimRuntime


@dataclass
class EdgeSpec:
    """One edge (or evaluator) under a leaf aggregator, or an evaluator under the root."""

    id: str
    model: Any
    data: Any = None
    trainer: Any = None
    train: bool = True
    compute: Any = None
    availability: Any = None
    val_data: Any = None
    tags: dict[str, Any] = field(default_factory=dict)
    attack: Any = None
    privacy: Any = None
    stream: Any = None  # an EdgeStream: rows arrive over time (continuum C3)
    replay: Any = None  # a replay memory of the stream's rows (continuum C4)
    replay_ratio: float = 0.25  # the share of each training that comes from it


@dataclass
class Federation:
    runtime: Any
    coordinator: Coordinator
    aggregators: dict[str, Aggregator] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    def run(self, until: float | None = None) -> None:
        self.runtime.run(until)


def _sharing(value: str | Mapping[str, Any] | SharingPolicy | None) -> SharingPolicy:
    if isinstance(value, SharingPolicy):
        return value
    return create(sharing_presets, value, default="fedavg")


def _diagnostics(value: Any) -> list[Any] | None:
    """``true`` (default) runs every diagnostic, ``false`` or ``[]`` none, a list those."""
    if value is True:
        return None
    if not value:
        return []
    return [create(diagnostic_plugins, item) for item in value]


def _round_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregator": create(aggregators, settings.get("aggregator"), "fedavg"),
        "quorum": settings.get("quorum", 1.0),
        "close_at_quorum": bool(settings.get("close_at_quorum", False)),
        "deadline": parse_duration(settings.get("deadline")),
        "participation": create(participations, settings.get("participation"), "all"),
        "staleness": create(stalenesses, settings.get("staleness"), "drop"),
        "register_timeout": parse_duration(settings.get("register_timeout")),
        "eval_every": (settings.get("eval") or {}).get("every"),
        "aggregate_children": (settings.get("eval") or {}).get(
            "aggregate_children", True
        ),
        "holdout": (settings.get("eval") or {}).get("holdout", True),
        "diagnostics": _diagnostics(settings.get("diagnostics", True)),
    }


def build_federation(
    topology: Topology,
    edges: Mapping[str, Sequence[EdgeSpec]],
    *,
    initial_state: Mapping[str, np.ndarray],
    rounds: int,
    sharing: str | Mapping[str, Any] | SharingPolicy | None = None,
    seed: int = 0,
    runtime: Any = None,
    metrics: Sequence[str] = ("loss", "accuracy"),
    evaluate: Evaluate | None = None,
    server_optimizer: Any = None,
    round_every: float | None = None,
    continuum: Any = None,
) -> Federation:
    """Coordinator, aggregators and edges of ``topology`` on ``runtime`` (a new SimRuntime).

    ``evaluate(model, data) -> (scores, samples)`` defaults to the ``metrics`` plugins.
    """
    policy = _sharing(sharing)
    policy.validate_for(topology)
    runtime = runtime or SimRuntime(seed=seed)
    root = topology.root
    leaves = {leaf.id for leaf in topology.leaves()}
    stray = sorted(set(edges) - leaves - {root.id})
    if stray:
        raise ValueError(f"edges under {stray}, which are not leaf aggregators")
    trainers_at_root = [e.id for e in edges.get(root.id, []) if e.train]
    if trainers_at_root:
        raise ValueError(
            f"the root {root.id!r} takes evaluators only, not {trainers_at_root}"
        )
    if evaluate is None:
        names = list(metrics)

        def evaluate(model: Any, data: Any) -> tuple[dict[str, float], int]:
            return score(model, data, names)

    edge_eval = topology.edge.settings.get("eval") or {}
    finetune = edge_eval.get("finetune")
    common = {"levels": topology.levels, "sharing": policy}
    children = {
        node.id: [c.id for c in topology.children(node.id)]
        + [e.id for e in edges.get(node.id, [])]
        for node in topology.nodes
    }
    coordinator = Coordinator(
        root.id,
        children[root.id],
        state=initial_state,
        rounds=rounds,
        round_every=round_every,
        continuum=continuum,
        server_optimizer=server_optimizer
        or create(server_optimizers, root.settings.get("server_optimizer"), "replace"),
        level=root.level,
        **common,
        **_round_settings(root.settings),
    )
    federation = Federation(runtime=runtime, coordinator=coordinator)
    runtime.add_node(coordinator)
    for node in topology.nodes:
        if node.parent is None:
            continue
        aggregator = Aggregator(
            node.id,
            children[node.id],
            parent=node.parent,
            hello_retry=parse_duration(node.settings.get("hello_retry", 5.0)),
            continuum=continuum,
            level=node.level,
            **common,
            **_round_settings(node.settings),
        )
        federation.aggregators[node.id] = aggregator
        runtime.add_node(aggregator)
    for leaf, specs in edges.items():
        for spec in specs:
            node = Edge(
                spec.id,
                leaf,
                model=spec.model,
                data=spec.data,
                trainer=spec.trainer,
                train=spec.train,
                val_data=spec.val_data,
                evaluate=evaluate,
                eval_every=edge_eval.get("every"),
                eval_models=edge_eval.get("models", ("received", "local")),
                attack=spec.attack,
                privacy=spec.privacy,
                stream=spec.stream,
                replay=spec.replay,
                replay_ratio=spec.replay_ratio,
                continuum=continuum,
                metrics=list(metrics),
                finetuner=(
                    create(trainer_plugins, finetune)
                    if finetune is not None and spec.train
                    else None
                ),
                tags=spec.tags,
                hello_retry=parse_duration(
                    topology.edge.settings.get("hello_retry", 5.0)
                ),
                **common,
            )
            federation.edges[spec.id] = node
            runtime.add_node(node, compute=spec.compute, availability=spec.availability)
    for node in topology.nodes:
        if node.parent is not None:
            link = node.link_up
            runtime.add_link(
                node.id,
                node.parent,
                profile=link.profile,
                codec=link.codec,
                transport=link.transport,
            )
    edge_link = topology.edge.link_up
    for leaf, specs in edges.items():
        for spec in specs:
            runtime.add_link(
                spec.id,
                leaf,
                profile=edge_link.profile,
                codec=edge_link.codec,
                transport=edge_link.transport,
            )
    return federation
