from __future__ import annotations

"""The state of a federation between runs, to continue one (continuum, spec §6).

A snapshot holds, per node, arrays (no pickles) and JSON-ready metadata:
the coordinator's global model, round and server optimizer; each aggregator's
zone groups and own state (the central DP budget); each edge's model, trainer
memory and local DP budget; and every node's random stream. Messages in flight
when a run ended are not kept.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from onion_fl.core.context import restore_rng, rng_state
from onion_fl.learning.model import load_arrays, state_arrays
from onion_fl.roles.federation import Federation


@dataclass
class NodeState:
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FederationSnapshot:
    round: int  # the last round the coordinator closed
    nodes: dict[str, NodeState]
    root: str = ""  # the coordinator's id


def _prefixed(prefix: str, arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {f"{prefix}/{key}": np.asarray(value) for key, value in arrays.items()}


def _part(arrays: Mapping[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    head = f"{prefix}/"
    return {k[len(head) :]: v for k, v in arrays.items() if k.startswith(head)}


def _collector(node: Any, runtime: Any) -> NodeState:
    arrays = _prefixed("previous", node.previous or {})
    arrays |= _prefixed("aggregator", getattr(node.aggregator, "state", dict)())
    return NodeState(arrays, {"rng": rng_state(runtime._rng(node.id))})


def snapshot_federation(federation: Federation) -> FederationSnapshot:
    """Everything a later run needs to continue this one exactly."""
    runtime, root = federation.runtime, federation.coordinator
    nodes: dict[str, NodeState] = {}
    state = _collector(root, runtime)
    state.arrays |= _prefixed("state", root.state)
    state.arrays |= _prefixed("server", getattr(root.server_optimizer, "state", dict)())
    nodes[root.id] = state
    for node_id, aggregator in federation.aggregators.items():
        state = _collector(aggregator, runtime)
        state.arrays |= _prefixed("zone", aggregator.zone)
        nodes[node_id] = state
    for node_id, edge in federation.edges.items():
        state = NodeState(
            _prefixed("model", state_arrays(edge.model)) if edge.model else {},
            {"rng": rng_state(runtime._rng(node_id))},
        )
        state.arrays |= _prefixed("privacy", getattr(edge.privacy, "state", dict)())
        # The replay memory is saved; restoring it waits for streams to continue.
        if getattr(edge, "replay", None) is not None:
            replay, meta = edge.replay.state()
            state.arrays |= _prefixed("replay", replay)
            state.meta["replay"] = meta
        export = getattr(edge.trainer, "export_memory", None)
        if export is not None:
            memory, meta = export()
            state.arrays |= _prefixed("memory", memory)
            state.meta["memory"] = meta
        nodes[node_id] = state
    return FederationSnapshot(round=root.round, nodes=nodes, root=root.id)


def restore_federation(federation: Federation, snapshot: FederationSnapshot) -> None:
    """Continue ``snapshot`` on a federation built from the same specs, before it runs.

    The round numbering continues. A node the snapshot does not hold starts
    fresh (a new edge); a node of the snapshot that is gone here is ignored; a
    node saved without some part (no model, no random stream) keeps its own.

    Exact only for what the snapshot holds: synchronous rounds over lossless
    links, with no time-based availability. Link streams, the virtual clock and
    late updates already buffered for the next round start afresh.
    """
    runtime, root = federation.runtime, federation.coordinator
    root.round = snapshot.round
    for node_id, node in [(root.id, root), *federation.aggregators.items()]:
        saved = snapshot.nodes.get(node_id)
        if saved is None:
            continue
        node.previous = _part(saved.arrays, "previous") or None
        load = getattr(node.aggregator, "load_state", None)
        if load is not None:
            load(_part(saved.arrays, "aggregator"))
        if "rng" in saved.meta:
            restore_rng(runtime._rng(node_id), saved.meta["rng"])
        if node is root:
            # New datasets keep their initial keys; the parent's override the rest.
            root.state = {**root.state, **_part(saved.arrays, "state")}
            load = getattr(root.server_optimizer, "load_state", None)
            if load is not None:
                load(_part(saved.arrays, "server"))
        else:
            node.zone = _part(saved.arrays, "zone")
    for node_id, edge in federation.edges.items():
        saved = snapshot.nodes.get(node_id)
        if saved is None:
            continue
        model = _part(saved.arrays, "model")
        if edge.model is not None and model:
            load_arrays(edge.model, model)
        if "memory" in saved.meta and hasattr(edge.trainer, "import_memory"):
            edge.trainer.import_memory(
                _part(saved.arrays, "memory"), saved.meta["memory"], edge.model
            )
        load = getattr(edge.privacy, "load_state", None)
        if load is not None:
            load(_part(saved.arrays, "privacy"))
        if "rng" in saved.meta:
            restore_rng(runtime._rng(node_id), saved.meta["rng"])
