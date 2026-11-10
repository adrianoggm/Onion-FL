from __future__ import annotations

"""Turn a scenario into a recorded run, or into a dry-run plan (spec §11, §12, §9.2).

1. Topology: from ``topologies/<name>.yaml``, a path or inline; the runtime
   codec and the evaluation settings are applied (node settings win).
2. Data: every dataset through the cache, then roles and placement.
3. Model: the global model with every dataset's shape and the ``init`` plugin.
4. Federation: edges and zone evaluators under each leaf, test evaluators
   under the root; then it runs inside a ``Run`` that writes ``runs/<run_id>/``.
"""

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from onion_fl.continuum.bundle import Bundle, load_bundle, save_bundle
from onion_fl.continuum.memory import NoMemory, memories
from onion_fl.continuum.pace import Continuum
from onion_fl.continuum.promotion import promotions
from onion_fl.continuum.triggers import drift_detectors, triggers
from onion_fl.continuum.unlabelled import NoObjective, objectives
from onion_fl.core.context import node_rng
from onion_fl.core.topology import Topology, load_topology, parse_topology
from onion_fl.data.cache import load_prepared, prepare
from onion_fl.data.contract import natural_key
from onion_fl.data.ingest import load_spec
from onion_fl.data.placement import Placement, place
from onion_fl.data.roles import DataSplit, split_subjects
from onion_fl.data.stream import (
    LabelsConfig,
    Staggered,
    bootstrap_rows,
    edge_stream,
)
from onion_fl.experiment.config import ConfigError, ExperimentConfig
from onion_fl.experiment.sweep import Scenario, identity, scenarios
from onion_fl.infrastructure.assignment import (
    Coupled,
    Resolved,
    assignments,
    resolve,
)
from onion_fl.infrastructure.backends import backends, execute
from onion_fl.infrastructure.failover import NoFailover, failovers
from onion_fl.infrastructure.model import (
    Infrastructure,
    InfrastructureError,
    anchor_traces,
    load_infrastructure,
    parse_infrastructure,
)
from onion_fl.infrastructure.network import (
    PhysicalNetwork,
    downtime,
    host_events,
    travel,
)
from onion_fl.infrastructure.paths import LOOPBACK
from onion_fl.learning.aggregators import aggregators, server_optimizers
from onion_fl.learning.attacks import attacks
from onion_fl.learning.metrics import compute, predict
from onion_fl.learning.model import (
    group_of,
    is_aux,
    load_arrays,
    models,
    param_groups,
    state_arrays,
)
from onion_fl.learning.privacy import Accountant, privacies
from onion_fl.learning.sharing import not_local, sharing, traffic
from onion_fl.learning.trainers import frozen_keys, inits, trainers
from onion_fl.observability.run import Run, save_model, verify_run
from onion_fl.observability.sinks import OtelSink, PrometheusSink, otlp_provider
from onion_fl.roles import (
    EdgeSpec,
    FederationSnapshot,
    NodeState,
    build_federation,
    restore_federation,
    snapshot_federation,
)
from onion_fl.roles.policies import create, stalenesses
from onion_fl.runtime.devices import availability_models, compute_models
from onion_fl.runtime.network import PathChannel, resolve_profile

ROLES = ("train", "val", "test")  # kept apart along a lineage


def resolve_topology(config: ExperimentConfig) -> Topology:
    ref = config.topology
    if isinstance(ref, dict):
        topology = parse_topology(ref)
    else:
        path = Path(ref)
        if path.suffix not in (".yaml", ".yml"):
            path = Path(config.paths.topologies) / f"{ref}.yaml"
        topology = load_topology(path)
    if config.infrastructure is not None:
        _single_source(topology)
    general = topology.model_dump()
    evaluation = config.evaluation
    for node in general["nodes"]:
        defaults = (
            evaluation.global_ if node["parent"] is None else evaluation.aggregators
        )
        node["settings"] = {"eval": defaults.model_dump(exclude_none=True)} | node[
            "settings"
        ]
        if config.runtime.codec and node["link_up"] is not None:
            node["link_up"]["codec"] = config.runtime.codec
    if config.learning.aggregator is not None:
        parents = {n["parent"] for n in general["nodes"]}
        for node in general["nodes"]:
            if node["id"] not in parents:  # a leaf aggregator: its children are edges
                node["settings"]["aggregator"] = config.learning.aggregator
    if config.learning.server_optimizer is not None:
        root = next(n for n in general["nodes"] if n["parent"] is None)
        root["settings"]["server_optimizer"] = config.learning.server_optimizer
    general["edge"]["settings"] = {
        "eval": evaluation.edge.model_dump(exclude_none=True)
    } | general["edge"]["settings"]
    if config.runtime.codec:
        general["edge"]["link_up"]["codec"] = config.runtime.codec
    return Topology.model_validate(general)


def _single_source(topology: Topology) -> None:
    """With an infrastructure, network and device profiles come from it only."""
    links = [(n.id, n.link_up) for n in topology.nodes if n.link_up is not None]
    links.append(("edge", topology.edge.link_up))
    given = [node for node, link in links if "profile" in link.model_fields_set]
    if given:
        raise ConfigError(
            f"topology: link_up.profile at {given}; with an infrastructure the "
            "links come from its network, so remove them from the topology"
        )
    if "device" in topology.edge.settings:
        raise ConfigError(
            "topology: edge.device; with an infrastructure the edges compute at "
            "the train_rate of their host's profile, so remove it"
        )


def resolve_infrastructure(config: ExperimentConfig) -> Infrastructure | None:
    if config.infrastructure is None:
        return None
    ref = config.infrastructure.name
    try:
        if isinstance(ref, dict):  # its traces live with the infrastructures
            raw = anchor_traces(deepcopy(ref), config.paths.infrastructures)
            return parse_infrastructure(raw)
        path = Path(ref)
        if path.suffix not in (".yaml", ".yml"):
            path = Path(config.paths.infrastructures) / f"{ref}.yaml"
        return load_infrastructure(path)
    except InfrastructureError as exc:
        raise ConfigError(f"infrastructure: {exc}") from None


def physical(
    scenario: Scenario,
    topology: Topology,
    placement: Placement,
    initial: Mapping[str, Any],
) -> Resolved | None:
    """Hosts and compiled links of a scenario, or None without an infrastructure;
    ``initial`` is the initial model state, for a coupling on the initial loss."""
    config = scenario.config
    infra = resolve_infrastructure(config)
    if infra is None:
        return None
    assignment = config.infrastructure.assignment
    strategy = create(assignments, assignment.edges)
    statistics = None
    if isinstance(strategy, Coupled):
        statistics = edge_statistics(strategy.statistic, scenario, placement, initial)
    try:
        return resolve(
            infra,
            topology,
            root=assignment.root,
            aggregators=assignment.aggregators,
            edges=strategy,
            leaves={
                leaf: [c.id for c in clients]
                for leaf, clients in placement.edges.items()
            },
            datasets={c.id: c.dataset for c in _placed(placement)},
            services=assignment.services,
            statistics=statistics,
        )
    except InfrastructureError as exc:
        raise ConfigError(f"infrastructure: {exc}") from None


def edge_statistics(
    name: str, scenario: Scenario, placement: Placement, initial: Mapping[str, Any]
) -> dict[str, float]:
    """Each training edge's ``samples``, ``label_divergence`` (KL divergence of its
    labels from those of all the training edges of its dataset) or
    ``initial_loss`` (cross-entropy of the initial model on its training data)."""
    clients = _placed(placement)
    if name == "samples":
        return {c.id: float(c.train.n_samples) for c in clients}
    if name == "label_divergence":
        counts = {
            c.id: np.bincount(c.train.y, minlength=c.train.n_classes) for c in clients
        }
        pooled: dict[str, Any] = {}
        for c in clients:
            pooled[c.dataset] = pooled.get(c.dataset, 0) + counts[c.id]
        out = {}
        for c in clients:
            p, q = counts[c.id], pooled[c.dataset]
            p, q = p / max(p.sum(), 1), q / max(q.sum(), 1)
            held = p > 0
            out[c.id] = float(np.sum(p[held] * np.log(p[held] / q[held])))
        return out
    family = create(models, scenario.config.learning.model)
    out = {}
    for c in clients:
        model = family.build([c.train.shape], seed=scenario.seed)
        load_arrays(model, initial)
        y = np.asarray(c.train.y, np.int64)
        logits = predict(model, c.train.X)
        loss = compute(logits, y, c.train.n_classes, ["loss"]) if len(y) else {}
        out[c.id] = loss.get("loss", 0.0)
    return out


def load_data(config: ExperimentConfig) -> tuple[list[Any], dict[str, str]]:
    """Every dataset through the cache: its subjects and the digest of each cache."""
    subjects, digests, loaded = [], {}, {}
    for name, use in sorted(config.data.datasets.items()):
        spec = load_spec(use.descriptor or Path(config.paths.datasets) / f"{name}.yaml")
        if spec.name in loaded:  # their subjects would overwrite each other
            raise ConfigError(
                f"data.datasets: {loaded[spec.name]} and {name} both load dataset "
                f"{spec.name!r}; give one descriptor another name"
            )
        loaded[spec.name] = name
        path = prepare(spec, use.options, cache_dir=config.paths.cache)
        digests[name] = hashlib.sha256((path / "meta.json").read_bytes()).hexdigest()
        subjects += load_prepared(path)
    return subjects, digests


def _newest_run(runs: Path, reference: str, seed: int) -> Path:
    """``experiment:<name>[/<scenario>]``: the newest finished run of that
    experiment (and scenario) with ``seed``, among the runs with a bundle."""
    name, _, scenario = reference.partition("/")
    found: dict[str, list[Path]] = {}
    for path in sorted(runs.iterdir()) if runs.is_dir() else []:
        if not (path / "run.json").is_file() or not (path / "bundle").is_dir():
            continue
        meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
        if (
            meta["config"]["name"] == name
            and meta["seed"] == seed
            and meta.get("status") == "finished"
            and (not scenario or meta.get("scenario") == scenario)
        ):
            found.setdefault(meta.get("scenario"), []).append(path)
    if not found:
        raise ConfigError(
            f"learning.init: no finished run of experiment {reference!r} with seed "
            f"{seed} in {runs}"
        )
    if len(found) > 1:
        raise ConfigError(
            f"learning.init: experiment {name!r} has runs of several scenarios "
            f"{sorted(found)}; name one as experiment:{name}/<scenario>"
        )
    (paths,) = found.values()
    return paths[-1]  # run ids start with their UTC time


def _algorithms(config: ExperimentConfig) -> dict[str, Any]:
    """The trainer, server optimizer and DP mechanisms whose state a bundle holds."""
    topology = resolve_topology(config)
    root = topology.root
    central = {
        node.id: _name(node.settings["aggregator"])
        for node in topology.nodes
        if isinstance(
            create(aggregators, node.settings.get("aggregator"), "fedavg"), Accountant
        )
    }
    return {
        "trainer": _name(config.learning.trainer),
        "server_optimizer": _name(root.settings.get("server_optimizer") or "replace"),
        "root": root.id,
        "privacy": {
            "edges": _name(config.privacy) if config.privacy else None,
            "aggregators": central,
        },
    }


def _nodes_of(topology: Topology) -> set[str]:
    return {node.id for node in topology.nodes}


def _check_privacy(
    saved: Mapping[str, Any], here: Mapping[str, Any], shared: set[str]
) -> None:
    """The edges and every aggregator ``shared`` with the parent keep their DP
    mechanism, so each budget composes; σ, C and δ may change."""
    central = saved.get("aggregators", {})
    for where, before, now in [
        ("the edges", saved.get("edges"), here["edges"]),
        *[(n, central.get(n), here["aggregators"].get(n)) for n in sorted(shared)],
    ]:
        if before != now:
            raise ConfigError(
                f"learning.init: {where} used privacy {before!r} in the parent, here "
                f"{now!r}; a continuation keeps each DP mechanism so its budget "
                "composes (σ, C and δ may change)"
            )


def _parent(config: ExperimentConfig, seed: int) -> tuple[Path, Bundle, Any] | None:
    """The run that ``init: run`` continues: its folder, verified bundle and restore.

    ``run`` is a run id, a run folder, or ``experiment:<name>[/<scenario>]``:
    the newest finished run of that experiment with this scenario's seed. The
    parent must be finished, signed and continuable exactly: same seed, same
    root, and the same trainer and server optimizer for the state it restores.
    """
    if _name(config.learning.init) != "run":
        return None
    if not config.simulated:
        raise ConfigError(
            "learning.init: run continues only in simulation for now; real runs "
            "write no bundle until the distributed continuum (C9)"
        )
    init = create(inits, config.learning.init)
    if init.restore.preprocessing and config.data.roles.scaler == "local":
        raise ConfigError(
            "learning.init: frozen local preprocessing is not supported yet (each "
            "holder would refit its own scaler); use scaler: global, or set "
            "restore.preprocessing: false"
        )
    trainer = create(trainers, config.learning.trainer)
    paired = getattr(trainer, "server_optimizer", None) and trainer._memory
    if paired and init.restore.server_state != init.restore.edge_state:
        raise ConfigError(
            f"learning.init: {_name(config.learning.trainer)} pairs the server's "
            "state with each edge's (c and c_i, h and the edge corrections); "
            "restore server_state and edge_state together"
        )
    runs = Path(config.paths.runs)
    if init.run.startswith("experiment:"):
        path = _newest_run(runs, init.run.split(":", 1)[1], seed)
    else:
        path = Path(init.run)
        if not path.is_dir():
            path = runs / init.run
    if not (path / "run.json").is_file():
        raise ConfigError(f"learning.init: run {init.run!r} not found in {path.parent}")
    if not verify_run(path):
        raise ConfigError(
            f"learning.init: run {path.name!r} does not verify against its run_hash"
        )
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if meta.get("status") != "finished":
        raise ConfigError(
            f"learning.init: run {path.name!r} is {meta.get('status')!r}; only a "
            "finished run can be continued"
        )
    if meta["seed"] != seed:
        raise ConfigError(
            f"learning.init: run {path.name!r} has seed {meta['seed']}, this scenario "
            f"{seed}; continue each seed from its own parent (experiment:<name>)"
        )
    if not (path / "bundle" / "bundle.json").is_file():
        raise ConfigError(f"learning.init: run {path.name!r} has no bundle")
    bundle = load_bundle(path / "bundle")
    saved, here = bundle.lineage.get("algorithms", {}), _algorithms(config)
    if saved.get("root") != here["root"]:
        raise ConfigError(
            f"learning.init: the parent's root is {saved.get('root')!r}, here "
            f"{here['root']!r}; a continuation keeps the coordinator"
        )
    for part, flag in (("trainer", "edge_state"), ("server_optimizer", "server_state")):
        if getattr(init.restore, flag) and saved.get(part) != here[part]:
            raise ConfigError(
                f"learning.init: the parent's {part} is {saved.get(part)!r}, here "
                f"{here[part]!r}; its state does not fit: set restore.{flag}: false"
            )
    shared = _nodes_of(resolve_topology(config)) & set(bundle.snapshot.nodes)
    _check_privacy(saved.get("privacy", {}), here["privacy"], shared)
    return path, bundle, init.restore


def _check_roles(parent: Mapping[str, Any], child: Mapping[str, Any]) -> None:
    """A subject keeps its role along a lineage: once it trained, validated or
    tested, it never takes another role; new subjects may join any. A bundle's
    roles are its whole lineage's, so they stay a partition."""
    if not parent:
        raise ConfigError("learning.init: the parent's bundle records no data roles")
    for dataset, roles in sorted(child.items()):
        before = parent.get(dataset) or {}
        for role, earlier in permutations(ROLES, 2):
            moved = set(roles.get(role, [])) & set(before.get(earlier, []))
            if moved:
                raise ConfigError(
                    f"learning.init: {dataset} subjects "
                    f"{sorted(moved, key=natural_key)} were {earlier} subjects in "
                    f"the lineage but are {role} subjects here; a subject keeps its "
                    "role along a lineage (keep the parent's data.roles; new "
                    "subjects may join)"
                )


def _lineage_roles(
    parent: Mapping[str, Any], child: Mapping[str, Any]
) -> dict[str, dict[str, list[str]]]:
    """Every subject that ever trained, validated or tested, per dataset, along
    the lineage, kept even for a dataset this run does not load."""
    return {
        dataset: {
            role: sorted(
                set((parent.get(dataset) or {}).get(role, []))
                | set((child.get(dataset) or {}).get(role, [])),
                key=natural_key,
            )
            for role in ROLES
        }
        for dataset in sorted(set(parent) | set(child))
    }


def _check_structure(
    bundle: Bundle, config: ExperimentConfig, split: DataSplit, seed: int
) -> None:
    """A restored model must fit: each dataset the parent knew keeps its task and
    classes, and each parameter group both models hold keeps its shapes."""
    for dataset, schema in schema_of(split).items():
        saved = bundle.schema.get(dataset)
        if saved is not None and saved != schema:
            raise ConfigError(
                f"learning.init: {dataset} had task and classes {saved} in the "
                f"parent, here {schema}; set restore.model: false"
            )
    _, state = _initial_state(config, _shapes(split), seed)
    root = bundle.snapshot.nodes[bundle.snapshot.root].arrays
    parent = {k[6:]: v for k, v in root.items() if k.startswith("state/")}

    def shapes(arrays: Mapping[str, Any], group: str) -> dict[str, tuple]:
        return {
            k: np.shape(v)
            for k, v in arrays.items()
            if not is_aux(k) and group_of(k) == group
        }

    mine = set(param_groups(state))
    loaded = set(schema_of(split)) | {s["task"] for s in schema_of(split).values()}
    for group in sorted(param_groups(parent)):
        # a group of a dataset or task this run does not load may be absent
        absent = group not in mine and group.partition(".")[2] not in loaded | {""}
        if not absent and shapes(parent, group) != shapes(state, group):
            raise ConfigError(
                f"learning.init: the parameters of {group} differ from the "
                "parent's: the model's structure changed (adapter_width, "
                "trunk_hidden, adapters, trunk, heads...); set restore.model: false"
            )


def _check_budgets(bundle: Bundle, clients: Sequence[Any]) -> None:
    """Under local DP, a new edge may not hold a subject that already released:
    its budget is the edge's, so it would start again at zero."""
    for client in clients:
        trained = (bundle.roles.get(client.dataset) or {}).get("train", [])
        released = sorted(set(trained) & set(client.subjects), key=natural_key)
        if released and client.id not in bundle.snapshot.nodes:
            raise ConfigError(
                f"learning.init: edge {client.id} is new, but {released} already "
                "released updates under local DP and their budget would restart; "
                "keep subjects_per_client and the training subjects"
            )


def _restore_parts(
    snapshot: FederationSnapshot, restore: Any, edges: set[str]
) -> FederationSnapshot:
    """The part of a parent's snapshot that ``restore`` asks for.

    ``model`` governs every model weight: the global model, zones, previous
    aggregates, edge models and the trainer memory that holds weights.
    ``server_state`` governs the server optimizer; ``edge_state`` the rest of an
    edge, its trainer memory, replay memory, what its stream consumed and its
    random stream. SCAFFOLD's c, auxiliary keys of
    the global state, is server state. DP budgets always carry over, and so does
    the stream of an edge that has one: what was released stays released.
    """

    def kept(key: str) -> bool:
        part, _, rest = key.partition("/")
        if part == "state":  # SCAFFOLD's c rides in the global state, auxiliary
            return restore.server_state if is_aux(rest) else restore.model
        if part in ("zone", "previous", "model"):
            return restore.model
        if part in ("replay", "stream", "unlabelled"):
            return restore.edge_state
        return restore.server_state if part == "server" else True

    nodes = {}
    for node_id, node in snapshot.nodes.items():
        meta = dict(node.meta)
        arrays = {k: v for k, v in node.arrays.items() if kept(k)}
        if "memory" in meta:  # weights follow model, the rest edge_state
            meta["memory"] = {
                name: info
                for name, info in meta["memory"].items()
                if (restore.model if info.get("weights") else restore.edge_state)
            }
            arrays = {
                k: v
                for k, v in arrays.items()
                if not k.startswith("memory/") or k.split("/")[1] in meta["memory"]
            }
        if not restore.edge_state:
            meta.pop("replay", None)
        # A DP edge keeps its stream: a fresh one would replay the parent's noise.
        private = any(k.startswith("privacy/") for k in node.arrays)
        if node_id in edges and not restore.edge_state and not private:
            meta.pop("rng", None)
        nodes[node_id] = NodeState(arrays, meta)
    return FederationSnapshot(snapshot.round, nodes, snapshot.root)


def _restore(federation: Any, bundle: Bundle, restore: Any) -> None:
    """Continue the parent's bundle, leaving out what ``restore`` does not ask for."""
    parts = _restore_parts(bundle.snapshot, restore, set(federation.edges))
    restore_federation(federation, parts)


def _scenario_data(
    scenario: Scenario, parent: Any = ...
) -> tuple[Topology, DataSplit, Placement, dict[str, str]]:
    """Topology, split, placement and data digests; ``parent`` as ``_parent`` gives
    it (resolved here when not passed)."""
    config = scenario.config
    topology = resolve_topology(config)
    subjects, digests = load_data(config)
    if parent is ...:
        parent = _parent(config, scenario.seed)
    frozen = parent[1].preprocessing if parent and parent[2].preprocessing else None
    fit_rows = None
    if config.stream is not None:
        # An edge handles its stream when a round reaches it, so every edge must
        # be in every round, or the rows of one left out until the end are lost.
        partial = sorted(
            node.id
            for node in topology.nodes
            if _name(node.settings.get("participation") or "all") != "all"
        )
        if partial:
            raise ConfigError(
                f"stream: participation other than all at {partial} would leave "
                "the arrivals of the edges not selected unhandled; streams need "
                "every edge in every round (with a continuum, an edge's local "
                "trigger decides whether it trains)"
            )
    if config.stream is not None:  # the preprocessing sees the bootstrap only
        untimed = sorted({s.dataset for s in subjects if s.t is None})
        if untimed:
            raise ConfigError(
                f"stream: {untimed} have no time per row; add a time step to their "
                "dataset descriptors"
            )

        def fit_rows(data: Any) -> np.ndarray:
            return bootstrap_rows(data, config.stream)

    split = split_subjects(
        subjects, config.data.roles, frozen=frozen, fit_rows=fit_rows
    )
    placement_ref = config.data.placement
    name = placement_ref if isinstance(placement_ref, str) else placement_ref["name"]
    params = (
        {}
        if isinstance(placement_ref, str)
        else {k: v for k, v in placement_ref.items() if k != "name"}
    )
    if parent:
        _check_roles(parent[1].roles, split.roles)
        if parent[2].model:
            _check_structure(parent[1], config, split, scenario.seed)
    placement = place(split, topology, name, params, seed=scenario.seed)
    if parent and config.privacy is not None:  # the edges as placed, pooled too
        _check_budgets(parent[1], _placed(placement))
    return topology, split, placement, digests


def _shapes(split: DataSplit) -> list[Any]:
    first = {}
    for data in [c.train for c in split.clients] + split.val + split.test:
        first.setdefault(data.dataset, data.shape)
    return [first[d] for d in sorted(first)]


def _initial_state(config: ExperimentConfig, shapes: Sequence[Any], seed: int):
    family = create(models, config.learning.model)
    model = family.build(shapes, seed=seed)
    create(inits, config.learning.init).init(model)
    state = state_arrays(model)
    _check_learning(config, state)
    return family, state


def _name(ref: Any) -> str:
    return ref if isinstance(ref, str) else ref["name"]


def _check_learning(config: ExperimentConfig, state: Mapping[str, Any]) -> None:
    """The trainer fits the sharing (FedRep) and pairs with the root optimizer (FedNova)."""
    trainer = create(trainers, config.learning.trainer)
    name = _name(config.learning.trainer)
    patterns = list(getattr(trainer, "local_groups", ()))
    policy = create(sharing, config.learning.sharing)
    leaving = not_local(policy, param_groups(state), patterns)
    if leaving:
        raise ConfigError(
            f"learning.trainer: {name} keeps {patterns} on the edge, but sharing "
            f"{policy.name!r} sends {leaving} up; use fedper or a custom rule "
            "that keeps them local"
        )
    features = sorted(
        g
        for g in param_groups(state)
        if not g.startswith("head.") and policy.scope_of(g) == "local"
    )
    if getattr(trainer, "shared_features", False) and features:
        raise ConfigError(
            f"learning.trainer: {name} contrasts with the global model's features, "
            f"but sharing {policy.name!r} keeps {features} on the edge, so the "
            "global ones never arrive; use fedavg, fedper or zone"
        )
    unlabelled = config.unlabelled is not None and _name(config.unlabelled) != "none"
    if unlabelled and not getattr(trainer, "takes_unlabelled", False):
        raise ConfigError(
            f"unlabelled: trainer {name} runs its own training loops, which add "
            "no term for unlabelled rows; use standard, fedprox, fedbabu, "
            "scaffold, fednova, feddyn or moon"
        )
    topology = resolve_topology(config)
    if getattr(trainer, "sends_aux", False):
        bounding = sorted(
            {
                _name(ref)
                for node in topology.nodes
                if (ref := node.settings.get("aggregator")) is not None
                and getattr(create(aggregators, ref), "bounds_updates", False)
            }
        )
        axes = [axis for axis in ("attack", "privacy") if getattr(config, axis)]
        if bounding or axes:
            raise ConfigError(
                f"learning.trainer: {name} sends auxiliary arrays that "
                f"{', '.join(bounding + axes)} would not clip, noise or poison; "
                "use a trainer without them"
            )
    frozen = sorted({group_of(k) for k in frozen_keys(trainer, state)})
    noising = sorted(
        node.id
        for node in topology.nodes
        if (ref := node.settings.get("aggregator")) is not None
        and isinstance(create(aggregators, ref), Accountant)
    )
    if frozen and noising:
        raise ConfigError(
            f"learning.trainer: {name} keeps {frozen} frozen, but {noising} add "
            "noise to every key they aggregate, so those groups would drift; "
            "unfreeze them or use privacy local_dp"
        )
    ref = topology.root.settings.get("server_optimizer") or "replace"
    optimizer_name = _name(ref)
    needed = getattr(trainer, "server_optimizer", None)
    late = sorted(
        node.id
        for node in topology.nodes
        if _name(node.settings.get("staleness") or "drop") != "drop"
    )
    if needed is not None and late:
        raise ConfigError(
            f"learning.trainer: {name} pairs with a server optimizer whose round "
            f"statistics count only fresh updates, but {late} also aggregate late "
            "ones; use staleness drop"
        )
    buffering = sorted(
        n.id for n in topology.nodes if n.settings.get("close_at_quorum")
    )
    if needed is not None and getattr(trainer, "_memory", ()) and buffering:
        raise ConfigError(
            f"learning.trainer: {name} keeps edge state that its server optimizer "
            f"tracks, but {buffering} close rounds at quorum (close_at_quorum), so "
            "the slower edges' updates are dropped while their state moves on"
        )
    zoned = sorted(
        g for g in param_groups(state) if policy.scope_of(g).startswith("level:")
    )
    if needed is not None and zoned:
        raise ConfigError(
            f"learning.trainer: {name} pairs with a server optimizer that only "
            f"runs at the root, but sharing {policy.name!r} keeps {zoned} below "
            "the root, where fogs only average"
        )
    if needed is not None and optimizer_name != needed:
        raise ConfigError(
            f"learning.trainer: {name} needs server_optimizer {needed!r}, "
            f"not {optimizer_name!r}; set learning.server_optimizer"
        )
    check = getattr(create(server_optimizers, ref), "check_trainer", None)
    if check is not None:
        try:
            check(name, trainer)
        except ValueError as exc:
            raise ConfigError(f"learning.server_optimizer: {exc}") from None


def link_warnings(topology: Topology, resolved: Resolved | None = None) -> list[str]:
    """Aggregators fed by lossy links without a deadline: one lost update stalls a round."""
    warnings = []
    leaves = {leaf.id for leaf in topology.leaves()}
    for node in topology.nodes:
        if resolved is not None:  # evaluators sit on their parent's host: no loss
            losses = [
                profile["loss"]
                for child, (_, profile) in resolved.links.items()
                if resolved.parents[child] == node.id
            ]
        else:
            links = [c.link_up for c in topology.children(node.id)]
            if node.id in leaves or node.id == topology.root.id:  # edges, evaluators
                links.append(topology.edge.link_up)
            losses = [resolve_profile(link.profile).loss for link in links]
        loss = max(losses, default=0.0)
        if loss > 0 and node.settings.get("deadline") is None:
            warnings.append(
                f"{node.id}: its children's links lose messages (up to {loss:.1%}) and it has "
                "no deadline, so a lost update or evaluation stalls the run; set a deadline"
            )
        staleness = create(stalenesses, node.settings.get("staleness"), "drop")
        forever = getattr(staleness, "max_staleness", 0) is None
        if loss > 0 and node.settings.get("close_at_quorum") and forever:
            warnings.append(
                f"{node.id}: it closes rounds at quorum and keeps late updates of any "
                "age, so a child whose update is lost gets no model again; set "
                "staleness max_staleness"
            )
    return warnings


def plan(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Dry run of every scenario: composition per leaf, roles and the groups on each link."""
    previews = []
    for scenario in scenarios(config):
        topology, split, placement, digests = _scenario_data(scenario)
        _, state = _initial_state(scenario.config, _shapes(split), scenario.seed)
        resolved = physical(scenario, topology, placement, state)
        if resolved is not None:
            check_memory(resolved, placement, state)
            check_capacity(scenario.config, topology, placement)
        policy = create(sharing, scenario.config.learning.sharing)
        previews.append(
            {
                "scenario": scenario.name,
                "seed": scenario.seed,
                "config_id": scenario.config_id,
                "topology_id": topology.topology_id,
                "graph": topology.to_graph(),
                "roles": split.roles,
                "composition": placement.composition(),
                "traffic": traffic(topology, policy, list(param_groups(state))),
                "data": digests,
                "warnings": link_warnings(topology, resolved),
            }
            | ({} if resolved is None else {"infrastructure": resolved.preview()})
            | (
                {}
                if scenario.config.stream is None
                else {
                    "stream": _stream_summary(
                        scenario.config, _streams(scenario, placement)
                    )
                }
            )
        )
    return previews


def _edge_runtime(topology: Topology) -> dict[str, Any]:
    settings = topology.edge.settings
    out: dict[str, Any] = {}
    device = settings.get("device")
    if isinstance(device, dict) and "samples_per_second" in device:
        out["compute"] = compute_models.create("samples_per_second", device)
    elif device is not None:
        out["compute"] = create(compute_models, device)
    if settings.get("availability") is not None:
        out["availability"] = create(availability_models, settings["availability"])
    return out


def _sinks(config: ExperimentConfig) -> list[Any]:
    out = []
    for sink in config.sinks:
        name = sink if isinstance(sink, str) else sink["name"]
        if name == "prometheus":
            port = sink.get("port", 9464) if isinstance(sink, dict) else 9464
            out.append(PrometheusSink.serving(port))
        elif name == "otel":
            endpoint = sink.get("endpoint") if isinstance(sink, dict) else None
            out.append(OtelSink(otlp_provider(endpoint) if endpoint else None))
    return out


def malicious_edges(
    config: ExperimentConfig, clients: Sequence[Any], seed: int
) -> set[str]:
    """The seeded ``fraction`` of each dataset's training edges that attack."""
    if config.attack is None:
        return set()
    fraction = create(attacks, config.attack).fraction
    chosen: set[str] = set()
    for dataset in sorted({c.dataset for c in clients}):
        ids = sorted(c.id for c in clients if c.dataset == dataset)
        n = int(fraction * len(ids) + 0.5)  # half up: round() is banker's
        if n:
            rng = node_rng(seed, f"attack/{dataset}")
            chosen |= set(rng.choice(ids, size=n, replace=False).tolist())
    return chosen


def _streams(scenario: Scenario, placement: Placement) -> dict[str, Any]:
    """Each training edge's stream, and each validation subject's evaluation
    stream (``val-<dataset>-<subject>``); none without ``stream``."""
    config = scenario.config
    if config.stream is None:
        return {}
    stream, labels = config.stream, config.labels or LabelsConfig()
    gap = stream.start.staggered if isinstance(stream.start, Staggered) else 0.0
    clients = sorted(
        (c for group in placement.edges.values() for c in group),
        key=lambda c: (c.dataset, natural_key(c.id)),
    )
    out = {
        c.id: edge_stream(c.train, stream, labels, scenario.seed, offset=k * gap)
        for k, c in enumerate(clients)
    }
    for group in placement.zone_evaluators.values():
        for data in group:
            out[f"val-{data.dataset}-{data.subject}"] = edge_stream(
                data, stream, labels, scenario.seed
            )
    return out


def _paced(
    config: ExperimentConfig, streams: Mapping[str, Any]
) -> tuple[int, Any, Continuum | None]:
    """Rounds, their virtual spacing, and a continuous federation's pace: a
    stream lasts until its last label; triggers or ``round_every`` pace it."""
    if config.stream is None:
        return config.rounds, None, None
    drain = max((s.drain for s in streams.values()), default=0.0)
    speed = config.stream.speed
    c = config.continuum
    if c is not None:
        trigger = create(triggers, c.trigger)
        local = None if c.edge_trigger is None else create(triggers, c.edge_trigger)
        pace = Continuum(
            trigger=trigger,
            edge_trigger=local,
            status_every=c.status_every / speed,
            speed=speed,
            until=drain / speed,
            drift=drift_detectors(trigger, local),
        )
        return config.rounds, None, pace  # rounds stays an upper bound
    every = config.stream.round_every
    rounds = min(config.rounds, math.ceil(drain / every) + 1)  # one at or after it
    return rounds, every / speed, None


def _rounds_preview(config: ExperimentConfig, learners: Mapping[str, Any]) -> dict:
    """How many rounds a stream run has, or which trigger decides them."""
    rounds, _, pace = _paced(config, learners)
    if pace is None:
        return {"rounds": rounds}
    return {"rounds": None, "trigger": _name(config.continuum.trigger)}


def _stream_summary(config: ExperimentConfig, streams: Mapping[str, Any]) -> dict:
    learners = {k: s for k, s in streams.items() if not k.startswith("val-")}
    return {
        "horizon": max((s.horizon for s in learners.values()), default=0.0),
        "drain": max((s.drain for s in learners.values()), default=0.0),
        **_rounds_preview(config, learners),
        "edges": {
            k: {
                "rows": len(s.history),
                "history": int(s.history.sum()),
                "labelled": int(np.isfinite(s.label_at).sum()),
            }
            for k, s in sorted(learners.items())
        },
    }


def _placed(placement: Placement) -> list[Any]:
    """The training edges as placed, in the split's order: a merging placement
    (pooled) replaces its clients, so attackers and budgets follow these."""
    clients = [c for group in placement.edges.values() for c in group]
    return sorted(clients, key=lambda c: (c.dataset, natural_key(c.id)))


def _replay(scenario: Scenario, edge: str) -> dict[str, Any]:
    """An edge's replay memory, on its own seeded stream, and its share."""
    continual = scenario.config.continual
    if continual is None:
        return {}
    memory = create(memories, continual.memory)
    if isinstance(memory, NoMemory):  # nothing to replay: the edge runs as in C3
        return {}
    memory.rng = node_rng(scenario.seed, f"memory/{edge}")
    return {"replay": memory, "replay_ratio": continual.replay_ratio}


def _update(scenario: Scenario) -> dict[str, Any]:
    """How a stream edge updates; warm, the default, runs as C3 to C5 do."""
    continual = scenario.config.continual
    if continual is None or continual.update == "warm":
        return {}
    return {"update": continual.update}


def _unlabelled(scenario: Scenario) -> dict[str, Any]:
    """An edge's own objective on its unlabelled rows; none runs as C3/C4 do."""
    ref = scenario.config.unlabelled
    objective = None if ref is None else create(objectives, ref)
    return (
        {}
        if objective is None or isinstance(objective, NoObjective)
        else {"unlabelled": objective}
    )


def edge_specs(
    scenario: Scenario, topology: Topology, split: DataSplit, placement: Placement
) -> tuple[dict[str, list[EdgeSpec]], dict[str, Any]]:
    """Edges and zone evaluators under each leaf, test evaluators under the root."""
    config = scenario.config
    family, initial = _initial_state(config, _shapes(split), scenario.seed)
    create(sharing, config.learning.sharing).check_model(family.config)
    device = _edge_runtime(topology)
    streams = _streams(scenario, placement)

    def evaluator(prefix: str, data: Any) -> EdgeSpec:
        # validation subjects stream for evaluation only; test subjects never do
        stream = (
            None
            if prefix == "test"
            else streams.get(f"val-{data.dataset}-{data.subject}")
        )
        return EdgeSpec(
            f"{prefix}-{data.dataset}-{data.subject}",
            family.build([data.shape], seed=scenario.seed),
            data=data,
            train=False,
            tags={"dataset": data.dataset},
            stream=stream,
            validates=prefix == "validate",
        )

    bad = malicious_edges(config, _placed(placement), scenario.seed)
    edges: dict[str, list[EdgeSpec]] = {}
    for leaf, clients in placement.edges.items():
        edges[leaf] = [
            EdgeSpec(
                client.id,
                family.build([client.train.shape], seed=scenario.seed),
                data=client.train,
                trainer=create(trainers, config.learning.trainer),
                val_data=client.local_val,
                tags={"dataset": client.dataset}
                | ({"malicious": True} if client.id in bad else {}),
                attack=(create(attacks, config.attack) if client.id in bad else None),
                privacy=(
                    None
                    if config.privacy is None
                    else create(privacies, config.privacy)
                ),
                stream=streams.get(client.id),
                **_replay(scenario, client.id),
                **_unlabelled(scenario),
                **_update(scenario),
                **device,
            )
            for client in clients
        ] + [evaluator("val", data) for data in placement.zone_evaluators.get(leaf, [])]
    if config.evaluation.global_.subjects == "val":  # selection runs: no test
        edges[topology.root.id] = [
            evaluator("gval", data)
            for leaf in sorted(placement.zone_evaluators)
            for data in placement.zone_evaluators[leaf]
        ]
    else:
        edges[topology.root.id] = [evaluator("test", data) for data in placement.test]
    if config.promotion is not None:  # the validation streams decide (C7)
        validators = [
            evaluator("validate", data)
            for leaf in sorted(placement.zone_evaluators)
            for data in placement.zone_evaluators[leaf]
        ]
        if not validators:
            raise ValueError(
                "promotion: there are no validation subjects to decide with; "
                "set data.roles.val"
            )
        edges[topology.root.id] += validators
    return edges, initial


def _on_hosts(
    edges: Mapping[str, Sequence[EdgeSpec]], resolved: Resolved
) -> tuple[dict[str, list[EdgeSpec]], dict[str, Any]]:
    """Edges that compute at their host's rate, and the compiled link profiles.

    Evaluators are not physical clients: they sit on their parent's host.
    """

    out = {
        leaf: [
            replace(
                s, host=resolved.hosts[s.id], compute=_rate(resolved.train_rate(s.id))
            )
            if s.train
            else replace(s, host=resolved.hosts[leaf])
            for s in specs
        ]
        for leaf, specs in edges.items()
    }
    profiles = {
        spec.id: dict(LOOPBACK)
        for specs in edges.values()
        for spec in specs
        if not spec.train
    } | {child: profile for child, (_, profile) in resolved.links.items()}
    return out, profiles


def _rate(rate: float | None) -> Any:
    return (
        None
        if rate is None
        else compute_models.create("samples_per_second", {"samples_per_second": rate})
    )


def _paths(resolved: Resolved) -> tuple[PhysicalNetwork, dict[str, Any] | None]:
    """The physical network of a run and, when it is shared or changes, a channel
    factory per logical link over it; otherwise the compiled profiles do."""
    network = PhysicalNetwork(resolved.infrastructure)
    if not (network.shared or network.changes):
        return network, None
    return network, {child: _channel(network, h) for child, h in resolved.hops.items()}


def _channel(network: PhysicalNetwork, hops: list) -> Any:
    return lambda direction, rng: PathChannel(
        network, travel(hops, direction), direction, rng
    )


def _on_machines(
    runtime: Any,
    resolved: Resolved,
    topology: Topology,
    network: PhysicalNetwork | None = None,
) -> None:
    """Hosts with their CPUs and their downtime, the nodes on them, the
    aggregators' cost and the network's changes as events."""
    infra = resolved.infrastructure
    links = network.events() if network is not None else []
    for at, name, tags in [*links, *host_events(infra)]:
        runtime.note(at, "network", name, **tags)
    used = set(resolved.hosts.values())
    for host in sorted(used):
        runtime.add_host(host, infra.profile_of(host).cpus)
    for host, windows in downtime(infra).items():
        for start, end in windows if host in used else ():
            runtime.host_offline(host, start, end)
    for node, host in resolved.hosts.items():
        runtime.place(node, host)
    for node in topology.nodes:  # combining costs bytes / aggregate_rate
        rate = infra.profile_of(resolved.hosts[node.id]).aggregate_rate
        if rate is not None:
            runtime.set_compute(node.id, _rate(rate))


def _failover_of(config: ExperimentConfig) -> Any:
    return create(failovers, config.infrastructure.failover, "none")


def _max_children(topology: Topology) -> dict[str, int]:
    """``max_children`` of each leaf aggregator that sets it."""
    out = {}
    for leaf in topology.leaves():
        cap = leaf.settings.get("max_children")
        if cap is None:
            continue
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ConfigError(f"topology: {leaf.id}.max_children must be >= 1")
        out[leaf.id] = cap
    return out


def check_capacity(
    config: ExperimentConfig, topology: Topology, placement: Placement
) -> None:
    """Refuse a leaf with more edges than its ``max_children`` when no failover
    can send the rest to another."""
    caps = _max_children(topology)
    if not isinstance(_failover_of(config), NoFailover):
        return
    for leaf, cap in caps.items():
        n = len(placement.edges.get(leaf, []))
        if n > cap:
            raise ConfigError(
                f"infrastructure: {leaf} has {n} edges but max_children {cap}; "
                "set a failover to send the rest to another aggregator"
            )


def _failover(
    federation: Any,
    resolved: Resolved,
    topology: Topology,
    policy: Any,
    network: PhysicalNetwork | None = None,
) -> None:
    """Move edges between leaf aggregators (I1 §9).

    The edges of a leaf whose host is down for ``after`` seconds move to the
    parent ``policy`` chooses among the leaves whose host is up and that have
    room (``max_children``); at the start, a leaf with more edges than its
    ``max_children`` sends the rest the same way. ``network``, when given,
    carries the new links as it carries the others.
    """
    if isinstance(policy, NoFailover):
        return
    runtime, infra = federation.runtime, resolved.infrastructure
    hosts, leaves = resolved.hosts, sorted(n.id for n in topology.leaves())
    caps, spec = _max_children(topology), topology.edge.link_up
    linked = {
        (child, leaf)
        for leaf in leaves
        for child in federation.aggregators[leaf].children
    }
    shared = network is not None and bool(network.shared or network.changes)

    def edges_of(leaf: str) -> list[str]:
        held = federation.aggregators[leaf].children
        return sorted(c for c in held if c in hosts and federation.edges[c].train)

    def move(edge: str, old: str, down: bool) -> Any:
        def call(node: Any, ctx: Any) -> None:
            if node.parent != old or (down and runtime.host_up(hosts[old])):
                return  # moved already, or its parent is back
            candidates = {
                leaf: hosts[leaf]
                for leaf in leaves
                if leaf != old
                and runtime.host_up(hosts[leaf])
                and len(edges_of(leaf)) < caps.get(leaf, math.inf)
            }
            new = policy.choose(edge, hosts[edge], candidates, infra)
            if new is None:
                ctx.emit("edge.unparented", parent=old)
                return
            federation.aggregators[old].release(edge)
            federation.aggregators[new].adopt(edge)
            if (edge, new) not in linked:
                _, crossed, profile = resolved.link(edge, new, spec.transport)
                runtime.add_link(
                    edge,
                    new,
                    profile=profile,
                    codec=spec.codec,
                    transport=spec.transport,
                    **({"channel": _channel(network, crossed)} if shared else {}),
                )
                linked.add((edge, new))
            node.reparent(new, ctx)

        return call

    for leaf, cap in caps.items():  # before anything starts
        for edge in edges_of(leaf)[cap:]:
            runtime.call_at(0.0, move(edge, leaf, down=False), edge)

    def failed(host: str) -> Any:
        def call() -> None:
            for leaf in leaves:
                if hosts[leaf] == host:
                    for edge in edges_of(leaf):
                        runtime.call_at(runtime.now, move(edge, leaf, True), edge)

        return call

    for host, windows in downtime(infra).items():
        for start, end in windows:
            if start + policy.after < end:
                runtime.call_at(start + policy.after, failed(host))


def check_memory(
    resolved: Resolved, placement: Placement, state: Mapping[str, Any]
) -> None:
    """Refuse a host whose nodes' models and data do not fit in its memory.

    ponytail: every node counts the whole initial model, an upper bound (an
    edge holds only its dataset's adapter); per-key sizes if it ever matters.
    """
    model = sum(np.asarray(v).nbytes for v in state.values())
    data = {
        c.id: sum(d.X.nbytes + d.y.nbytes for d in (c.train, c.local_val) if d)
        for c in _placed(placement)
    }
    infra = resolved.infrastructure
    for host, nodes in resolved.load().items():
        memory = infra.profile_of(host).memory
        need = sum(model + data.get(n, 0) for n in nodes)
        if memory is not None and need > memory:
            raise ConfigError(
                f"infrastructure: host {host} needs {need / 2**20:.1f} MiB for the "
                f"models and data of {nodes}, but has {memory / 2**20:.1f} MiB"
            )


def build_scenario(
    scenario: Scenario,
    topology: Topology,
    split: DataSplit,
    placement: Placement,
    *,
    evaluate: Callable[..., Any] | None = None,
    runtime: Any = None,
    parent: Any = ...,
) -> Any:
    """The federation of a scenario, on ``runtime`` (a new SimRuntime by default)."""
    config = scenario.config
    edges, initial = edge_specs(scenario, topology, split, placement)
    resolved, link_profiles = physical(scenario, topology, placement, initial), None
    network, channels = None, None
    if resolved is not None:
        check_memory(resolved, placement, initial)
        check_capacity(config, topology, placement)
        edges, link_profiles = _on_hosts(edges, resolved)
        network, channels = _paths(resolved)
    if parent is ...:
        parent = _parent(config, scenario.seed)
    start = parent[1].snapshot.round if parent else 0
    learners = {s.id: s.stream for g in edges.values() for s in g if s.train}
    rounds, round_every, pace = _paced(config, {k: v for k, v in learners.items() if v})
    federation = build_federation(
        topology,
        edges,
        initial_state=initial,
        rounds=start + rounds,  # a continuation's numbering goes on
        round_every=round_every,
        continuum=pace,
        promotion=(
            None if config.promotion is None else create(promotions, config.promotion)
        ),
        sharing=config.learning.sharing,
        seed=scenario.seed,
        metrics=config.evaluation.metrics,
        evaluate=evaluate,
        runtime=runtime,
        link_profiles=link_profiles,
        link_channels=channels,
    )
    if resolved is not None:
        _on_machines(federation.runtime, resolved, topology, network)
        _failover(federation, resolved, topology, _failover_of(config), network)
    if parent:
        _restore(federation, parent[1], parent[2])
    return federation


def _versions(coordinator: Any) -> dict[str, Any]:
    """The champion and every version's lineage, with a promotion policy; the
    bundle's model is the champion."""
    if coordinator.promotion is None:
        return {}
    return {
        "champion": coordinator.champion,
        "versions": [v.record() for _, v in sorted(coordinator.versions.items())],
    }


def run_scenario(
    scenario: Scenario, evaluate: Callable[..., Any] | None = None
) -> Path:
    """Run one scenario on its backend and return its ``runs/<run_id>/`` folder."""
    config = scenario.config
    infra = config.infrastructure
    return execute(
        create(backends, config.backend_ref),
        scenario,
        infra,
        None if infra is None else infra.assignment,
        evaluate=evaluate,
    )


def simulate(scenario: Scenario, evaluate: Callable[..., Any] | None = None) -> Path:
    """Run one scenario in the simulator (the ``sim`` backend) and sign it."""
    config = scenario.config
    parent = _parent(config, scenario.seed)  # resolved once for the whole run
    topology, split, placement, digests = _scenario_data(scenario, parent)
    infra = resolve_infrastructure(config)
    run = Run(
        config.paths.runs,
        config=identity(config),
        topology=topology,
        seed=scenario.seed,
        data_ids=digests,
        scenario=scenario.name,
        sinks=_sinks(config),
        infrastructure_id=None if infra is None else infra.infrastructure_id,
    )
    lineage_parent = None
    if parent:
        meta = json.loads((parent[0] / "run.json").read_text(encoding="utf-8"))
        lineage_parent = {
            "run_id": meta["run_id"],
            "run_hash": meta["run_hash"],
            "version": parent[1].snapshot.round,
        }
        run.meta["parent"] = lineage_parent  # signed with the rest at finish
    try:
        record_data(run, scenario, split, placement)
        federation = build_scenario(
            scenario, topology, split, placement, evaluate=evaluate, parent=parent
        )
        run.attach(federation)
        if config.stream is not None:
            streams = {k: e.stream for k, e in federation.edges.items() if e.train}
            summary = _stream_summary(config, streams)
            run.record(
                "data.stream",
                summary["horizon"],
                drain=summary["drain"],
                rounds=summary["rounds"],
                edges=summary["edges"],
            )
        federation.run()
        save_model(run.path, federation.coordinator.state)
        # What the lineage knows of a dataset or node outlives a run without it.
        earlier = parent[1] if parent else None
        snapshot, algorithms = snapshot_federation(federation), _algorithms(config)
        if earlier:  # nodes gone here keep what this run kept of them
            kept = _restore_parts(earlier.snapshot, parent[2], set(federation.edges))
            snapshot.nodes = {**kept.nodes, **snapshot.nodes}
            saved = earlier.lineage.get("algorithms", {}).get("privacy", {})
            central = saved.get("aggregators", {})
            algorithms["privacy"]["aggregators"] = {
                **{n: m for n, m in central.items() if n not in _nodes_of(topology)},
                **algorithms["privacy"]["aggregators"],
            }
        save_bundle(
            run.path / "bundle",
            snapshot,
            preprocessing={
                **(earlier.preprocessing if earlier else {}),
                **split.preprocessing,
            },
            schema={**(earlier.schema if earlier else {}), **schema_of(split)},
            lineage={
                "version": federation.coordinator.round,
                "run_id": run.run_id,
                "parent": lineage_parent,
                "algorithms": algorithms,
                **_versions(federation.coordinator),
            },
            config=identity(config),
            roles=_lineage_roles(earlier.roles if earlier else {}, split.roles),
        )
    except BaseException:
        run.finish(status="failed")
        raise
    # The queue can run dry before the last round (lost messages, no deadlines).
    run.finish(status="finished" if federation.coordinator.finished else "incomplete")
    return run.path


def schema_of(split: DataSplit) -> dict[str, dict[str, Any]]:
    """Per dataset, the task and classes its heads were built for."""
    return {
        c.dataset: {"task": c.train.task, "n_classes": c.train.n_classes}
        for c in sorted(split.clients, key=lambda c: c.dataset)
    }


def record_data(
    run: Run, scenario: Scenario, split: DataSplit, placement: Placement
) -> None:
    run.record("data.roles", None, roles=split.roles)
    for leaf, composition in placement.composition().items():
        run.record("data.composition", composition["samples"], leaf=leaf, **composition)
    bad = malicious_edges(scenario.config, _placed(placement), scenario.seed)
    if bad:
        run.record("data.attack", float(len(bad)), edges=sorted(bad))


def check_scenarios(todo: Sequence[Scenario]) -> None:
    """Refuse a sweep before its first run if any scenario cannot be built.

    Builds each scenario's initial model, which checks that its trainer fits
    its sharing (FedRep); a bad scenario then fails before any run folder.
    """
    for scenario in todo:
        config = scenario.config
        split = split_subjects(load_data(config)[0], config.data.roles)
        _initial_state(config, _shapes(split), scenario.seed)


def run_experiment(
    config: ExperimentConfig,
    workers: int = 1,
    only: str | None = None,
    evaluate: Callable[..., Any] | None = None,
) -> list[Path]:
    """Run every scenario (or the one named ``only``); in parallel processes when ``workers > 1``."""
    every = scenarios(config)
    todo = [s for s in every if only is None or s.name == only]
    if not todo:
        names = sorted({s.name for s in every})
        raise ConfigError(f"no scenario named {only!r}; the scenarios are {names}")
    return run_scenarios(todo, workers=workers, evaluate=evaluate)


def run_scenarios(
    todo: Sequence[Scenario],
    workers: int = 1,
    evaluate: Callable[..., Any] | None = None,
) -> list[Path]:
    """Run the scenarios of one experiment, in order; in parallel when ``workers > 1``."""
    if workers > 1 and evaluate is not None:
        raise ValueError(
            "a custom evaluate cannot be sent to worker processes; use workers=1"
        )
    live = [
        s
        for s in todo[0].config.sinks
        if (s if isinstance(s, str) else s["name"]) == "prometheus"
    ]
    if workers > 1 and live:
        raise ConfigError(
            "sinks: prometheus serves one port per process, and worker processes "
            "would all bind it; use workers=1"
        )
    check_scenarios(todo)  # also fills the caches before any worker reads them
    if workers <= 1:
        return [run_scenario(s, evaluate) for s in todo]
    return _in_parallel(run_scenario, todo, workers)


def _in_parallel(fn: Callable[[Any], Any], items: Sequence[Any], workers: int) -> list:
    """``fn`` over ``items`` in worker processes, in order; the first failure, in
    whatever order they finish, cancels what has not started and is raised."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        for done in as_completed(futures):
            if done.exception() is not None:
                for future in futures:
                    future.cancel()
                raise done.exception()
        return [future.result() for future in futures]
