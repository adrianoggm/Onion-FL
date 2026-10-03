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
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from onion_fl.continuum.bundle import Bundle, load_bundle, save_bundle
from onion_fl.core.context import node_rng
from onion_fl.core.topology import Topology, load_topology, parse_topology
from onion_fl.data.cache import load_prepared, prepare
from onion_fl.data.ingest import load_spec
from onion_fl.data.placement import Placement, place
from onion_fl.data.roles import DataSplit, split_subjects
from onion_fl.experiment.config import ConfigError, ExperimentConfig
from onion_fl.experiment.sweep import Scenario, identity, scenarios
from onion_fl.learning.aggregators import aggregators, server_optimizers
from onion_fl.learning.attacks import attacks
from onion_fl.learning.model import models, param_groups, state_arrays
from onion_fl.learning.privacy import privacies
from onion_fl.learning.sharing import not_local, sharing, traffic
from onion_fl.learning.trainers import inits, trainers
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
from onion_fl.roles.policies import create
from onion_fl.runtime.devices import availability_models, compute_models
from onion_fl.runtime.network import resolve_profile


def resolve_topology(config: ExperimentConfig) -> Topology:
    ref = config.topology
    if isinstance(ref, dict):
        topology = parse_topology(ref)
    else:
        path = Path(ref)
        if path.suffix not in (".yaml", ".yml"):
            path = Path(config.paths.topologies) / f"{ref}.yaml"
        topology = load_topology(path)
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


def load_data(config: ExperimentConfig) -> tuple[list[Any], dict[str, str]]:
    """Every dataset through the cache: its subjects and the digest of each cache."""
    subjects, digests = [], {}
    for name, use in sorted(config.data.datasets.items()):
        spec = load_spec(use.descriptor or Path(config.paths.datasets) / f"{name}.yaml")
        path = prepare(spec, use.options, cache_dir=config.paths.cache)
        digests[name] = hashlib.sha256((path / "meta.json").read_bytes()).hexdigest()
        subjects += load_prepared(path)
    return subjects, digests


def _newest_run(runs: Path, experiment: str, seed: int) -> Path | None:
    """The newest finished run of ``experiment`` with ``seed`` that has a bundle."""
    found = []
    for path in sorted(runs.iterdir()) if runs.is_dir() else []:
        if not (path / "run.json").is_file() or not (path / "bundle").is_dir():
            continue
        meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
        if (
            meta["config"]["name"] == experiment
            and meta["seed"] == seed
            and meta.get("status") == "finished"
        ):
            found.append(path)
    return found[-1] if found else None  # run ids start with their UTC time


def _parent(config: ExperimentConfig, seed: int) -> tuple[Path, Bundle, Any] | None:
    """The run that ``init: run`` continues: its folder, verified bundle and restore.

    ``run`` is a run id, a run folder, or ``experiment:<name>``: the newest
    finished run of that experiment with this scenario's seed.
    """
    if _name(config.learning.init) != "run":
        return None
    init = create(inits, config.learning.init)
    runs = Path(config.paths.runs)
    if init.run.startswith("experiment:"):
        name = init.run.split(":", 1)[1]
        newest = _newest_run(runs, name, seed)
        if newest is None:
            raise ConfigError(
                f"learning.init: no finished run of experiment {name!r} with seed "
                f"{seed} in {runs}"
            )
        path = newest
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
    if not (path / "bundle" / "bundle.json").is_file():
        raise ConfigError(f"learning.init: run {path.name!r} has no bundle")
    return path, load_bundle(path / "bundle"), init.restore


def _restore(federation: Any, bundle: Bundle, restore: Any) -> None:
    """Continue the parent's bundle, leaving out what ``restore`` does not ask for."""
    snapshot = bundle.snapshot
    nodes = {}
    for node_id, node in snapshot.nodes.items():
        arrays = dict(node.arrays)
        if node_id == snapshot.root:
            dropped = [] if restore.model else ["state/"]
            dropped += [] if restore.server_state else ["server/"]
            arrays = {
                k: v for k, v in arrays.items() if not k.startswith(tuple(dropped))
            }
        elif node_id in federation.edges and not restore.edge_state:
            continue
        nodes[node_id] = NodeState(arrays, node.meta)
    restore_federation(
        federation, FederationSnapshot(snapshot.round, nodes, snapshot.root)
    )


def _scenario_data(
    scenario: Scenario,
) -> tuple[Topology, DataSplit, Placement, dict[str, str]]:
    config = scenario.config
    topology = resolve_topology(config)
    subjects, digests = load_data(config)
    parent = _parent(config, scenario.seed)
    frozen = parent[1].preprocessing if parent and parent[2].preprocessing else None
    split = split_subjects(subjects, config.data.roles, frozen=frozen)
    placement_ref = config.data.placement
    name = placement_ref if isinstance(placement_ref, str) else placement_ref["name"]
    params = (
        {}
        if isinstance(placement_ref, str)
        else {k: v for k, v in placement_ref.items() if k != "name"}
    )
    return (
        topology,
        split,
        place(split, topology, name, params, seed=scenario.seed),
        digests,
    )


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


def link_warnings(topology: Topology) -> list[str]:
    """Aggregators fed by lossy links without a deadline: one lost update stalls a round."""
    warnings = []
    leaves = {leaf.id for leaf in topology.leaves()}
    for node in topology.nodes:
        links = [c.link_up for c in topology.children(node.id)]
        if node.id in leaves:
            links.append(topology.edge.link_up)
        loss = max((resolve_profile(link.profile).loss for link in links), default=0.0)
        if loss > 0 and node.settings.get("deadline") is None:
            warnings.append(
                f"{node.id}: its children's links lose messages (up to {loss:.1%}) and it has "
                "no deadline, so a lost update stalls the round; set a deadline"
            )
    return warnings


def plan(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Dry run of every scenario: composition per leaf, roles and the groups on each link."""
    previews = []
    for scenario in scenarios(config):
        topology, split, placement, digests = _scenario_data(scenario)
        _, state = _initial_state(scenario.config, _shapes(split), scenario.seed)
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
                "warnings": link_warnings(topology),
            }
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
            prometheus = PrometheusSink()
            prometheus.serve(sink.get("port", 9464) if isinstance(sink, dict) else 9464)
            out.append(prometheus)
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


def edge_specs(
    scenario: Scenario, topology: Topology, split: DataSplit, placement: Placement
) -> tuple[dict[str, list[EdgeSpec]], dict[str, Any]]:
    """Edges and zone evaluators under each leaf, test evaluators under the root."""
    config = scenario.config
    family, initial = _initial_state(config, _shapes(split), scenario.seed)
    create(sharing, config.learning.sharing).check_model(family.config)
    device = _edge_runtime(topology)

    def evaluator(prefix: str, data: Any) -> EdgeSpec:
        return EdgeSpec(
            f"{prefix}-{data.dataset}-{data.subject}",
            family.build([data.shape], seed=scenario.seed),
            data=data,
            train=False,
            tags={"dataset": data.dataset},
        )

    bad = malicious_edges(config, split.clients, scenario.seed)
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
    return edges, initial


def build_scenario(
    scenario: Scenario,
    topology: Topology,
    split: DataSplit,
    placement: Placement,
    *,
    evaluate: Callable[..., Any] | None = None,
    runtime: Any = None,
) -> Any:
    """The federation of a scenario, on ``runtime`` (a new SimRuntime by default)."""
    config = scenario.config
    edges, initial = edge_specs(scenario, topology, split, placement)
    parent = _parent(config, scenario.seed)
    start = parent[1].snapshot.round if parent else 0
    federation = build_federation(
        topology,
        edges,
        initial_state=initial,
        rounds=start + config.rounds,  # a continuation's numbering goes on
        sharing=config.learning.sharing,
        seed=scenario.seed,
        metrics=config.evaluation.metrics,
        evaluate=evaluate,
        runtime=runtime,
    )
    if parent:
        _restore(federation, parent[1], parent[2])
    return federation


def run_scenario(
    scenario: Scenario, evaluate: Callable[..., Any] | None = None
) -> Path:
    """Run one scenario and return its ``runs/<run_id>/`` folder."""
    config = scenario.config
    if config.runtime.mode == "real":
        if evaluate is not None:
            raise ValueError("a custom evaluate cannot be sent to the node processes")
        from onion_fl.experiment.real import run_real

        return run_real(scenario)
    topology, split, placement, digests = _scenario_data(scenario)
    run = Run(
        config.paths.runs,
        config=identity(config),
        topology=topology,
        seed=scenario.seed,
        data_ids=digests,
        scenario=scenario.name,
        sinks=_sinks(config),
    )
    parent = _parent(config, scenario.seed)
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
            scenario, topology, split, placement, evaluate=evaluate
        )
        run.attach(federation)
        federation.run()
        save_model(run.path, federation.coordinator.state)
        save_bundle(
            run.path / "bundle",
            snapshot_federation(federation),
            preprocessing=split.preprocessing,
            schema=schema_of(split),
            lineage={
                "version": federation.coordinator.round,
                "run_id": run.run_id,
                "parent": lineage_parent,
            },
            config=identity(config),
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
    bad = malicious_edges(scenario.config, split.clients, scenario.seed)
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
    if workers > 1 and evaluate is not None:
        raise ValueError(
            "a custom evaluate cannot be sent to worker processes; use workers=1"
        )
    check_scenarios(todo)  # also fills the caches before any worker reads them
    if workers <= 1:
        return [run_scenario(s, evaluate) for s in todo]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_scenario, todo))
