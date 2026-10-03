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
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from onion_fl.core.topology import Topology, load_topology, parse_topology
from onion_fl.data.cache import load_prepared, prepare
from onion_fl.data.ingest import load_spec
from onion_fl.data.placement import Placement, place
from onion_fl.data.roles import DataSplit, split_subjects
from onion_fl.experiment.config import ConfigError, ExperimentConfig
from onion_fl.experiment.sweep import Scenario, identity, scenarios
from onion_fl.learning.model import models, param_groups, state_arrays
from onion_fl.learning.sharing import not_local, sharing, traffic
from onion_fl.learning.trainers import inits, trainers
from onion_fl.observability.run import Run, save_model
from onion_fl.observability.sinks import OtelSink, PrometheusSink, otlp_provider
from onion_fl.roles import EdgeSpec, build_federation
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


def _scenario_data(
    scenario: Scenario,
) -> tuple[Topology, DataSplit, Placement, dict[str, str]]:
    config = scenario.config
    topology = resolve_topology(config)
    subjects, digests = load_data(config)
    split = split_subjects(subjects, config.data.roles)
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
    _check_local_groups(config, state)
    return family, state


def _check_local_groups(config: ExperimentConfig, state: Mapping[str, Any]) -> None:
    """A trainer that keeps groups on the edge (FedRep) needs sharing that keeps them."""
    trainer = create(trainers, config.learning.trainer)
    patterns = list(getattr(trainer, "local_groups", ()))
    policy = create(sharing, config.learning.sharing)
    leaving = not_local(policy, param_groups(state), patterns)
    if leaving:
        ref = config.learning.trainer
        name = ref if isinstance(ref, str) else ref["name"]
        raise ConfigError(
            f"learning.trainer: {name} keeps {patterns} on the edge, but sharing "
            f"{policy.name!r} sends {leaving} up; use fedper or a custom rule "
            "that keeps them local"
        )


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

    edges: dict[str, list[EdgeSpec]] = {}
    for leaf, clients in placement.edges.items():
        edges[leaf] = [
            EdgeSpec(
                client.id,
                family.build([client.train.shape], seed=scenario.seed),
                data=client.train,
                trainer=create(trainers, config.learning.trainer),
                val_data=client.local_val,
                tags={"dataset": client.dataset},
                **device,
            )
            for client in clients
        ] + [evaluator("val", data) for data in placement.zone_evaluators.get(leaf, [])]
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
    return build_federation(
        topology,
        edges,
        initial_state=initial,
        rounds=config.rounds,
        sharing=config.learning.sharing,
        seed=scenario.seed,
        metrics=config.evaluation.metrics,
        evaluate=evaluate,
        runtime=runtime,
    )


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
    try:
        record_data(run, split, placement)
        federation = build_scenario(
            scenario, topology, split, placement, evaluate=evaluate
        )
        run.attach(federation)
        federation.run()
        save_model(run.path, federation.coordinator.state)
    except BaseException:
        run.finish(status="failed")
        raise
    # The queue can run dry before the last round (lost messages, no deadlines).
    run.finish(status="finished" if federation.coordinator.finished else "incomplete")
    return run.path


def record_data(run: Run, split: DataSplit, placement: Placement) -> None:
    run.record("data.roles", None, roles=split.roles)
    for leaf, composition in placement.composition().items():
        run.record("data.composition", composition["samples"], leaf=leaf, **composition)


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
    todo = [s for s in scenarios(config) if only is None or s.name == only]
    if workers > 1 and evaluate is not None:
        raise ValueError(
            "a custom evaluate cannot be sent to worker processes; use workers=1"
        )
    check_scenarios(todo)  # also fills the caches before any worker reads them
    if workers <= 1:
        return [run_scenario(s, evaluate) for s in todo]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_scenario, todo))
