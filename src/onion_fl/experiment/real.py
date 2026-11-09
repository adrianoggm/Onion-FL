from __future__ import annotations

"""Real runs: one process per group of nodes over MQTT or gRPC (spec §9.2).

The ``local`` backend (``backend: local``, or its alias ``runtime.mode:
real``) in three steps. ``prepare_local`` creates ``runs/<run_id>/`` and
writes ``scenario.json``; ``launch`` starts one ``onion_fl node`` process per
group: the root with the test evaluators, and each aggregator with the edges
and evaluators under it. Every process rebuilds the same scenario (same data,
roles, placement and seed), hosts only its group and writes
``events.<group>.jsonl``. When the coordinator finishes, a stop travels down
the tree and the processes exit; ``merge`` merges their events into
``events.jsonl`` and signs the run as usual.
"""

import json
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from onion_fl.core.topology import Topology
from onion_fl.experiment.config import ConfigError, parse_experiment
from onion_fl.experiment.sweep import Scenario, identity
from onion_fl.observability.events import JsonlSink, read_events
from onion_fl.observability.run import Run, enrich, node_roles, save_model
from onion_fl.runtime.real import RealRuntime

PART = "events.{group}.jsonl"


def process_groups(
    topology: Topology, edges: Mapping[str, Sequence[str]]
) -> dict[str, list[str]]:
    """``group -> node ids``: every coordinator or aggregator with the edges hanging from it."""
    return {node.id: [node.id, *edges.get(node.id, [])] for node in topology.nodes}


def _brokers(topology: Topology) -> set[str]:
    links = [n.link_up for n in topology.nodes if n.link_up is not None] + [
        topology.edge.link_up
    ]
    out = set()
    for link in links:
        spec = link.transport
        name = spec if isinstance(spec, str) else spec.get("name")
        if name == "mqtt":
            out.add(
                "localhost:1883"
                if isinstance(spec, str)
                else spec.get("broker", "localhost:1883")
            )
    return out


def check_brokers(topology: Topology, timeout: float = 2.0) -> None:
    """Fail fast, before starting any process, when an MQTT broker is not reachable."""
    for broker in sorted(_brokers(topology)):
        host, _, port = broker.rpartition(":")
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                pass
        except OSError as exc:
            raise ConnectionError(
                f"MQTT broker {broker} is not reachable: {exc}"
            ) from None


def _root_evaluators(config: Any, placement: Any) -> list[str]:
    """The evaluators under the root, as edge_specs names them: the test (or
    global validation) evaluators, and the validators of a promotion (C7)."""
    zone = [
        e
        for leaf in sorted(placement.zone_evaluators)
        for e in placement.zone_evaluators[leaf]
    ]
    if config.evaluation.global_.subjects == "val":
        out = [f"gval-{e.dataset}-{e.subject}" for e in zone]
    else:
        out = [f"test-{e.dataset}-{e.subject}" for e in placement.test]
    if config.promotion is not None:
        out += [f"validate-{e.dataset}-{e.subject}" for e in zone]
    return out


CROSS_PROCESS = ("mqtt", "grpc")


def _check_links(topology: Topology) -> None:
    """Every link of a real run must cross processes: MQTT or gRPC."""
    links = [n.link_up for n in topology.nodes if n.link_up is not None]
    links.append(topology.edge.link_up)
    for link in links:
        spec = link.transport
        name = spec if isinstance(spec, str) else spec.get("name")
        if name not in CROSS_PROCESS:
            raise ConfigError(
                f"runtime.mode: real needs mqtt or grpc links, but a link uses "
                f"{name!r}; a memory bus does not cross processes"
            )


@dataclass
class LocalPlan:
    """What the ``local`` backend starts: one ``onion_fl node`` per group."""

    run: Run
    scenario: Scenario
    groups: list[str]
    scenario_file: Path
    python: str
    codes: list[int] = field(default_factory=list)
    grace: float = 30.0  # seconds a process may take to start, on top of the timeout

    def command(self, group: str) -> list[str]:
        """The process that hosts ``group``."""
        return [
            self.python,
            "-m",
            "onion_fl",
            "node",
            "--id",
            group,
            "--run",
            self.run.run_id,
            "--config",
            str(self.scenario_file),
        ]


def run_real(scenario: Scenario, python: str = sys.executable) -> Path:
    """One scenario on the ``local`` backend: prepare, run and collect it."""
    from onion_fl.experiment.backends import LocalBackend
    from onion_fl.infrastructure.backends import execute

    return execute(LocalBackend(python=python), scenario)


def prepare_local(scenario: Scenario, python: str = sys.executable) -> LocalPlan:
    """Check the links and brokers, create the run and write ``scenario.json``."""
    from onion_fl.experiment.runner import _scenario_data

    config = scenario.config
    topology, split, placement, digests = _scenario_data(scenario)
    _check_links(topology)
    check_brokers(topology)
    run = open_run(scenario, topology, split, placement, digests, backend="local")
    try:
        scenario_file = write_scenario(run, scenario, config.dump())
        edges = edge_groups(config, topology, placement)
        groups = list(process_groups(topology, edges))
    except BaseException:  # never leave a run that looks still running
        run.finish(status="failed")
        raise
    return LocalPlan(run, scenario, groups, scenario_file, python)


def open_run(
    scenario: Scenario,
    topology: Topology,
    split: Any,
    placement: Any,
    digests: Mapping[str, str],
    *,
    backend: str,
    infrastructure_id: str | None = None,
) -> Run:
    """Create the run of a scenario that node processes host, with its data."""
    from onion_fl.experiment.runner import _sinks, record_data

    config = scenario.config
    run = Run(
        config.paths.runs,
        config=identity(config),
        topology=topology,
        seed=scenario.seed,
        data_ids=digests,
        scenario=scenario.name,
        sinks=_sinks(config),  # fed as the events are merged, once the run ends
        backend=backend,
        infrastructure_id=infrastructure_id,
    )
    try:
        record_data(run, scenario, split, placement)
    except BaseException:  # never leave a run that looks still running
        run.finish(status="failed")
        raise
    return run


def write_scenario(
    run: Run, scenario: Scenario, config: Mapping[str, Any], **launcher: Any
) -> Path:
    """``scenario.json``: what every node process rebuilds. ``launcher`` adds
    what the processes take from the launcher (``groups``, ``broker``,
    ``grpc``, ``initial``)."""
    document = {
        "name": scenario.name,
        "seed": scenario.seed,
        "config_id": scenario.config_id,
        "config": config,
        "epoch": time.time(),
        **launcher,
    }
    path = run.path / "scenario.json"
    path.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    return path


def edge_groups(
    config: Any, topology: Topology, placement: Any
) -> dict[str, list[str]]:
    """The edges and evaluators under each leaf, and the root's evaluators."""
    out = {leaf: [c.id for c in clients] for leaf, clients in placement.edges.items()}
    for leaf, evaluators in placement.zone_evaluators.items():
        out.setdefault(leaf, []).extend(
            f"val-{e.dataset}-{e.subject}" for e in evaluators
        )
    out[topology.root.id] = _root_evaluators(config, placement)
    return out


def launch(plan: Any) -> None:
    """Start one node process per group and wait for them, up to the timeout.

    ``plan`` is a ``LocalPlan``, or any plan with its ``run``, ``scenario``,
    ``groups``, ``codes``, ``grace`` and ``command(group)``.
    """
    try:
        processes = [subprocess.Popen(plan.command(group)) for group in plan.groups]
        limit = time.monotonic() + plan.scenario.config.runtime.timeout + plan.grace
        for process in processes:
            try:
                wait = max(0.0, limit - time.monotonic())
                plan.codes.append(process.wait(timeout=wait))
            except subprocess.TimeoutExpired:
                process.kill()
                plan.codes.append(process.wait())
    except BaseException:  # never leave a run that looks still running
        plan.run.finish(status="failed")
        raise


def merge(plan: Any) -> Path:
    """Merge the events of every process into the run and sign it."""
    run, codes = plan.run, plan.codes
    events = []
    for part in sorted(run.path.glob(PART.format(group="*"))):
        events += read_events(part)
        part.unlink()
    # The node processes knew each node's level and role; the traffic summary
    # the run adds when it finishes needs them too.
    run.nodes |= {e["node"]: (e["level"], e["role"]) for e in events if e["role"]}
    for event in sorted(events, key=lambda e: (e["t_wall"], e["t_virtual"])):
        run.adopt(event)
    finished = any(e["name"] == "run.finished" for e in events)
    status = "finished" if finished else ("failed" if any(codes) else "incomplete")
    run.meta["processes"] = {"groups": len(plan.groups), "exit_codes": codes}
    run.finish(status=status)
    return run.path


def load_scenario(
    config: str | Path, scenario: str | None = None, seed: int | None = None
) -> tuple[Scenario, float]:
    """The scenario of a ``scenario.json`` written by ``run_real``, or one of an experiment YAML."""
    from onion_fl.experiment.config import load_experiment
    from onion_fl.experiment.sweep import scenarios

    path = Path(config)
    if path.suffix == ".json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        return Scenario(
            doc["name"], doc["seed"], parse_experiment(doc["config"]), doc["config_id"]
        ), doc["epoch"]
    options = [
        s
        for s in scenarios(load_experiment(path))
        if scenario in (None, s.name) and seed in (None, s.seed)
    ]
    if not options:
        raise ValueError(f"{path}: no scenario {scenario!r} with seed {seed!r}")
    return options[0], time.time()


def run_node(
    group: str,
    run_id: str,
    config: str | Path,
    scenario: str | None = None,
    seed: int | None = None,
) -> int:
    """Host one group of nodes of a real run until the tree says stop. Exit code 0 when it did."""
    from onion_fl.experiment.runner import _scenario_data, build_scenario

    chosen, epoch = load_scenario(config, scenario, seed)
    launcher = _launcher(config)
    topology, split, placement, _ = _scenario_data(chosen)
    if "groups" in launcher:  # the launcher chose them (one per emulated host)
        if group not in launcher["groups"]:
            raise ValueError(
                f"no group {group!r}; there are {sorted(launcher['groups'])}"
            )
        federation_nodes = set(launcher["groups"][group])
    else:
        federation_nodes = _group_members(chosen.config, topology, placement, group)
    grpc = launcher.get("grpc")  # across machines: where each node listens
    if grpc is not None:
        grpc = {"listen": grpc["listen"][group], "peers": grpc["peers"]}
    runtime = RealRuntime(
        run_id,
        seed=chosen.seed,
        hosted=federation_nodes,
        epoch=epoch,
        heartbeat_s=chosen.config.runtime.heartbeat,
        broker=launcher.get("broker"),
        grpc=grpc,
    )
    federation = build_scenario(chosen, topology, split, placement, runtime=runtime)
    if "initial" in launcher:
        take_initial(federation, Path(config).parent / launcher["initial"])
    roles = node_roles(federation, topology)
    (Path(chosen.config.paths.runs) / run_id).mkdir(parents=True, exist_ok=True)
    part = JsonlSink(
        Path(chosen.config.paths.runs) / run_id / PART.format(group=group), flush=True
    )

    def write(raw: Mapping[str, Any]) -> None:
        part.write(
            enrich(
                raw,
                run_id=run_id,
                topology_id=topology.topology_id,
                scenario=chosen.name,
                seed=chosen.seed,
                nodes=roles,
            )
        )

    runtime.listeners.append(write)
    try:
        runtime.run(timeout=chosen.config.runtime.timeout)
    finally:
        part.close()
    if topology.root.id in federation_nodes:
        save_model(
            Path(chosen.config.paths.runs) / run_id, federation.coordinator.state
        )
    return 0 if runtime._all_stopped() else 3


def _launcher(config: str | Path) -> dict[str, Any]:
    """What a ``scenario.json`` carries from its launcher; nothing for a YAML."""
    path = Path(config)
    if path.suffix != ".json":
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {k: doc[k] for k in ("groups", "broker", "grpc", "initial") if k in doc}


def take_initial(federation: Any, path: str | Path) -> None:
    """Start the global model from the launcher's initial state.

    A process on another platform (a Linux container, an ARM board) builds the
    same model from the same seed with other bits: torch draws the initial
    weights differently. The root bootstraps every child with its state, so
    taking the launcher's keeps the run bit for bit the simulation's (I1 §10).

    ponytail: keys that never leave an edge (fedper) keep the weights the edge
    draws on its platform; ship the edges' initial models too if that matters.
    """
    with np.load(path, allow_pickle=False) as saved:
        federation.coordinator.state = {key: saved[key] for key in saved.files}


def _group_members(
    config: Any, topology: Topology, placement: Any, group: str
) -> set[str]:
    if group == topology.root.id:
        return {group, *_root_evaluators(config, placement)}
    topology.node(group)  # raises for an unknown id
    members = {group, *(c.id for c in placement.edges.get(group, []))}
    members |= {
        f"val-{e.dataset}-{e.subject}" for e in placement.zone_evaluators.get(group, [])
    }
    return members
