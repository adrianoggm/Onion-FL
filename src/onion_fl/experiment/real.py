from __future__ import annotations

"""Real runs: one process per group of nodes over MQTT (spec §9.2).

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
    """The evaluators under the root, as edge_specs names them."""
    if config.evaluation.global_.subjects == "val":
        return [
            f"gval-{e.dataset}-{e.subject}"
            for leaf in sorted(placement.zone_evaluators)
            for e in placement.zone_evaluators[leaf]
        ]
    return [f"test-{e.dataset}-{e.subject}" for e in placement.test]


def _root_members(config: Any, topology: Topology, placement: Any) -> list[str]:
    """The edges under the root: its training clients when it is the leaf (two
    levels), then its evaluators; as edge_specs builds them."""
    clients = [c.id for c in placement.edges.get(topology.root.id, [])]
    return clients + _root_evaluators(config, placement)


def _check_links(topology: Topology) -> None:
    """Every link of a real run must cross processes: MQTT."""
    links = [n.link_up for n in topology.nodes if n.link_up is not None]
    links.append(topology.edge.link_up)
    for link in links:
        spec = link.transport
        name = spec if isinstance(spec, str) else spec.get("name")
        if name != "mqtt":
            raise ConfigError(
                f"runtime.mode: real needs mqtt links, but a link uses {name!r}; a "
                "memory bus does not cross processes"
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


def run_real(scenario: Scenario, python: str = sys.executable) -> Path:
    """One scenario on the ``local`` backend: prepare, run and collect it."""
    from onion_fl.experiment.backends import LocalBackend
    from onion_fl.infrastructure.backends import execute

    return execute(LocalBackend(python=python), scenario)


def prepare_local(scenario: Scenario, python: str = sys.executable) -> LocalPlan:
    """Check the links and brokers, create the run and write ``scenario.json``."""
    from onion_fl.experiment.runner import _scenario_data, _sinks, record_data

    config = scenario.config
    topology, split, placement, digests = _scenario_data(scenario)
    _check_links(topology)
    check_brokers(topology)
    run = Run(
        config.paths.runs,
        config=identity(config),
        topology=topology,
        seed=scenario.seed,
        data_ids=digests,
        scenario=scenario.name,
        sinks=_sinks(config),  # fed as the events are merged, once the run ends
        backend="local",
    )
    try:
        record_data(run, scenario, split, placement)
        document = {
            "name": scenario.name,
            "seed": scenario.seed,
            "config_id": scenario.config_id,
            "config": config.dump(),
            "epoch": time.time(),
        }
        scenario_file = run.path / "scenario.json"
        scenario_file.write_text(
            json.dumps(document, indent=2, default=str), encoding="utf-8"
        )
        edge_ids = {
            leaf: [c.id for c in clients] for leaf, clients in placement.edges.items()
        }
        for leaf, evaluators in placement.zone_evaluators.items():
            edge_ids.setdefault(leaf, []).extend(
                f"val-{e.dataset}-{e.subject}" for e in evaluators
            )
        edge_ids[topology.root.id] = _root_members(config, topology, placement)
        groups = list(process_groups(topology, edge_ids))
    except BaseException:  # never leave a run that looks still running
        run.finish(status="failed")
        raise
    return LocalPlan(run, scenario, groups, scenario_file, python)


def launch(plan: LocalPlan) -> None:
    """Start one node process per group and wait for them, up to the timeout."""
    try:
        processes = [
            subprocess.Popen(
                [
                    plan.python,
                    "-m",
                    "onion_fl",
                    "node",
                    "--id",
                    group,
                    "--run",
                    plan.run.run_id,
                    "--config",
                    str(plan.scenario_file),
                ]
            )
            for group in plan.groups
        ]
        limit = time.monotonic() + plan.scenario.config.runtime.timeout + 30
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


def merge(plan: LocalPlan) -> Path:
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
    topology, split, placement, _ = _scenario_data(chosen)
    federation_nodes = _group_members(chosen.config, topology, placement, group)
    runtime = RealRuntime(
        run_id,
        seed=chosen.seed,
        hosted=federation_nodes,
        epoch=epoch,
        heartbeat_s=chosen.config.runtime.heartbeat,
    )
    federation = build_scenario(chosen, topology, split, placement, runtime=runtime)
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
    if group == topology.root.id:
        save_model(
            Path(chosen.config.paths.runs) / run_id, federation.coordinator.state
        )
    return 0 if runtime._all_stopped() else 3


def _group_members(
    config: Any, topology: Topology, placement: Any, group: str
) -> set[str]:
    if group == topology.root.id:
        return {group, *_root_members(config, topology, placement)}
    topology.node(group)  # raises for an unknown id
    members = {group, *(c.id for c in placement.edges.get(group, []))}
    members |= {
        f"val-{e.dataset}-{e.subject}" for e in placement.zone_evaluators.get(group, [])
    }
    return members
