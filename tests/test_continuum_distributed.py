"""The distributed continuum (continuum C9, issue #164).

An edge of a real run saves its part of the bundle on its machine after each
event and comes back from it when its process dies; the launcher kills and
restarts hosts, and starts late ones, on the run's clock. The edges use the
stub trainer on hand-written format rows (``test_roles_stream``): what is
checked is the state an edge keeps, never how well it learns (docs/RULES.md).
The run on containers acting as machines is in ``test_runtime_equivalence``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from onion_fl.continuum.bundle import load_node, save_node
from onion_fl.continuum.memory import memories
from onion_fl.experiment.backends import DistributedBackend
from onion_fl.experiment.config import ConfigError
from onion_fl.experiment.distributed import launch_with_faults
from onion_fl.learning.model import ModularMLP, state_arrays
from onion_fl.roles import EdgeSpec, build_federation
from onion_fl.roles.snapshot import checkpoint_edge, resume_edge
from tests.test_experiment_stream import descriptor, experiment, workspace  # noqa: F401
from tests.test_roles_stream import (
    CONFIG,
    EVERY,
    INITIAL,
    SPEED,
    A,
    Recording,
    stream_of,
    tree,
)
from tests.test_runtime_equivalence import BROKER, _broker_up

ROUNDS = 8


def federation():
    replay = memories.create("reservoir", {"capacity": 6})
    replay.rng = np.random.default_rng(7)
    spec = EdgeSpec(
        "a1",
        ModularMLP(CONFIG, [A], seed=0),
        trainer=Recording(),
        stream=stream_of("a-1", fraction=0.5, delay=60),
        replay=replay,
        replay_ratio=0.5,
    )
    other = EdgeSpec(
        "a2",
        ModularMLP(CONFIG, [A], seed=0),
        trainer=Recording(),
        stream=stream_of("a-2"),
    )
    return build_federation(
        tree(1),
        {"fog_0": [spec, other]},
        initial_state=INITIAL,
        rounds=ROUNDS,
        round_every=EVERY / SPEED,
    )


def trace(fed) -> tuple:
    edge = fed.edges["a1"]
    trained = [
        (e["t"], e["value"])
        for e in fed.runtime.events
        if e["name"] == "edge.trained" and e["node"] == "a1"
    ]
    return state_arrays(edge.model), edge._consumed_by.copy(), trained


def test_an_edge_restarted_from_its_saved_part_goes_on_as_if_it_never_stopped(
    tmp_path: Path,
) -> None:
    whole = federation()
    whole.run()
    cut = federation()
    cut.run(until=3.5 * EVERY / SPEED)  # in the middle of the run
    before = cut.edges["a1"]
    assert (before._consumed_by >= 0).any() and len(before.replay.rows())

    save_node(tmp_path, "a1", checkpoint_edge(before, cut.runtime._rng("a1")))
    # A new process builds the same edge from the same specs, with nothing done.
    fresh = federation().edges["a1"]
    resume_edge(fresh, load_node(tmp_path, "a1"), cut.runtime._rng("a1"))
    fresh.acknowledged = before.acknowledged
    cut.runtime._nodes["a1"] = cut.edges["a1"] = fresh
    cut.run()

    model, consumed, trained = trace(cut)
    want_model, want_consumed, want_trained = trace(whole)
    assert trained == want_trained
    np.testing.assert_array_equal(consumed, want_consumed)
    np.testing.assert_array_equal(fresh.replay.rows(), whole.edges["a1"].replay.rows())
    np.testing.assert_array_equal(fresh._logits, whole.edges["a1"]._logits)
    for key, value in want_model.items():
        np.testing.assert_array_equal(model[key], value, err_msg=key)


def test_an_edge_restarted_without_its_state_trains_its_rows_again() -> None:
    """The check above is meaningful: a fresh edge forgets what it consumed."""
    whole = federation()
    whole.run()
    cut = federation()
    cut.run(until=3.5 * EVERY / SPEED)
    fresh = federation().edges["a1"]
    fresh.acknowledged = True
    cut.runtime._nodes["a1"] = cut.edges["a1"] = fresh
    cut.run()

    def examples(fed) -> int:
        return sum(
            e["tags"]["examples"]
            for e in fed.runtime.events
            if e["name"] == "edge.trained" and e["node"] == "a1"
        )

    assert examples(cut) > examples(whole)


def test_a_saved_part_is_one_file_replaced_whole(tmp_path: Path) -> None:
    fed = federation()
    fed.run(until=2 * EVERY / SPEED)
    state = checkpoint_edge(fed.edges["a1"], fed.runtime._rng("a1"))

    save_node(tmp_path, "a1", state)
    save_node(tmp_path, "a1", state)

    names = sorted(p.name for p in (tmp_path / "nodes").iterdir())
    assert names == ["a1.json", "a1.npz"]  # no temporary left behind
    loaded = load_node(tmp_path, "a1")
    assert loaded.meta == json.loads(json.dumps(state.meta))
    assert loaded.meta["live"]["clock"] == state.meta["live"]["clock"]
    assert set(loaded.arrays) == set(state.arrays)
    assert load_node(tmp_path, "nobody") is None


# --- the launcher's failures ---------------------------------------------------------


def test_restarts_need_checkpoints_and_a_host_cannot_restart_and_join() -> None:
    with pytest.raises(ValueError, match="checkpoints: true"):
        DistributedBackend(inventory="x", restarts=[{"host": "h", "at": 5, "down": 1}])
    with pytest.raises(ValueError, match="pick one"):
        DistributedBackend(
            inventory="x",
            checkpoints=True,
            restarts=[{"host": "h", "at": 5, "down": 1}],
            joins={"h": 9},
        )


class FakeProcess:
    def __init__(self, line: list[str], log: list) -> None:
        self.line, self.log = line, log
        self.code: int | None = None

    def wait(self, timeout=None) -> int:
        return 0 if self.code is None else self.code

    def kill(self) -> None:
        self.code = -9


class FakePlan:
    """The parts of a ``DistributedPlan`` the fault launcher uses."""

    def __init__(self, **params) -> None:
        from types import SimpleNamespace

        self.params = DistributedBackend(inventory="x", **params).params
        self.groups = ["cloud-1", "field-1", "field-2"]
        self.codes: list[int] = []
        self.grace = 1.0
        self.run = SimpleNamespace(meta={"distributed": {}}, finish=lambda **kw: None)
        self.scenario = SimpleNamespace(
            config=SimpleNamespace(runtime=SimpleNamespace(timeout=60.0))
        )

    def command(self, host: str, resume: bool = False) -> list[str]:
        return [host, *(["--resume"] if resume else [])]

    def stop_command(self, host: str, signal: str = "TERM") -> list[str]:
        return ["kill", signal, host]


def test_the_launcher_kills_restarts_and_starts_late_on_the_runs_clock() -> None:
    plan = FakePlan(
        checkpoints=True,
        restarts=[{"host": "field-1", "at": 10, "down": 5}],
        joins={"field-2": 12},
    )
    now = [100.0]
    log: list = []

    def popen(line):
        log.append((now[0], "start", line))
        return FakeProcess(line, log)

    def run(line, **kw):
        log.append((now[0], "run", line))
        return subprocess.CompletedProcess(line, 0)

    def sleep(seconds: float) -> None:
        now[0] += seconds

    launch_with_faults(
        plan, epoch=100.0, popen=popen, run=run, clock=lambda: now[0], sleep=sleep
    )

    assert log == [
        (100.0, "start", ["cloud-1"]),
        (100.0, "start", ["field-1"]),
        (110.0, "run", ["kill", "KILL", "field-1"]),
        (112.0, "start", ["field-2"]),
        (115.0, "start", ["field-1", "--resume"]),
    ]
    faults = plan.run.meta["distributed"]["faults"]
    assert [(f["host"], f["action"], f["at"]) for f in faults] == [
        ("field-1", "kill", 10.0),
        ("field-2", "join", 12.0),
        ("field-1", "resume", 15.0),
    ]
    assert plan.codes == [0, 0, 0]


def test_only_a_host_of_edges_restarts_and_a_late_edges_parent_must_not_wait(
    tmp_path: Path,
) -> None:
    from onion_fl.experiment.distributed import check_faults
    from onion_fl.experiment.sweep import scenarios
    from tests.test_distributed import _params, grpc, on_machines

    config = on_machines(tmp_path, grpc())
    (scenario,) = scenarios(config)

    from onion_fl.experiment.emulated import host_load
    from onion_fl.experiment.real import edge_groups
    from onion_fl.experiment.runner import (
        _initial_state,
        _scenario_data,
        _shapes,
        physical,
    )

    topology, split, placement, _ = _scenario_data(scenario)
    _, state = _initial_state(config, _shapes(split), scenario.seed)
    resolved = physical(scenario, topology, placement, state)
    load = host_load(resolved, edge_groups(config, topology, placement))
    base = _params(config)

    def check(**faults):
        params = DistributedBackend(**base, checkpoints=True, **faults).params
        check_faults(params, topology, resolved, load)

    check(restarts=[{"host": "field-device-1", "at": 5, "down": 2}])
    with pytest.raises(ConfigError, match="only a host of edges"):
        check(restarts=[{"host": "fog-a", "at": 5, "down": 2}])
    with pytest.raises(ConfigError, match="no register_timeout"):
        check(joins={"field-device-2": 5})


# --- a stream over real processes ----------------------------------------------------


@pytest.mark.skipif(not _broker_up(), reason=f"no MQTT broker at {BROKER}")
def test_a_stream_runs_on_real_processes_with_the_rounds_of_the_simulation(
    workspace: Path,  # noqa: F811 - the fixture of test_experiment_stream
) -> None:
    from onion_fl.experiment.config import parse_experiment
    from onion_fl.experiment.real import run_real
    from onion_fl.experiment.runner import run_scenario
    from onion_fl.experiment.sweep import scenarios

    raw = experiment(workspace)
    link = {"transport": {"name": "mqtt", "broker": BROKER}, "codec": "json"}
    raw["topology"] = raw["topology"] | {
        "fog": raw["topology"]["fog"] | {"defaults": {"link_up": link}},
        "edge": {"link_up": link},
    }
    (sim,) = scenarios(parse_experiment(raw))
    (real,) = scenarios(
        parse_experiment(raw | {"runtime": {"timeout": 120}} | {"backend": "local"})
    )

    expected = json.loads((run_scenario(sim) / "summary.json").read_text("utf-8"))
    path = run_real(real)

    meta = json.loads((path / "run.json").read_text("utf-8"))
    summary = json.loads((path / "summary.json").read_text("utf-8"))
    assert meta["status"] == "finished", meta.get("processes")
    assert summary["rounds"] == expected["rounds"] == 5


def test_real_processes_host_the_validators_of_a_promotion(
    workspace: Path,  # noqa: F811
) -> None:
    """Every node edge_specs builds is hosted by some process, validators too."""
    from onion_fl.experiment.config import parse_experiment
    from onion_fl.experiment.real import edge_groups
    from onion_fl.experiment.runner import _scenario_data, edge_specs
    from onion_fl.experiment.sweep import scenarios

    raw = experiment(workspace) | {"promotion": "champion_challenger"}
    raw["stream"] = {k: v for k, v in raw["stream"].items() if k != "round_every"}
    raw["continuum"] = {
        "trigger": {"name": "schedule", "every": 300},
        "status_every": 300,
    }
    (scenario,) = scenarios(parse_experiment(raw))
    topology, split, placement, _ = _scenario_data(scenario)

    specs, _ = edge_specs(scenario, topology, split, placement)
    groups = edge_groups(scenario.config, topology, placement)

    built = {spec.id for members in specs.values() for spec in members}
    hosted = {node for members in groups.values() for node in members}
    assert any(n.startswith("validate-") for n in built)
    assert built == hosted


# --- the same failures in the simulation ---------------------------------------------


def with_deadlines():
    """``federation`` with a fog that closes its rounds without a down edge
    and starts without a late one, as on the machines."""
    fed = federation()
    fog = fed.aggregators["fog_0"]
    fog.deadline, fog.register_timeout = 1.0, 1.0
    return fed


def events(fed, node: str, name: str) -> list[dict]:
    return [e for e in fed.runtime.events if e["node"] == node and e["name"] == name]


def trained(fed, node: str) -> list[tuple[float, int]]:
    return [(e["t"], e["tags"]["examples"]) for e in events(fed, node, "edge.trained")]


def test_a_sim_restart_without_downtime_goes_on_as_if_it_never_stopped() -> None:
    from onion_fl.experiment.runner import fail_nodes

    whole = with_deadlines()
    whole.run()
    cut = with_deadlines()
    fail_nodes(cut, restarts={"a1": (3.5 * EVERY / SPEED, 0.0)}, joins={})
    cut.run()

    (resumed,) = events(cut, "a1", "node.resumed")
    assert resumed["tags"]["state"] == "restored"
    assert cut.edges["a1"] is not whole.edges["a1"]  # rebuilt, as by a new process
    model, consumed, steps = trace(cut)
    want_model, want_consumed, want_steps = trace(whole)
    assert steps == want_steps
    np.testing.assert_array_equal(consumed, want_consumed)
    for key, value in want_model.items():
        np.testing.assert_array_equal(model[key], value, err_msg=key)


def test_a_sim_restart_comes_back_with_the_state_saved_before_it_died() -> None:
    from onion_fl.experiment.runner import fail_nodes

    at, down = 3.5 * EVERY / SPEED, 7.0  # down through the round at 20 s
    whole = with_deadlines()
    whole.run()
    before = with_deadlines()
    before.run(until=at)
    saved = checkpoint_edge(before.edges["a2"], before.runtime._rng("a2"))
    cut = with_deadlines()
    fail_nodes(cut, restarts={"a2": (at, down)}, joins={})
    cut.run()

    (resumed,) = events(cut, "a2", "node.resumed")
    edge = before.edges["a2"]
    assert resumed["t"] == at + down
    assert resumed["tags"] == {
        "state": "restored",
        "round": saved.meta["round"],
        "consumed": int((edge._consumed_by >= 0).sum()),
        "predicted": int(edge._predicted.sum()),
        "clock": float(edge._clock),
        "fresh": [],
    }
    # Nothing while it was down; then its backlog, each row trained once.
    steps, want = trained(cut, "a2"), trained(whole, "a2")
    assert not [t for t, _ in steps if at <= t < at + down]
    assert steps[:-1] == want[:-1] and steps[-1][0] > at + down > want[-1][0]
    assert sum(n for _, n in steps) == sum(n for _, n in want)
    np.testing.assert_array_equal(
        cut.edges["a2"]._consumed_by >= 0, whole.edges["a2"]._consumed_by >= 0
    )
    assert cut.coordinator.finished


def test_a_late_edge_first_takes_part_after_it_joins() -> None:
    from onion_fl.experiment.runner import fail_nodes

    at = 2.2 * EVERY / SPEED
    whole = with_deadlines()
    whole.run()
    late = with_deadlines()
    fail_nodes(late, restarts={}, joins={"a2": at})
    late.run()

    assert min(e["t"] for e in late.runtime.events if e["node"] == "a2") == at
    steps = trained(late, "a2")
    assert steps and steps[0][0] > at
    # Its first training takes the rows that came before it joined, and in the
    # end it trained every row it trains when it is there from the start.
    consumed = late.edges["a2"]._consumed_by
    first = consumed[late.edges["a2"].stream.available_at < at * SPEED]
    assert len(first) and (first == first[0]).all() and first[0] >= 0
    np.testing.assert_array_equal(consumed >= 0, whole.edges["a2"]._consumed_by >= 0)
    assert late.coordinator.finished


def test_the_sim_backend_takes_the_failures_of_distributed() -> None:
    from onion_fl.experiment.backends import SimBackend

    restart = [{"host": "h", "at": 5, "down": 1}]
    with pytest.raises(ValueError, match="checkpoints: true"):
        SimBackend(restarts=restart)
    backend = SimBackend(checkpoints=True, restarts=restart, joins={"g": 9})
    assert backend.params.restarts[0].down == 1 and backend.params.joins == {"g": 9}
    with pytest.raises(ValueError, match="need.*infrastructure|an infrastructure"):
        backend.prepare(scenario=None)
