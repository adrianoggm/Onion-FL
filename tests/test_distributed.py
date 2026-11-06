"""Tests for the distributed backend (issue #199, spec I1 §10).

The inventory and the probe are tested in ``test_inventory``, and the gRPC
address map in ``test_transport_grpc``.

Nothing here needs SSH, Docker or another machine. SSH is replaced by a
stand-in that runs the remote line's program here, in one folder per machine:
the probe that runs is the real one, over the same pipes. The run on
containers acting as machines, against the simulation, is in
``test_runtime_equivalence``. Edges use the stub trainer on a CSV format
fixture (docs/RULES.md).
"""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import onion_fl
import onion_fl.experiment.distributed as distributed
from onion_fl.experiment.backends import DistributedBackend
from onion_fl.experiment.config import ConfigError, experiment_schema, parse_experiment
from onion_fl.experiment.distributed import (
    NUMERIC,
    DistributedPlan,
    Expected,
    PreflightError,
    clock_offset,
    judge,
    numeric_pins,
    prepare_distributed,
    pull_line,
    push,
    remote_line,
    run_probes,
    ssh_command,
    unpack,
)
from onion_fl.experiment.runner import plan
from onion_fl.experiment.sweep import scenarios
from onion_fl.infrastructure.backends import Backend, backends
from onion_fl.infrastructure.inventory import parse_inventory
from tests.test_runtime_equivalence import LAB, experiment

SRC = Path(onion_fl.__file__).resolve().parents[1]
HOSTS = ["cloud-1", "fog-a", "fog-b", *(f"field-device-{i}" for i in range(1, 5))]


def free_ports(n: int) -> list[int]:
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def inventory(**changes) -> dict:
    ports = free_ports(len(HOSTS))
    return {
        "machines": {
            host: {
                "address": f"machine-{host}",
                "user": "onion",
                "advertise": "127.0.0.1",
                "listen": "127.0.0.1",
                "grpc_port": port,
                "env": {"ONIONFL_HOST": host},
            }
            for host, port in zip(HOSTS, ports, strict=True)
        },
    } | changes


class LocalSsh:
    """Stands in for ``ssh``: runs the remote line's program here.

    Each machine is a folder named after its address. Only what the backend
    sends is understood: ``mkdir -p R && cd R && [env K=V...] program...``.
    """

    def __init__(self, homes: Path) -> None:
        self.homes = homes
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **kw):
        self.calls.append(command)
        home = self.homes / command[-2].split("@")[-1]
        tokens = shlex.split(command[-1])
        root = home / tokens[2]
        root.mkdir(parents=True, exist_ok=True)
        argv = tokens[tokens.index("&&", 4) + 1 :]
        env = dict(os.environ) | {"PYTHONPATH": str(SRC)}
        if argv[0] == "env":
            argv = argv[1:]
            while "=" in argv[0]:
                key, _, value = argv.pop(0).partition("=")
                env[key] = value
        argv[0] = sys.executable
        return subprocess.Popen(argv, cwd=root, env=env, **kw)


def on_machines(tmp_path: Path, transport, **backend) -> object:
    base = experiment(tmp_path, transport)
    raw = base.dump() | {
        "backend": {"name": "distributed", "inventory": inventory(), **backend},
        "infrastructure": {
            "name": LAB,
            "assignment": {
                "root": "cloud-1",
                "aggregators": {"fog_a": "fog-a", "fog_b": "fog-b"},
                "edges": {"name": "colocated", "per_host": 2},
            },
        },
    }
    return parse_experiment(raw)


def grpc() -> dict:
    pytest.importorskip("grpc")
    return {"name": "grpc"}


# --- commands ---------------------------------------------------------------------------


def test_the_ssh_command_never_prompts_and_names_the_key_and_the_known_hosts() -> None:
    inv = parse_inventory(
        {
            "machines": {"a": {"address": "10.0.0.5", "port": 2201, "user": "pi"}},
            "known_hosts": "/tmp/known",
            "host_key_check": "yes",
        }
    )
    machine = inv.machine("a").model_copy(update={"key": "/keys/id"})

    command = ssh_command("ssh", inv, machine, "true", no_stdin=True)

    assert command[0] == "ssh" and command[-2:] == ["pi@10.0.0.5", "true"]
    options = " ".join(command)
    assert "BatchMode=yes" in options and "StrictHostKeyChecking=yes" in options
    assert "-p 2201" in options and "-i /keys/id" in options
    assert "UserKnownHostsFile=/tmp/known" in options and "-n" in command


def test_the_remote_line_quotes_everything_and_runs_in_the_root() -> None:
    inv = parse_inventory(
        {
            "machines": {
                "a": {
                    "address": "a",
                    "root": "~/onion fl",
                    "env": {"PYTHONPATH": "/opt/x y"},
                }
            }
        }
    )

    line = remote_line(inv.machine("a"), ["python", "-c", "print('hi')"], pidfile="p")

    tokens = shlex.split(line)
    assert tokens[:6] == ["mkdir", "-p", "onion fl", "&&", "cd", "onion fl"]
    assert tokens[-5:] == ["env", "PYTHONPATH=/opt/x y", "python", "-c", "print('hi')"]
    assert "echo $$ > p && exec env" in line


def test_a_node_runs_on_its_machine_with_its_pid_recorded(tmp_path: Path) -> None:
    config = on_machines(tmp_path, grpc())
    (scenario,) = scenarios(config)
    inv = parse_inventory(inventory())

    class FakeRun:
        run_id = "r-1"

    params = DistributedBackend(**_params(config)).params
    plan_ = DistributedPlan(FakeRun(), scenario, ["fog-a"], inv, params, {}, {}, {})

    command = plan_.command("fog-a")

    assert command[-2] == "onion@machine-fog-a" and "-n" in command
    tokens = shlex.split(command[-1])
    assert tokens[-8:] == [
        "onion_fl",
        "node",
        "--id",
        "fog-a",
        "--run",
        "r-1",
        "--config",
        "runs/r-1/scenario.json",
    ]
    assert "runs/r-1/node.fog-a.pid" in command[-1]
    assert "node.fog-a.pid" in plan_.stop_command("fog-a")[-1]


def _params(config) -> dict:
    ref = config.backend_ref
    return {k: v for k, v in ref.items() if k != "name"}


# --- the preflight ----------------------------------------------------------------------


def expected(**changes) -> Expected:
    base = Expected(
        source="abc",
        pins={"numpy": "2.0.0", "torch": "2.5.1"},
        files={"cache/demo/k/meta.json": "d1"},
        max_offset=0.05,
    )
    for key, value in changes.items():
        setattr(base, key, value)
    return base


def facts(**changes) -> dict:
    return {
        "python": "3.11.9",
        "machine": "aarch64",
        "system": "Linux",
        "onion_fl": "0.1.0",
        "source": "abc",
        "pins": {"numpy": "2.0.0", "torch": "2.5.1"},
        "files": {"cache/demo/k/meta.json": "d1"},
        "reach": {"broker:1883": True},
        "offset": 0.004,
        "rtt": 0.002,
    } | changes


def test_a_machine_like_the_launcher_passes() -> None:
    verdict = judge("pi", facts(), expected())

    assert (verdict.errors, verdict.warnings, verdict.stale) == ([], [], [])


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"python": "3.9.2"}, "needs 3.11"),
        ({"source": "other"}, "other onion_fl code"),
        ({"source": None, "error": "import onion_fl: boom"}, "not installed"),
        ({"offset": -0.2}, r"-0\.200 s off the launcher"),
        ({"reach": {"broker:1883": False}}, "cannot reach broker:1883"),
        ({"bind": False}, "gRPC port is in use"),
    ],
)
def test_what_keeps_a_machine_out_is_said(change, reason) -> None:
    verdict = judge("pi", facts(**change), expected(bind=True))

    assert any(re.search(reason, e) for e in verdict.errors), verdict.errors


def test_ntp_is_checked_when_it_is_the_reference() -> None:
    ok = judge("pi", facts(ntp=0.01), expected(reference="ntp"))
    late = judge("pi", facts(ntp=0.3), expected(reference="ntp"))
    silent = judge(
        "pi", facts(ntp=None, ntp_error="timeout"), expected(reference="ntp")
    )

    assert ok.errors == []
    assert "off NTP" in late.errors[0] and "no NTP answer (timeout)" in silent.errors[0]


def test_other_numeric_versions_warn_and_missing_data_is_stale() -> None:
    verdict = judge(
        "pi",
        facts(pins={"numpy": "2.0.0", "torch": "2.4.0"}, files={}),
        expected(),
    )

    assert verdict.errors == []
    assert verdict.warnings == [
        "torch 2.4.0 here, 2.5.1 on the launcher: the bits may differ from sim"
    ]
    assert verdict.stale == ["cache/demo/k/meta.json"]


def test_the_clock_offset_comes_from_the_shortest_round_trip() -> None:
    samples = [(10.0, 10.9, 11.0), (20.0, 20.02, 20.01), (30.0, 31.0, 30.5)]

    offset, rtt = clock_offset(samples)

    assert offset == pytest.approx(0.015) and rtt == pytest.approx(0.01)


def test_every_machine_is_probed_at_once_through_ssh(tmp_path: Path) -> None:
    inv = parse_inventory(inventory())
    hosts = HOSTS[:3]
    ports = {h: inv.machine(h).grpc_port for h in hosts}
    requests = {
        h: {"pins": list(NUMERIC), "bind": ["127.0.0.1", ports[h]]} for h in hosts
    }
    reach = {h: [f"127.0.0.1:{ports[o]}" for o in hosts if o != h] for h in hosts}
    ssh = LocalSsh(tmp_path / "homes")

    found = run_probes("ssh", inv, requests, reach, samples=3, popen=ssh)

    assert sorted(found) == hosts
    for host in hosts:
        assert "error" not in found[host], found[host]
        assert all(found[host]["reach"].values()), found[host]["reach"]
        assert abs(found[host]["offset"]) < 1 and found[host]["rtt"] >= 0
    assert len(ssh.calls) == 3


def test_a_machine_that_does_not_answer_gets_an_error(tmp_path: Path) -> None:
    inv = parse_inventory(inventory())

    def unreachable(command, **kw):
        return subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stderr.write('no route')"], **kw
        )

    found = run_probes("ssh", inv, {"cloud-1": {}}, {}, popen=unreachable)

    assert "no answer" in found["cloud-1"]["error"]
    assert "no route" in found["cloud-1"]["error"]
    verdict = judge("cloud-1", found["cloud-1"], expected())
    assert verdict.errors and "cannot probe it" in verdict.errors[0]


# --- the backend ------------------------------------------------------------------------


def test_the_distributed_backend_is_registered_and_exported() -> None:
    assert "distributed" in backends.names()
    names = {p["name"] for p in experiment_schema()["plugins"]["backend"]}
    assert "distributed" in names
    assert isinstance(DistributedBackend(inventory="x.yaml"), Backend)


def test_the_distributed_backend_needs_an_infrastructure(tmp_path: Path) -> None:
    raw = experiment(tmp_path, grpc()).dump() | {
        "backend": {"name": "distributed", "inventory": inventory()}
    }
    with pytest.raises(ConfigError, match="give the experiment an infrastructure"):
        parse_experiment(raw)


def test_plan_shows_the_machines_and_refuses_a_host_without_one(tmp_path: Path) -> None:
    (preview,) = plan(on_machines(tmp_path, grpc()))

    shown = preview["backend"]
    assert shown["name"] == "distributed"
    assert set(shown["machines"]) == set(HOSTS)
    assert shown["machines"]["fog-a"]["address"] == "machine-fog-a"
    short = inventory()
    del short["machines"]["fog-b"]
    with pytest.raises(ConfigError, match=r"no machine for \['fog-b'\]"):
        plan(on_machines(tmp_path, grpc(), inventory=short))


def test_prepare_checks_every_machine_and_writes_what_each_one_reads(
    tmp_path: Path,
) -> None:
    config = on_machines(tmp_path, grpc())
    (scenario,) = scenarios(config)
    ssh = LocalSsh(tmp_path / "homes")

    prepared = prepare_distributed(
        scenario, DistributedBackend(**_params(config)).params, popen=ssh
    )

    assert prepared.groups == sorted(HOSTS)
    document = json.loads((prepared.run.path / "scenario.json").read_text("utf-8"))
    assert document["config"]["paths"] == {
        k: k for k in ("topologies", "infrastructures", "datasets", "runs", "cache")
    }
    assert document["config"]["data"]["datasets"]["demo"]["descriptor"] == (
        "descriptors/demo/demo.yaml"
    )
    assert "descriptors/demo/demo.yaml" in prepared.files
    inv = prepared.inventory
    peers = document["grpc"]["peers"]
    assert peers["cloud"] == f"127.0.0.1:{inv.machine('cloud-1').grpc_port}"
    assert peers["fog_a"] == f"127.0.0.1:{inv.machine('fog-a').grpc_port}"
    nodes = [n for members in document["groups"].values() for n in members]
    assert sorted(peers) == sorted(nodes) and len(nodes) == len(set(nodes))
    assert document["grpc"]["listen"]["fog-b"].startswith("127.0.0.1:")
    assert "broker" not in document  # no MQTT link
    assert (prepared.run.path / document["initial"]).is_file()
    (cache,) = prepared.caches
    assert all(prepared.stale[h] == [cache] for h in HOSTS)  # no machine has it yet
    meta = prepared.run.meta
    assert meta["backend"] == "distributed" and meta["infrastructure_id"]
    machines = meta["distributed"]["machines"]
    assert set(machines) == set(HOSTS)
    assert machines["fog-a"]["python"] and "offset" in machines["fog-a"]
    assert "key" not in json.dumps(meta["distributed"])  # no key path is recorded
    assert len(ssh.calls) == len(HOSTS)  # one probe per machine, nothing else yet
    prepared.run.finish(status="failed")  # nothing ran


def test_mqtt_links_take_the_broker_of_the_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = on_machines(tmp_path, {"name": "mqtt", "broker": "localhost:1883"})
    raw = config.dump()
    raw["infrastructure"]["assignment"]["services"] = {"mqtt_broker": "cloud-1"}
    (scenario,) = scenarios(parse_experiment(raw))
    seen = {}

    def probes(ssh, inv, requests, reach, **kw):
        seen.update(reach)
        return {h: facts(source=None, error="x") for h in requests}

    monkeypatch.setattr(distributed, "run_probes", probes)
    with pytest.raises(PreflightError, match="onion_fl is not installed"):
        prepare_distributed(scenario, DistributedBackend(**_params(config)).params)

    assert all(targets == ["127.0.0.1:1883"] for targets in seen.values())
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def test_a_failed_preflight_lists_every_reason_and_creates_no_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = on_machines(tmp_path, grpc(), push_data=False)
    (scenario,) = scenarios(config)

    def probes(ssh, inv, requests, reach, **kw):
        out = {h: facts(source="abc", files={}) for h in requests}
        out["fog-a"] = facts(offset=1.0, files={})
        return out

    monkeypatch.setattr(distributed, "run_probes", probes)
    with pytest.raises(PreflightError) as caught:
        prepare_distributed(scenario, DistributedBackend(**_params(config)).params)

    message = str(caught.value)
    assert "fog-a: its clock is +1.000 s off" in message
    assert "other onion_fl code" in message  # "abc" is not this source
    assert "push_data is off" in message
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def test_one_grpc_port_per_machine_needs_one_grpc_setting(tmp_path: Path) -> None:
    config = on_machines(tmp_path, grpc())
    raw = config.dump()
    raw["topology"]["edge"]["link_up"] = {
        "transport": {"name": "grpc", "call_timeout": 3}
    }
    (scenario,) = scenarios(parse_experiment(raw))

    with pytest.raises(ConfigError, match="same transport settings"):
        prepare_distributed(scenario, DistributedBackend(**_params(config)).params)


# --- moving files ----------------------------------------------------------------------


def test_push_sends_one_tar_stream_with_the_remote_paths(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "demo.yaml").write_text("name: demo\n", encoding="utf-8")
    inv = parse_inventory(inventory())
    received = {}
    names = tmp_path / "names.json"

    def capture(command, **kw):
        received["command"] = command
        script = (
            "import json, sys, tarfile; t = tarfile.open(fileobj=sys.stdin.buffer, "
            "mode='r|'); open(sys.argv[1], 'w').write(json.dumps([m.name for m in t]))"
        )
        return subprocess.Popen([sys.executable, "-c", script, str(names)], **kw)

    push(
        "ssh",
        inv,
        inv.machine("cloud-1"),
        {
            "descriptors/demo/demo.yaml": tmp_path / "demo.yaml",
            "cache/demo/k": tmp_path / "cache",
        },
        run=capture,
    )

    assert received["command"][-1] == "mkdir -p onionfl && tar xf - -C onionfl"
    assert sorted(json.loads(names.read_text())) == [
        "cache/demo/k",
        "cache/demo/k/meta.json",
        "descriptors/demo/demo.yaml",
    ]


def test_unpack_takes_only_event_files_and_the_model(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in (
            ("events.fog-a.jsonl", b"{}\n"),
            ("../../evil.jsonl", b"x"),
            ("model.npz", b"npz"),
            ("events.x.jsonl/../../../escape", b"x"),
            ("node.fog-a.pid", b"1"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    written = unpack(buffer.getvalue(), tmp_path)

    assert written == ["events.fog-a.jsonl", "model.npz"]
    assert sorted(p.name for p in tmp_path.iterdir()) == written
    assert "tar cf - -T -" in pull_line("r-1") and "runs/r-1" in pull_line("r-1")


def test_numeric_pins_cover_the_libraries_that_decide_the_bits() -> None:
    assert set(numeric_pins()) == set(NUMERIC)
