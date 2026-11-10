"""Simulation and real runtime give the same final model (spec §9.3, issue #100).

With full quorum and the same seed, the coordinator's final model must match
in SimRuntime, in RealRuntime over the memory bus, gRPC and Flower, and over
MQTT, gRPC and Flower with one process per group. Flower needs the ``flower``
extra, and its tests skip without it. The edges use the stub trainer with
per-node noise: it
learns nothing, but every edge sends a different, seeded update, so any
difference in participants, rng streams or aggregation order would show.
The dataset is a CSV format fixture (docs/RULES.md).
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from onion_fl.experiment.config import parse_experiment
from onion_fl.experiment.runner import _scenario_data, build_scenario, run_experiment
from onion_fl.experiment.sweep import scenarios
from onion_fl.runtime.real import RealRuntime

BROKER = os.environ.get("ONIONFL_MQTT", "localhost:1883")


def _broker_up() -> bool:
    host, _, port = BROKER.partition(":")
    try:
        with socket.create_connection((host, int(port or 1883)), timeout=0.5):
            return True
    except OSError:
        return False


def experiment(tmp_path: Path, transport, mode: str = "sim"):
    rows = ["pp,cond,f1,f2"] + [
        f"{s},{'NT'[i % 2]},{s + i},{s * 3 - i}" for s in range(1, 11) for i in range(4)
    ]
    (tmp_path / "raw").mkdir(exist_ok=True)
    (tmp_path / "raw" / "t.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "demo.yaml").write_text(
        textwrap.dedent(
            f"""
            name: demo
            root: {(tmp_path / "raw").as_posix()}
            source: {{reader: csv, path: t.csv}}
            steps:
              - subject: {{column: pp}}
              - label: {{task: stress, column: cond, map: {{"N": 0, "T": 1}}}}
              - features: {{}}
            """
        ),
        encoding="utf-8",
    )
    link = {"transport": transport}
    return parse_experiment(
        {
            "name": "equivalence",
            "topology": {
                "name": "two_fogs",
                "levels": ["global", "fog", "edge"],
                "root": {"id": "cloud"},
                "fog": {
                    "defaults": {"link_up": link},
                    "nodes": [{"id": "fog_a"}, {"id": "fog_b"}],
                },
                "edge": {"link_up": link},
            },
            "data": {
                "datasets": {"demo": {"descriptor": str(tmp_path / "demo.yaml")}},
                "roles": {"test": 0.2},
                "placement": {"name": "mixing", "alpha": 1.0},
            },
            "learning": {
                "model": {
                    "name": "modular_mlp",
                    "adapter_width": 4,
                    "trunk_hidden": [4],
                },
                "trainer": {"name": "stub", "shift": 0.0, "noise": 0.1},
            },
            "rounds": 3,
            "evaluation": {"global": {"every": None}},
            "runtime": {"mode": mode, "timeout": 60, "heartbeat": None},
            "paths": {"runs": str(tmp_path / "runs"), "cache": str(tmp_path / "cache")},
        }
    )


def simulated(config) -> dict[str, np.ndarray]:
    (scenario,) = scenarios(config)
    topology, split, placement, _ = _scenario_data(scenario)
    federation = build_scenario(scenario, topology, split, placement)
    federation.run()
    assert federation.coordinator.finished
    return federation.coordinator.state


def assert_same_model(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-7, err_msg=key)


def test_the_noise_makes_the_check_meaningful(tmp_path: Path) -> None:
    config = experiment(tmp_path, "memory")
    (scenario,) = scenarios(config)
    topology, split, placement, _ = _scenario_data(scenario)
    initial = build_scenario(scenario, topology, split, placement).coordinator.state

    final = simulated(config)

    assert any(not np.allclose(final[k], initial[k]) for k in final)


def test_the_real_runtime_in_one_process_matches_the_simulation(tmp_path: Path) -> None:
    config = experiment(tmp_path, {"name": "memory", "bus": "equivalence"})
    (scenario,) = scenarios(config)
    topology, split, placement, _ = _scenario_data(scenario)
    runtime = RealRuntime("equivalence", seed=scenario.seed, heartbeat_s=None)
    federation = build_scenario(scenario, topology, split, placement, runtime=runtime)

    runtime.run(timeout=30)

    assert federation.coordinator.finished
    assert_same_model(simulated(config), federation.coordinator.state)


DIRECT = ["grpc", "flower"]  # transports with direct links, no broker


def _direct(tmp_path: Path, name: str = "grpc") -> dict:
    if name == "flower":
        pytest.importorskip("flwr", minversion="1.21")
    pytest.importorskip("grpc")
    return {"name": name, "directory": str(tmp_path / f"peers_{name}")}


@pytest.mark.parametrize("name", DIRECT)
def test_the_real_runtime_over_a_direct_transport_matches_the_simulation(
    tmp_path: Path, name: str
) -> None:
    config = experiment(tmp_path, _direct(tmp_path, name))
    (scenario,) = scenarios(config)
    topology, split, placement, _ = _scenario_data(scenario)
    runtime = RealRuntime(f"equivalence-{name}", seed=scenario.seed, heartbeat_s=None)
    federation = build_scenario(scenario, topology, split, placement, runtime=runtime)

    runtime.run(timeout=30)

    assert federation.coordinator.finished
    assert_same_model(simulated(config), federation.coordinator.state)


@pytest.mark.parametrize("name", DIRECT)
def test_a_direct_transport_with_one_process_per_group_matches_the_simulation(
    tmp_path: Path, name: str
) -> None:
    transport = _direct(tmp_path, name)
    expected = simulated(experiment(tmp_path, transport))

    (path,) = run_experiment(experiment(tmp_path, transport, mode="real"))

    with np.load(path / "model.npz", allow_pickle=False) as saved:
        assert_same_model(expected, {k: saved[k] for k in saved.files})


def assert_no_secrets(path: Path, *secrets: str) -> None:
    """Nothing in a run folder holds a key, a certificate or a password (#227)."""
    for file in sorted(p for p in path.rglob("*") if p.suffix in (".json", ".jsonl")):
        text = file.read_text(encoding="utf-8")
        assert "-----BEGIN" not in text, file.name
        for secret in secrets:
            assert secret not in text, file.name


def test_grpc_over_mutual_tls_with_one_process_per_group_matches_the_simulation(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cryptography")
    from tests.certs import Pki

    pki = Pki(tmp_path / "pki")
    cert, key = pki.issue("node", ["127.0.0.1"])
    tls = {"ca": str(pki.ca), "cert": str(cert), "key": str(key)}
    grpc = _direct(tmp_path) | {"tls": tls}
    expected = simulated(experiment(tmp_path, grpc))

    (path,) = run_experiment(experiment(tmp_path, grpc, mode="real"))

    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished", meta.get("processes")
    with np.load(path / "model.npz", allow_pickle=False) as saved:
        assert_same_model(expected, {k: saved[k] for k in saved.files})
    pem = key.read_text(encoding="utf-8").splitlines()[1]  # a line of the key itself
    assert_no_secrets(path, pem)
    assert "node.key" in json.dumps(meta)  # the path is what the run keeps


def test_local_over_grpc_and_over_flower_give_the_same_model_and_bytes(
    tmp_path: Path,
) -> None:
    from onion_fl.experiment.fidelity import fidelity

    for name in DIRECT:
        raw = experiment(tmp_path, _direct(tmp_path, name)).dump()
        raw = raw | {"backend": "local"}
        raw["paths"] = raw["paths"] | {"runs": str(tmp_path / f"runs_{name}")}
        run_experiment(parse_experiment(raw))

    result = fidelity(
        tmp_path / "runs_grpc", tmp_path / "runs_flower", metrics=["time"]
    )

    (pair,) = result["pairs"]
    assert pair["functional"]["model"]["bitwise"], pair["functional"]
    assert result["functional"]["complete"], pair["functional"]
    assert pair["functional"]["bytes"]["a"] == pair["functional"]["bytes"]["b"] > 0


def _round_bytes(events) -> dict:
    out: dict = {}
    for e in events:
        if e["name"] == "message.sent" and e["tags"]["round"] is not None:
            key = (e["node"], e["tags"]["dst"], e["tags"]["round"])
            out[key] = out.get(key, 0) + e["value"]
    return out


def test_the_real_runtime_counts_the_bytes_the_simulation_counts(
    tmp_path: Path,
) -> None:
    config = experiment(tmp_path, {"name": "memory", "bus": "bytes"})
    (scenario,) = scenarios(config)
    topology, split, placement, _ = _scenario_data(scenario)
    sim = build_scenario(scenario, topology, split, placement)
    sim.run()
    runtime = RealRuntime("bytes", seed=scenario.seed, heartbeat_s=None)
    build_scenario(scenario, topology, split, placement, runtime=runtime)

    runtime.run(timeout=30)

    expected = _round_bytes(sim.runtime.events)
    assert expected and _round_bytes(runtime.events) == expected
    framed = [e for e in runtime.events if e["name"] == "message.sent"]
    assert all(e["tags"]["frame"] > 0 for e in framed)


@pytest.mark.skipif(not _broker_up(), reason=f"no MQTT broker at {BROKER}")
def test_mqtt_with_one_process_per_group_matches_the_simulation(tmp_path: Path) -> None:
    mqtt = {"name": "mqtt", "broker": BROKER}
    expected = simulated(experiment(tmp_path, mqtt))

    (path,) = run_experiment(experiment(tmp_path, mqtt, mode="real"))

    with np.load(path / "model.npz", allow_pickle=False) as saved:
        assert_same_model(expected, {k: saved[k] for k in saved.files})


@pytest.mark.skipif(not _broker_up(), reason=f"no MQTT broker at {BROKER}")
def test_the_fidelity_of_local_against_sim_is_functionally_complete(
    tmp_path: Path,
) -> None:
    from onion_fl.experiment.fidelity import fidelity

    mqtt = {"name": "mqtt", "broker": BROKER}
    folders = {}
    for backend in ("sim", "local"):
        raw = experiment(tmp_path, mqtt).dump() | {"backend": backend}
        raw["paths"] = raw["paths"] | {"runs": str(tmp_path / f"runs_{backend}")}
        run_experiment(parse_experiment(raw))
        folders[backend] = tmp_path / f"runs_{backend}"

    result = fidelity(folders["local"], folders["sim"], metrics=["time"])

    assert result["backends"] == {"a": ["local"], "b": ["sim"]}
    assert result["functional"]["complete"], result["pairs"][0]["functional"]
    assert {"round_time", "link_latency", "duration"} <= set(result["performance"])


@pytest.mark.skipif(not _broker_up(), reason=f"no MQTT broker at {BROKER}")
def test_a_real_run_feeds_its_sinks_and_labels_its_traffic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onion_fl.experiment import runner
    from onion_fl.observability.events import read_events

    seen = []

    class Recorder:
        def write(self, event) -> None:
            seen.append(event["name"])

        def close(self) -> None:
            pass

    monkeypatch.setattr(runner, "_sinks", lambda config: [Recorder()])
    mqtt = {"name": "mqtt", "broker": BROKER}

    (path,) = run_experiment(experiment(tmp_path, mqtt, mode="real"))

    traffic = [
        e
        for e in read_events(path / "events.jsonl")
        if e["name"] == "diagnostic.communication"
    ]
    assert traffic and all(e["level"] and e["role"] for e in traffic)
    assert {"run.finished", "diagnostic.communication"} <= set(seen)


@pytest.mark.skipif(not _broker_up(), reason=f"no MQTT broker at {BROKER}")
def test_local_over_mqtt_and_over_grpc_give_the_same_model_and_measure_the_rest(
    tmp_path: Path,
) -> None:
    from onion_fl.experiment.fidelity import fidelity

    transports = {"mqtt": {"name": "mqtt", "broker": BROKER}, "grpc": _direct(tmp_path)}
    for name, transport in transports.items():
        raw = experiment(tmp_path, transport).dump() | {"backend": "local"}
        raw["paths"] = raw["paths"] | {"runs": str(tmp_path / f"runs_{name}")}
        run_experiment(parse_experiment(raw))

    result = fidelity(tmp_path / "runs_mqtt", tmp_path / "runs_grpc", metrics=["time"])

    (pair,) = result["pairs"]
    assert pair["functional"]["model"]["bitwise"], pair["functional"]
    assert result["functional"]["complete"], pair["functional"]
    assert pair["functional"]["bytes"]["a"] == pair["functional"]["bytes"]["b"] > 0
    assert {"round_time", "link_latency", "duration"} <= set(result["performance"])


# --- the emulated backend (issue #195) -------------------------------------------

EMULATED = {
    "name": "emulated",
    "image": os.environ.get("ONIONFL_EMULATED_IMAGE", "onionfl-emulated:latest"),
    "project": os.environ.get("ONIONFL_EMULATED_PROJECT", "onionfl-test"),
    "subnet": os.environ.get("ONIONFL_EMULATED_SUBNET", "10.195.0.0/16"),
}


def _emulator_up() -> bool:
    """Docker answers and has the host image: the emulator is there."""
    try:
        found = subprocess.run(
            ["docker", "image", "inspect", EMULATED["image"]],
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return found.returncode == 0


LAB = {
    "name": "emulated_lab",
    "profiles": {
        "cloud": {"cpus": 2, "memory": "2GB"},
        "fog": {"cpus": 1, "memory": "1GB"},
        "device": {"cpus": 1, "memory": "1GB"},
    },
    "sites": {
        "dc": {"hosts": {"cloud-1": "cloud"}},
        "fog_a": {"hosts": {"fog-a": "fog"}},
        "fog_b": {"hosts": {"fog-b": "fog"}},
        "field": {"pool": {"device": 4}},
    },
    "network": {
        "links": [
            {"between": ["field", "fog_a"], "latency": "5ms", "bandwidth": "50Mbps"},
            {"between": ["field", "fog_b"], "latency": "10ms", "bandwidth": "20Mbps"},
            {"between": ["fog_a", "dc"], "latency": "20ms", "bandwidth": "10Mbps"},
            {"between": ["fog_b", "dc"], "latency": "30ms", "bandwidth": "10Mbps"},
        ]
    },
    "services": {"mqtt_broker": "cloud-1"},
}


def on_the_lab(config, backend, runs: Path):
    """``config`` on ``LAB``, two edges per device, run by ``backend``."""
    raw = config.dump() | {
        "backend": backend,
        "infrastructure": {
            "name": LAB,
            "assignment": {
                "root": "cloud-1",
                "aggregators": {"fog_a": "fog-a", "fog_b": "fog-b"},
                "edges": {"name": "colocated", "per_host": 2},
            },
        },
    }
    raw["paths"] = raw["paths"] | {"runs": str(runs)}
    return parse_experiment(raw)


@pytest.mark.skipif(not _emulator_up(), reason="no Docker or no emulated host image")
def test_the_emulated_backend_matches_the_simulation(tmp_path: Path) -> None:
    from onion_fl.experiment.fidelity import fidelity
    from onion_fl.observability.events import read_events

    mqtt = {"name": "mqtt", "broker": BROKER}  # the emulation brings its own broker
    base = experiment(tmp_path, mqtt)
    sim = on_the_lab(base, "sim", tmp_path / "runs_sim")
    emulated = on_the_lab(base, EMULATED, tmp_path / "runs_emulated")

    (sim_path,) = run_experiment(sim)
    (path,) = run_experiment(emulated)

    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished", meta.get("processes")
    assert meta["backend"] == "emulated" and meta["infrastructure_id"]
    assert meta["emulation"]["containers"] == 12  # 7 hosts, 4 routers, the broker
    with np.load(path / "model.npz", allow_pickle=False) as saved:
        assert_same_model(simulated(sim), {k: saved[k] for k in saved.files})
    expected = _round_bytes(read_events(sim_path / "events.jsonl"))
    assert expected and _round_bytes(read_events(path / "events.jsonl")) == expected
    result = fidelity(
        tmp_path / "runs_emulated", tmp_path / "runs_sim", metrics=["time"]
    )
    functional = result["pairs"][0]["functional"]
    assert result["functional"]["complete"], functional
    assert functional["model"]["bitwise"], functional["model"]


# --- the distributed backend (issue #199) ----------------------------------------------
#
# Docker containers act as the machines: each runs an SSH server and the
# emulated backend's Python, with this checkout's onion_fl mounted read-only.
# The launcher reaches them only through SSH, as it would real machines, with
# a throwaway key made for the test. The broker runs in cloud-1's network
# namespace, as on a real cloud host.
#
# With ``cryptography`` installed, each machine also gets its own certificate
# (one test CA) at the same path, ``TLS_ROOT``, and cloud-1 a second broker
# that speaks only TLS (issue #227).

DISTRIBUTED = {
    "base": EMULATED["image"],
    "image": os.environ.get("ONIONFL_DISTRIBUTED_IMAGE", "onionfl-distributed:latest"),
    "prefix": os.environ.get("ONIONFL_DISTRIBUTED_PREFIX", "onionfl-dist-test"),
    "mqtt_port": os.environ.get("ONIONFL_DISTRIBUTED_MQTT_PORT"),  # published if set
    "ssh": os.environ.get("ONIONFL_SSH", "ssh"),
}
MACHINES = ["cloud-1", "fog-a", "fog-b", "field-device-1", "field-device-2"]
TLS_ROOT = "/onionfl/tls"  # on every machine: ca.pem, node.pem, node.key, password
MQTT_USER, MQTT_PASSWORD = "onion", f"pw-{os.urandom(8).hex()}"
MACHINE_LAB = LAB | {
    "name": "distributed_lab",
    "sites": LAB["sites"] | {"field": {"pool": {"device": 2}}},
}


def _machines_possible() -> bool:
    import shutil

    tools = (DISTRIBUTED["ssh"], "ssh-keygen")
    return all(shutil.which(t) for t in tools) and _emulator_up()


def _docker(*args: str, check: bool = True) -> str:
    done = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=600
    )
    if check and done.returncode != 0:
        raise RuntimeError(f"docker {args[0]}: {done.stderr.strip()}")
    return done.stdout.strip()


@pytest.fixture(scope="module")
def machines(tmp_path_factory: pytest.TempPathFactory):
    """Five containers acting as machines; yields their inventory."""
    from importlib import resources

    import onion_fl
    from onion_fl.experiment.distributed import ssh_command
    from onion_fl.experiment.emulated import ensure_image
    from onion_fl.infrastructure.inventory import parse_inventory

    ensure_image(DISTRIBUTED["base"])  # the launcher's numeric versions
    dockerfile = resources.files("onion_fl.infrastructure") / "distributed.Dockerfile"
    subprocess.run(
        ["docker", "build", "-q", "-t", DISTRIBUTED["image"]]
        + ["--build-arg", f"BASE={DISTRIBUTED['base']}", "-"],
        input=dockerfile.read_bytes(),
        capture_output=True,
        check=True,
        timeout=900,
    )
    folder = tmp_path_factory.mktemp("machines")
    key = folder / "id_ed25519"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "onionfl-test"]
        + ["-f", str(key)],
        check=True,
        capture_output=True,
    )
    public = (folder / "id_ed25519.pub").read_text(encoding="utf-8").strip()
    source = Path(onion_fl.__file__).resolve().parent.as_posix()
    mount = f"type=bind,source={source},target=/onionfl/src/onion_fl,readonly"
    prefix = DISTRIBUTED["prefix"]
    pki = None
    if importlib.util.find_spec("cryptography") is not None:
        from tests.certs import Pki

        pki = Pki(folder / "pki")
        for host in MACHINES:
            own = folder / "tls" / host
            pki.issue("node", [f"{prefix}-{host}", host], own)
            (own / "ca.pem").write_bytes(pki.ca.read_bytes())
            (own / "password").write_text(MQTT_PASSWORD, encoding="utf-8")
    network, started = f"{prefix}-net", []
    _docker("network", "create", network)
    try:
        for host in MACHINES:
            name = f"{prefix}-{host}"
            published = ["-p", "127.0.0.1::22"]
            if host == "cloud-1" and DISTRIBUTED["mqtt_port"]:
                published += ["-p", f"127.0.0.1:{DISTRIBUTED['mqtt_port']}:1883"]
            mounts = ["--mount", mount]
            if pki is not None:
                own = (folder / "tls" / host).as_posix()
                mounts += [
                    "--mount", f"type=bind,source={own},target={TLS_ROOT},readonly"
                ]  # fmt: skip
            _docker(
                "run", "-d", "--name", name, "--hostname", host,
                "--network", network, *published,
                "-e", f"AUTHORIZED_KEY={public}", *mounts,
                DISTRIBUTED["image"],
            )  # fmt: skip
            started.append(name)
        broker = f"{prefix}-mqtt"
        _docker(
            "run", "-d", "--name", broker, "--network", f"container:{prefix}-cloud-1",
            "eclipse-mosquitto:2", "mosquitto", "-c", "/mosquitto-no-auth.conf",
        )  # fmt: skip
        started.append(broker)
        if pki is not None:
            from tests.certs import BROKER_TLS, mosquitto_tls, wait_for_mosquitto

            command = mosquitto_tls(
                pki,
                folder / "mqtts",
                [f"{prefix}-cloud-1", "cloud-1"],
                MQTT_USER,
                MQTT_PASSWORD,
            )
            mqtts = f"{prefix}-mqtts"
            target = (
                f"type=bind,source={(folder / 'mqtts').as_posix()},target={BROKER_TLS}"
            )
            _docker(
                "run", "-d", "--name", mqtts,
                "--network", f"container:{prefix}-cloud-1", "--mount", target,
                "eclipse-mosquitto:2", *command,
            )  # fmt: skip
            started.append(mqtts)
            wait_for_mosquitto(mqtts)
        ports = {
            host: int(_docker("port", f"{prefix}-{host}", "22").rsplit(":")[-1])
            for host in MACHINES
        }
        raw = {
            "machines": {
                host: {
                    "address": "127.0.0.1",
                    "port": ports[host],
                    "user": "root",
                    "key": str(key),
                    "python": "python",
                    "root": "/root/onionfl",
                    "advertise": f"{prefix}-{host}",
                    "env": {"PYTHONPATH": "/onionfl/src"},
                }
                for host in MACHINES
            },
            "known_hosts": str(folder / "known_hosts"),
            # Docker Desktop's VM keeps its clock close to the host's, not exact.
            "clock": {"max_offset": 0.25},
        }
        inventory = parse_inventory(raw)
        deadline = time.monotonic() + 60
        for host in MACHINES:
            machine = inventory.machine(host)
            line = ssh_command(
                DISTRIBUTED["ssh"], inventory, machine, "true", no_stdin=True
            )
            while subprocess.run(line, capture_output=True, timeout=30).returncode:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no SSH to {host}")
                time.sleep(0.5)
        yield raw
    finally:
        for name in started:
            _docker("rm", "-f", name, check=False)
        _docker("network", "rm", network, check=False)


def fresh_ports(raw: dict) -> dict:
    """The inventory with a gRPC port per machine that no earlier run took.

    Each port was free on this host a moment ago; inside the containers it
    only has to differ from the last run's, whose sockets may linger."""
    socks = [socket.socket() for _ in raw["machines"]]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    machines = {
        host: machine | {"grpc_port": port}
        for (host, machine), port in zip(raw["machines"].items(), ports, strict=True)
    }
    return raw | {"machines": machines}


def on_the_machines(config, backend, runs: Path):
    """``config`` on ``MACHINE_LAB``, four edges per device, run by ``backend``."""
    raw = config.dump() | {
        "backend": backend,
        "infrastructure": {
            "name": MACHINE_LAB,
            "assignment": {
                "root": "cloud-1",
                "aggregators": {"fog_a": "fog-a", "fog_b": "fog-b"},
                "edges": {"name": "colocated", "per_host": 4},
            },
        },
    }
    raw["paths"] = raw["paths"] | {"runs": str(runs)}
    return parse_experiment(raw)


@pytest.mark.skipif(
    not _machines_possible(), reason="no Docker, no ssh or no emulated host image"
)
@pytest.mark.parametrize("transport", ["mqtt", "grpc"])
def test_the_distributed_backend_matches_the_simulation(
    tmp_path: Path, machines, transport: str
) -> None:
    from onion_fl.experiment.fidelity import fidelity
    from onion_fl.observability.events import read_events

    if transport == "grpc":
        pytest.importorskip("grpc")
    link = {"name": transport} | ({"broker": BROKER} if transport == "mqtt" else {})
    base = experiment(tmp_path, link)
    backend = {
        "name": "distributed",
        "inventory": fresh_ports(machines),
        "ssh": DISTRIBUTED["ssh"],
    }
    sim = on_the_machines(base, "sim", tmp_path / "runs_sim")
    distributed = on_the_machines(base, backend, tmp_path / "runs_distributed")

    (sim_path,) = run_experiment(sim)
    (path,) = run_experiment(distributed)

    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished", meta.get("processes")
    assert meta["backend"] == "distributed" and meta["infrastructure_id"]
    ran = meta["distributed"]["machines"]
    assert sorted(ran) == sorted(MACHINES)
    assert all(m["system"] == "Linux" and abs(m["offset"]) < 0.25 for m in ran.values())
    assert meta["distributed"]["transports"] == [transport]
    assert "model.npz" in meta["distributed"]["pulled"]["cloud-1"]
    with np.load(path / "model.npz", allow_pickle=False) as saved:
        assert_same_model(simulated(sim), {k: saved[k] for k in saved.files})
    expected = _round_bytes(read_events(sim_path / "events.jsonl"))
    assert expected and _round_bytes(read_events(path / "events.jsonl")) == expected
    result = fidelity(
        tmp_path / "runs_distributed", tmp_path / "runs_sim", metrics=["time"]
    )
    functional = result["pairs"][0]["functional"]
    assert result["functional"]["complete"], functional
    assert functional["model"]["bitwise"], functional["model"]


@pytest.mark.skipif(
    not _machines_possible(), reason="no Docker, no ssh or no emulated host image"
)
@pytest.mark.parametrize("transport", ["mqtt", "grpc"])
def test_the_distributed_backend_runs_over_tls_and_keeps_no_secret(
    tmp_path: Path, machines, transport: str
) -> None:
    """MQTTS with a password from a file, and gRPC with mutual TLS (#227).

    Every machine reads its own certificate at the same path; the run folder
    keeps those paths and never what the files hold.
    """
    pytest.importorskip("cryptography")
    from tests.certs import PASSWORD_PORT

    backend = {
        "name": "distributed",
        "inventory": fresh_ports(machines),
        "ssh": DISTRIBUTED["ssh"],
    }
    if transport == "mqtt":
        link = {
            "name": "mqtt",
            "broker": BROKER,
            "tls": {"ca": f"{TLS_ROOT}/ca.pem"},
            "username": MQTT_USER,
            "password_file": f"{TLS_ROOT}/password",
        }
        backend["mqtt_port"] = PASSWORD_PORT
    else:
        pytest.importorskip("grpc")
        link = {
            "name": "grpc",
            "tls": {
                "ca": f"{TLS_ROOT}/ca.pem",
                "cert": f"{TLS_ROOT}/node.pem",
                "key": f"{TLS_ROOT}/node.key",
            },
        }
    base = experiment(tmp_path, link)
    sim = on_the_machines(base, "sim", tmp_path / "runs_sim")
    secure = on_the_machines(base, backend, tmp_path / "runs_distributed")

    (path,) = run_experiment(secure)

    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished", meta.get("processes")
    assert meta["distributed"]["transports"] == [transport]
    with np.load(path / "model.npz", allow_pickle=False) as saved:
        assert_same_model(simulated(sim), {k: saved[k] for k in saved.files})
    assert_no_secrets(path, MQTT_PASSWORD)
    assert f"{TLS_ROOT}/ca.pem" in json.dumps(meta["config"])
