"""Tests for FedScale's device traces as profiles (issue #253).

The traces are FedScale's pickles; these tests write tiny ones in its format
(format fixtures, docs/RULES.md) and run the protocol with the stub trainer. The
test on the real traces skips without ``data/FedScale/manifest.json``.
"""

from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from onion_fl.experiment.config import ConfigError, parse_experiment
from onion_fl.experiment.runner import plan, run_scenario
from onion_fl.experiment.sweep import scenarios
from onion_fl.infrastructure.fedscale import (
    TRAIN_FACTOR,
    expand,
    import_fedscale,
    safe_load,
)
from onion_fl.infrastructure.model import (
    InfrastructureError,
    load_infrastructure,
    parse_infrastructure,
)
from onion_fl.infrastructure.paths import effective, hops
from onion_fl.runtime.devices import availability_models
from tests.test_experiment import (  # noqa: F401
    TOPOLOGY,
    experiment,
    stub_score,
    workspace,
)
from tests.test_infrastructure import lab

REAL = Path("data")

# --- reading the pickles safely ---------------------------------------------------


class _Evil:
    def __reduce__(self):
        return (print, ("pwned",))


@pytest.mark.parametrize("protocol", [3, 4, 5])  # FedScale used 3 and 4
def test_the_unpickler_reads_plain_values_and_numpy_floats(protocol: int) -> None:
    value = OrderedDict(
        {1: {"computation": np.float64(39.0), "communication": np.float32(2.5)}}
    )
    value[2] = {"active": [1, 2], "inactive": [3, 4], "model": "SM-A105M"}

    loaded = safe_load(pickle.dumps(value, protocol=protocol))

    assert loaded == {
        1: {"computation": 39.0, "communication": 2.5},
        2: {"active": [1, 2], "inactive": [3, 4], "model": "SM-A105M"},
    }
    assert type(loaded[1]["computation"]) is float


@pytest.mark.parametrize(
    "payload", [_Evil(), np.zeros(3), np.dtype("O"), {1: Path("x")}]
)
def test_the_unpickler_refuses_anything_else(payload: object) -> None:
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        safe_load(pickle.dumps(payload, protocol=3))


# --- importing and sampling ---------------------------------------------------------


def fedscale_folder(root: Path, clients: int = 10) -> Path:
    """FedScale's two pickles for ``clients`` clients, imported."""
    folder = root / "FedScale"
    folder.mkdir(parents=True)
    capacity = {
        c: {"computation": np.float64(10.0 * c), "communication": 1000.0 + c}
        for c in range(1, clients + 1)
    }
    behaviour = OrderedDict(
        (
            c,
            {
                "duration": 900,
                "active": [0, 20 + c],
                "inactive": [1, 1000 + c],
                "finish_time": 2000 + c,
                "model": f"phone-{c}",
            },
        )
        for c in range(1, clients + 1)
    )
    (folder / "client_device_capacity").write_bytes(pickle.dumps(capacity, 3))
    (folder / "client_behave_trace").write_bytes(pickle.dumps(behaviour, 4))
    import_fedscale(folder)
    return folder


def traced(devices: int = 4, seed: int = 0, **pool) -> dict:
    return lab(
        sites={
            "dc": {"hosts": {"cloud-1": "vm"}},
            "fog": {"hosts": {"fog-1": "vm", "fog-2": "vm"}},
            "field": {"traces": {"devices": devices, "seed": seed, **pool}},
        }
    )


def test_import_writes_plain_files_with_the_hashes_and_units(tmp_path: Path) -> None:
    folder = fedscale_folder(tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))

    assert manifest["license"] == "Apache-2.0"
    assert set(manifest["sources"]) == {"client_device_capacity", "client_behave_trace"}
    assert manifest["clients"] == {
        "client_device_capacity": 10,
        "client_behave_trace": 10,
    }
    assert (folder / "capacity.csv").read_text("utf-8").splitlines()[
        1
    ] == "1,10.0,1001.0"
    assert manifest["units"]["computation"].startswith("ms per sample")


def test_the_same_seed_gives_the_same_profiles(tmp_path: Path) -> None:
    fedscale_folder(tmp_path)
    one = parse_infrastructure(expand(traced(seed=3), tmp_path))
    again = parse_infrastructure(expand(traced(seed=3), tmp_path))
    other = parse_infrastructure(expand(traced(seed=4), tmp_path))

    assert one.profiles == again.profiles
    assert one.device_traces == again.device_traces
    assert one.infrastructure_id == again.infrastructure_id
    assert one.device_traces != other.device_traces
    assert one.infrastructure_id != other.infrastructure_id
    clients = [c["capacity_client"] for c in one.device_traces["field"]["clients"]]
    assert len(set(clients)) == 4  # without replacement


def test_profiles_map_fedscale_units_and_say_they_come_from_a_trace(
    tmp_path: Path,
) -> None:
    fedscale_folder(tmp_path)
    infra = parse_infrastructure(expand(traced(cpus=1), tmp_path))
    record = infra.device_traces["field"]

    assert record["rates"] == "trace" and record["seed"] == 0
    for client in record["clients"]:
        c = client["capacity_client"]
        profile = infra.profiles[client["profile"]]
        assert profile.origin == "trace" and profile.cpus == 1
        assert profile.train_rate == pytest.approx(1000 / (TRAIN_FACTOR * 10.0 * c))
        assert profile.bandwidth == pytest.approx(1000 * (1000.0 + c))
        a = client["availability_client"]
        assert profile.availability == {
            "name": "windows",
            "active": [0.0, 20.0 + a],
            "inactive": [1.0, 1000.0 + a],
            "period": 2000.0 + a,
        }
    hosts = [h for h, (site, _) in infra.hosts().items() if site == "field"]
    assert len(hosts) == 4 and "field-fedscale_field_1-1" in hosts


def test_without_availability_the_devices_are_always_up(tmp_path: Path) -> None:
    fedscale_folder(tmp_path)
    infra = parse_infrastructure(expand(traced(availability=False), tmp_path))
    assert all(p.availability is None for p in infra.profiles.values())
    assert all(
        "availability_client" not in c for c in infra.device_traces["field"]["clients"]
    )


def test_sampling_refuses_more_devices_than_clients_and_missing_traces(
    tmp_path: Path,
) -> None:
    with pytest.raises(InfrastructureError, match="not found"):
        expand(traced(), tmp_path)
    fedscale_folder(tmp_path)
    with pytest.raises(InfrastructureError, match="11 devices"):
        expand(traced(devices=11), tmp_path)
    with pytest.raises(InfrastructureError, match="not sampled"):
        parse_infrastructure(traced())


def test_a_changed_trace_file_is_refused(tmp_path: Path) -> None:
    folder = fedscale_folder(tmp_path)
    with (folder / "capacity.csv").open("a", encoding="utf-8") as f:
        f.write("99,1.0,1.0\n")
    with pytest.raises(InfrastructureError, match="does not match its manifest"):
        expand(traced(), tmp_path)


def test_new_fields_leave_existing_infrastructure_ids_alone() -> None:
    infra = parse_infrastructure(lab())
    dumped = infra.model_dump(mode="json")
    assert "device_traces" not in dumped
    assert all(
        set(p) == {"train_rate", "aggregate_rate", "cpus", "memory"}
        for p in dumped["profiles"].values()
    )
    assert all("traces" not in s for s in dumped["sites"].values())


# --- what the network and the runtime see -------------------------------------------


def test_a_device_bandwidth_is_its_own_access_segment(tmp_path: Path) -> None:
    infra = parse_infrastructure(
        lab(profiles={"vm": {"train_rate": 100}, "pi": {"bandwidth": "1Mbps"}})
    )
    _, crossed = hops(infra, "field-pi-1", "fog-1")
    assert crossed == [
        ("host:field-pi-1", True),
        ("link:field|fog", True),
    ]
    _, back = hops(infra, "fog-1", "field-pi-2")
    assert back[-1] == ("host:field-pi-2", False)
    _, profile = effective(infra, "field-pi-1", "fog-1")
    assert profile["bandwidth_up_bps"] == 1e6  # narrower than the 100Mbps link
    _, local = hops(infra, "field-pi-1", "field-pi-2")
    assert [k for k, _ in local] == ["host:field-pi-1", "lan:field", "host:field-pi-2"]


def test_windows_repeat_with_their_period() -> None:
    model = availability_models.create(
        "windows", {"active": [10, 40], "inactive": [20, 50], "period": 100}
    )
    up = [t for t in range(0, 200, 5) if model.is_up(t, None, "e", 0)]
    assert up == [10, 15, 20, 40, 45, 50, 110, 115, 120, 140, 145, 150]
    assert model.up_again(25) == 40 and model.up_again(60) == 110
    assert model.up_again(15) == 15
    once = availability_models.create("windows", {"active": [10], "inactive": [20]})
    assert once.up_again(5) == 10 and once.up_again(30) is None
    with pytest.raises(ValueError, match="ordered"):
        availability_models.create("windows", {"active": [10], "inactive": [5]})


def with_traces(workspace: Path, **pool) -> dict:  # noqa: F811
    fedscale_folder(workspace / "traces", clients=20)
    fogs = {"defaults": {"deadline": "60s", "quorum": 1}} | TOPOLOGY["fog"]
    return experiment(
        workspace,
        topology=TOPOLOGY | {"fog": fogs},  # offline edges miss rounds
        infrastructure={
            "name": traced(devices=12, seed=1, **pool),
            "assignment": {
                "root": "cloud-1",
                "aggregators": {"fog_a": "fog-1", "fog_b": "fog-2"},
                "edges": {"name": "one_per_device", "seed": 0},
            },
        },
    ) | {
        "paths": experiment(workspace)["paths"]
        | {"device_traces": str(workspace / "traces")}
    }


def test_a_run_on_fedscale_devices_cites_the_traces_in_run_json(
    workspace: Path,  # noqa: F811
) -> None:
    config = parse_experiment(with_traces(workspace))
    (preview,) = plan(config)
    assert preview["infrastructure"]["device_traces"]["field"]["devices"] == 12

    (scenario,) = scenarios(config)
    folder = run_scenario(scenario, evaluate=stub_score)
    meta = json.loads((folder / "run.json").read_text("utf-8"))
    traces = meta["device_traces"]["field"]
    assert meta["status"] == "finished"
    assert traces["rates"] == "trace" and traces["seed"] == 1
    manifest = json.loads(
        (workspace / "traces" / "FedScale" / "manifest.json").read_text("utf-8")
    )
    assert traces["files"] == manifest["files"]
    assert traces["sources"] == manifest["sources"]
    events = [
        json.loads(line)
        for line in (folder / "events.jsonl").read_text("utf-8").splitlines()
    ]
    assert any(e["name"] == "message.dropped_offline" for e in events)


def test_a_topology_availability_clashes_with_a_traced_one(
    workspace: Path,  # noqa: F811
) -> None:
    raw = with_traces(workspace)
    raw["topology"] = raw["topology"] | {
        "edge": {"availability": {"name": "bernoulli", "p": 0.1}}
    }
    (scenario,) = scenarios(parse_experiment(raw))
    with pytest.raises(ConfigError, match="its own availability"):
        run_scenario(scenario, evaluate=stub_score)


# --- the real traces ---------------------------------------------------------------------


@pytest.mark.skipif(
    not (REAL / "FedScale" / "manifest.json").exists(),
    reason="needs FedScale's traces imported in data/FedScale (README §2)",
)
def test_the_real_traces_give_the_same_profiles_for_the_same_seed() -> None:
    one = load_infrastructure("infrastructures/fedscale_field.yaml", REAL)
    again = load_infrastructure("infrastructures/fedscale_field.yaml", REAL)
    record = one.device_traces["field"]
    assert one.infrastructure_id == again.infrastructure_id
    assert one.profiles == again.profiles
    assert record["license"] == "Apache-2.0" and record["devices"] == 160
    rates = [p.train_rate for p in one.profiles.values() if p.origin == "trace"]
    assert len(rates) == 160 and min(rates) > 0
