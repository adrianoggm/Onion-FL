"""Tests for calibrations of profiles and links (issue #197, I1 §7, §13).

Protocol tests: the runs use the stub trainer on the CSV format fixture of
``test_experiment``, and the microbenchmark rows are timing records, not data
anything learns from (docs/RULES.md).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from onion_fl.experiment import calibrate
from onion_fl.experiment.cli import parser
from onion_fl.experiment.config import parse_experiment
from onion_fl.experiment.fidelity import fidelity
from onion_fl.experiment.runner import run_scenario
from onion_fl.experiment.sweep import scenarios
from onion_fl.infrastructure.calibration import (
    Calibration,
    Estimate,
    LinkCalibration,
    ProfileCalibration,
    Workload,
    aggregate_rate,
    load_calibration,
    speed,
    train_rate,
    workload_key,
    write_calibration,
)
from onion_fl.infrastructure.emulators import compose_project, layout
from onion_fl.infrastructure.model import InfrastructureError, parse_infrastructure
from onion_fl.infrastructure.paths import effective
from tests.test_experiment import experiment, stub_score, workspace  # noqa: F401
from tests.test_infrastructure import lab

SMALL = workload_key({"w": [4, 8], "b": [8]}, 32)
LARGE = workload_key({"w": [40, 8], "b": [8]}, 32)


def _estimate(value: float, n: int = 10) -> Estimate:
    return Estimate(value=value, low=value * 0.9, high=value * 1.1, n=n)


def _workload(shapes: dict, flops: float) -> Workload:
    return Workload(
        model="m", trainer="t", batch_size=32, shapes=shapes, flops_per_sample=flops
    )


def calibration(**changes: Any) -> tuple[Calibration, bytes]:
    measurements = b'{"kind": "train"}\n'
    base = {
        "backend": "emulated",
        "infrastructure_id": "0" * 64,
        "created_at": "2026-11-06T00:00:00+00:00",
        "code": {"commit": "abc", "dirty": False},
        "environment": {},
        "spec": {},
        "measurements_sha256": hashlib.sha256(measurements).hexdigest(),
        "workloads": {
            SMALL[0]: _workload({"w": [4, 8], "b": [8]}, SMALL[1]),
        },
        "profiles": {
            "pi": ProfileCalibration(train={SMALL[0]: _estimate(400.0)}),
            "vm": ProfileCalibration(aggregate=_estimate(1e6)),
        },
        "links": {
            "fog|dc": LinkCalibration(
                latency=_estimate(0.08),
                bandwidth_up=_estimate(9e6),
                bandwidth_down=_estimate(9e6),
                loss=_estimate(0.0),
            )
        },
    }
    return Calibration(**(base | changes)), measurements


def cited_lab(cid: str, **profiles: Any) -> dict:
    raw = lab()
    raw["profiles"] = {"vm": {"calibration": cid}, "pi": {"calibration": cid}}
    raw["profiles"] |= profiles
    raw["network"]["links"][1]["calibration"] = cid
    return raw


def stored(tmp_path: Path, **changes: Any) -> tuple[Path, str]:
    folder = write_calibration(tmp_path / "calibrations", *calibration(**changes))
    return folder.parent, folder.name


# --- the artifact --------------------------------------------------------------------


def test_a_calibration_is_signed_by_its_content(tmp_path: Path) -> None:
    folder, cid = stored(tmp_path)
    assert load_calibration(folder, cid).calibration_id == cid
    path = folder / cid / "calibration.json"
    document = json.loads(path.read_text("utf-8"))
    document["profiles"]["pi"]["train"][SMALL[0]]["value"] = 4000.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(InfrastructureError, match="does not match"):
        load_calibration(folder, cid)


def test_a_checkout_with_crlf_line_ends_keeps_the_signature(tmp_path: Path) -> None:
    folder, cid = stored(tmp_path)
    path = folder / cid / "measurements.jsonl"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert load_calibration(folder, cid).calibration_id == cid


def test_changed_measurements_break_the_signature(tmp_path: Path) -> None:
    folder, cid = stored(tmp_path)
    (folder / cid / "measurements.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(InfrastructureError, match="measurements changed"):
        load_calibration(folder, cid)


# --- profiles and links that cite one ------------------------------------------------


def test_a_profile_takes_its_rates_from_the_calibration(tmp_path: Path) -> None:
    folder, cid = stored(tmp_path)
    infra = parse_infrastructure(cited_lab(cid), folder)
    assert train_rate(infra, "field-pi-1", SMALL) == (400.0, None)
    assert aggregate_rate(infra, "fog-1") == 1e6
    assert speed(infra, "field-pi-1") == pytest.approx(400.0 * SMALL[1])
    assert infra.infrastructure_id != parse_infrastructure(lab()).infrastructure_id


def test_an_unmeasured_model_is_estimated_by_flops_and_says_so(tmp_path: Path) -> None:
    folder, cid = stored(tmp_path)
    infra = parse_infrastructure(cited_lab(cid), folder)
    rate, note = train_rate(infra, "field-pi-2", LARGE)
    assert rate == pytest.approx(400.0 * SMALL[1] / LARGE[1])
    assert note["from_workload"] == SMALL[0] and note["workload"] == LARGE[0]
    assert note["host"] == "field-pi-2" and note["calibration"] == cid


def test_rates_and_a_calibration_together_are_refused() -> None:
    raw = cited_lab("a" * 64, pi={"train_rate": 10, "calibration": "a" * 64})
    with pytest.raises(InfrastructureError, match="come from the calibration"):
        parse_infrastructure(raw)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"pi": {"calibration": None, "cpus": 2}}, "measured it with None"),
        ({"other": {}}, "did not measure it"),
    ],
)
def test_a_calibration_that_does_not_fit_the_profile_is_refused(
    tmp_path: Path, change: dict, message: str
) -> None:
    folder, cid = stored(tmp_path)
    raw = cited_lab(cid)
    if "other" in change:  # a profile the calibration never measured
        raw["profiles"]["other"] = {"calibration": cid}
        raw["sites"]["field"]["pool"]["other"] = 1
    else:
        raw["profiles"]["pi"] = {"cpus": 2, "calibration": cid}
    with pytest.raises(InfrastructureError, match=message):
        parse_infrastructure(raw, folder)


def test_a_calibration_not_loaded_is_said(tmp_path: Path) -> None:
    _, cid = stored(tmp_path)
    infra = parse_infrastructure(cited_lab(cid))  # no folder given
    with pytest.raises(InfrastructureError, match="not loaded"):
        train_rate(infra, "field-pi-1", SMALL)


def test_the_simulation_uses_the_calibrated_link_and_the_emulator_the_nominal(
    tmp_path: Path,
) -> None:
    folder, cid = stored(tmp_path)
    infra = parse_infrastructure(cited_lab(cid), folder)
    _, profile = effective(infra, "fog-1", "cloud-1")
    assert profile["latency_s"] == 0.08 and profile["bandwidth_up_bps"] == 9e6
    nominal = parse_infrastructure(lab())
    _, before = effective(nominal, "field-pi-1", "fog-1")
    _, after = effective(infra, "field-pi-1", "fog-1")
    assert before == after  # field|fog cites no calibration
    emulation = compose_project(
        layout(infra, ["fog-1", "cloud-1"]), project="p", image="i"
    )
    shaped = " ".join(c for cs in emulation.shaping.values() for c in cs)
    assert "delay 70000us" in shaped and "rate 10000000bit" in shaped


# --- runs ----------------------------------------------------------------------------


def _calibrated_experiment(workspace: Path, cid: str, folder: Path) -> Any:  # noqa: F811
    raw = experiment(
        workspace,
        infrastructure={
            "name": cited_lab(cid),
            "assignment": {
                "root": "cloud-1",
                "aggregators": {"fog_a": "fog-1", "fog_b": "fog-2"},
                "edges": {"name": "explicit", "hosts": {"*": "field-pi-1"}},
            },
        },
    )
    raw["paths"]["calibrations"] = str(folder)
    return parse_experiment(raw)


def test_a_run_with_estimated_rates_is_marked_and_refused_by_fidelity(
    tmp_path: Path,
    workspace: Path,  # noqa: F811
) -> None:
    folder, cid = stored(tmp_path)
    (scenario,) = scenarios(_calibrated_experiment(workspace, cid, folder))
    path = run_scenario(scenario, evaluate=stub_score)
    meta = json.loads((path / "run.json").read_text("utf-8"))
    assert meta["status"] == "finished" and meta["calibrations"] == [cid]
    assert meta["estimated"] and {e["profile"] for e in meta["estimated"]} == {"pi"}
    with pytest.raises(ValueError, match="estimated by FLOPs"):
        fidelity(path.parent, path.parent)

    # Measuring the edges' workload removes the estimate.
    wid = meta["estimated"][0]["workload"]
    flops = SMALL[1] * 400.0 / meta["estimated"][0]["train_rate"]
    measured = calibration(
        workloads={wid: _workload({"w": [1]}, flops)},
        profiles={
            "pi": ProfileCalibration(train={wid: _estimate(400.0)}),
            "vm": ProfileCalibration(aggregate=_estimate(1e6)),
        },
    )
    again = write_calibration(tmp_path / "calibrations", *measured)
    (scenario,) = scenarios(_calibrated_experiment(workspace, again.name, again.parent))
    meta = json.loads(
        (run_scenario(scenario, evaluate=stub_score) / "run.json").read_text("utf-8")
    )
    assert "estimated" not in meta and meta["calibrations"] == [again.name]


def test_without_calibrations_the_paths_and_ids_do_not_change(
    workspace: Path,  # noqa: F811
) -> None:
    assert "calibrations" not in parse_experiment(experiment(workspace)).dump()["paths"]


# --- fitting -------------------------------------------------------------------------


def _train(host: str, job: int, t0: float, wall: float, samples: int = 600) -> dict:
    return {
        "kind": "train",
        "workload": SMALL[0],
        "host": host,
        "profile": "pi",
        "job": job,
        "samples": samples,
        "wall_s": wall,
        "t0": t0,
        "t1": t0 + wall,
    }


def test_only_concurrent_trainings_after_the_warm_up_are_kept() -> None:
    rows = [_train("a", j, 10.0 + j, 1.0) for j in range(5)]  # 10..15
    rows += [_train("b", j, 12.0 + j, 1.0) for j in range(5)]  # 12..17
    kept = calibrate.concurrent(rows, "job")
    # The window is [12, 15], when both train; b's warm-up (job 0) goes too.
    assert sorted((r["host"], r["job"]) for r in kept) == [
        ("a", 2),
        ("a", 3),
        ("a", 4),
        ("b", 1),
        ("b", 2),
    ]


def test_a_rate_is_total_samples_over_total_time_with_an_interval() -> None:
    estimate = calibrate.ratio([600, 600, 300], [1.0, 1.5, 0.5])
    assert estimate.value == pytest.approx(500.0)
    assert estimate.low <= 500.0 <= estimate.high and estimate.n == 3


def _messages(src: str, dst: str, latency: float, bps: float) -> list[dict]:
    rows = []
    for seq, size in enumerate([100, 1000, 10_000, 100_000] * 3):
        rows.append({"kind": "sent", "src": src, "dst": dst, "seq": seq, "size": size})
        delay = latency + 8 * size / bps + (seq % 3 - 1) * 1e-4
        rows.append(
            {
                "kind": "link",
                "src": src,
                "dst": dst,
                "seq": seq,
                "t_send": 100.0,
                "t_recv": 100.0 + delay,
            }
        )
    return rows


def test_link_latencies_are_solved_from_the_paths_that_cross_them() -> None:
    """Paths through the broker on cloud-1: fog -> cloud crosses fog|dc once and
    field -> fog crosses field|fog once and fog|dc twice."""
    infra = parse_infrastructure(lab(services={"mqtt_broker": "cloud-1"}))
    rows = _messages("fog-1", "cloud-1", 0.071, 9.5e6)
    rows += _messages("field-pi-1", "fog-1", 0.011 + 2 * 0.071, 9.5e6 / 2)
    rows += _messages("cloud-1", "cloud-1", 0.0005, 1e9)
    links, transport, notes = calibrate._links(
        infra, "cloud-1", calibrate.link_rows(rows), seed=0
    )
    assert links["fog|dc"].latency.value == pytest.approx(0.071, abs=2e-4)
    assert links["field|fog"].latency.value == pytest.approx(0.011, abs=4e-4)
    assert links["fog|dc"].bandwidth_up.value == pytest.approx(
        1e7 * (0.95 + 0.475) / 2, rel=0.02
    )  # the bottleneck of both paths: the mean ratio
    assert links["field|fog"].derived == ["bandwidth_up", "bandwidth_down"]
    assert links["fog|dc"].loss.value == 0.0
    assert transport["loopback_latency_s"].value == pytest.approx(5e-4, abs=2e-4)
    assert notes


def test_the_cli_has_the_three_steps() -> None:
    p = parser()
    assert p.parse_args(["calibrate", "measure", "s.yaml", "--out", "x"]).step == (
        "measure"
    )
    assert p.parse_args(["calibrate", "fit", "x"]).out == "calibrations"
    args = p.parse_args(["calibrate", "check", "x", "--infrastructure", "i.yaml"])
    assert args.calibrations == "calibrations"


def test_the_spec_of_the_emulated_lab_loads() -> None:
    spec = calibrate.load_spec("calibrations/emulated_lab.yaml")
    assert spec.infrastructure == "emulated_lab" and len(spec.workloads) == 2


def _session(folder: Path, infra_path: Path, jitter: float) -> Path:
    """A session as ``measure`` writes it, with timing rows for two devices,
    one fog and two paths."""
    import yaml

    rows = []
    for h, host in enumerate(["field-pi-1", "field-pi-2"]):
        for j in range(8):
            wall = 1.5 + jitter * ((j + h) % 3 - 1)
            rows.append(_train(host, j, 10.0 + 1.5 * j + 0.1 * h, wall))
    for r in range(8):
        rows.append(
            {
                "kind": "aggregate",
                "host": "fog-1",
                "profile": "vm",
                "updates": 10,
                "repeat": r,
                "bytes": 1e5,
                "wall_s": 0.01 + jitter * 0.001 * (r % 2),
                "t0": 50.0 + 0.01 * r,
                "t1": 50.01 + 0.01 * r,
            }
        )
    rows += _messages("fog-1", "cloud-1", 0.071, 9.5e6)
    rows += _messages("field-pi-1", "fog-1", 0.011 + 2 * 0.071, 9.5e6 / 2)
    infra = parse_infrastructure(yaml.safe_load(infra_path.read_text("utf-8")))
    small = _workload({"w": [4, 8], "b": [8]}, SMALL[1])
    meta = {
        "spec": {
            "name": "lab",
            "infrastructure": str(infra_path),
            "data": {"datasets": {"demo": {}}},
            "workloads": [{"model": "m", "trainer": "t"}],
        },
        "infrastructure_id": infra.infrastructure_id,
        "hosts": {"field-pi-1": "pi", "field-pi-2": "pi", "fog-1": "vm"},
        "broker": "cloud-1",
        "workloads": {SMALL[0]: small.model_dump()},
        "clients": {},
        "environment": {},
        "code": {"commit": "abc", "dirty": False},
        "started_at": "2026-11-06T00:00:00+00:00",
    }
    folder.mkdir(parents=True)
    (folder / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    (folder / "measurements.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return folder


def test_a_calibration_is_fitted_on_one_session_and_checked_on_another(
    tmp_path: Path,
) -> None:
    import yaml

    infra_path = tmp_path / "lab.yaml"
    infra_path.write_text(
        yaml.safe_dump(lab(services={"mqtt_broker": "cloud-1"})), encoding="utf-8"
    )
    fit_on = _session(tmp_path / "fit", infra_path, jitter=0.1)
    folder = calibrate.fit(fit_on, tmp_path / "calibrations")
    measured = load_calibration(folder.parent, folder.name)
    rate = measured.profiles["pi"].train[SMALL[0]]
    assert rate.value == pytest.approx(600 / 1.5, rel=0.02) and rate.covers(400)
    calibrated = tmp_path / "calibrated.yaml"
    calibrated.write_text(
        yaml.safe_dump(calibrate.calibrated_infrastructure(infra_path, measured)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fitted on"):
        calibrate.check(fit_on, calibrated, folder.parent)

    other = _session(tmp_path / "other", infra_path, jitter=0.12)
    result = calibrate.check(other, calibrated, folder.parent)
    assert result["calibrations"] == [folder.name]
    (train,) = result["train"]
    assert train["inside"] and not train["estimated"]
    assert result["aggregate"][0]["inside"]
    assert {(p["src"], p["dst"]) for p in result["paths"]} == {
        ("fog-1", "cloud-1"),
        ("field-pi-1", "fog-1"),
    }


def test_pinning_gives_each_host_its_own_cpus_and_the_rest_one_set(
    tmp_path: Path,
) -> None:
    infra = parse_infrastructure(
        lab(
            profiles={"vm": {"cpus": 2, "train_rate": 100}, "pi": {"cpus": 1}},
            services={"mqtt_broker": "cloud-1"},
        )
    )
    lay = layout(infra, ["field-pi-1", "fog-1"], "cloud-1")
    emulation = compose_project(lay, project="p", image="i")
    spec = calibrate.PinSpec(cpus=[2, 4, 6, 8, 10], others="20-23")
    sets = calibrate.pin(emulation, spec)
    services = emulation.compose["services"]
    # in name order: cloud-1 (2 CPUs), field-pi-1 (1), fog-1 (2)
    assert [services[lay.services[h]]["cpuset"] for h in sorted(lay.hosts)] == [
        "2,4",
        "6",
        "8,10",
    ]
    assert services["broker"]["cpuset"] == "20-23" == sets["router-dc"]
    with pytest.raises(ValueError, match="do not cover"):
        calibrate.pin(emulation, calibrate.PinSpec(cpus=[2, 4], others="20"))

    calibrate.broker_nodelay(emulation, tmp_path)
    assert "set_tcp_nodelay true" in (tmp_path / "mosquitto.conf").read_text("utf-8")
    assert services["broker"]["command"][-1].endswith("/mosquitto.conf")
