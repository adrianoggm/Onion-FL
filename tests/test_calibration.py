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
