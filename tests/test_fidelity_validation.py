"""The validation of issue #198 fixes its setup before any run.

What its pre-registration (``results/fidelity_sim_emulated/PREREGISTRATION.md``)
states is checked against the files: the validation infrastructure is the
calibrated one with the calibration's five devices, both cite the same single
calibration, the emulated backend pins each host to the CPUs of the
calibration's sessions, and the edges train the workload the calibration
measured. Nothing here trains or needs data.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from onion_fl.experiment.calibrate import load_spec
from onion_fl.experiment.config import load_experiment
from onion_fl.infrastructure.calibration import cited
from onion_fl.infrastructure.model import load_infrastructure

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "fidelity_sim_emulated.yaml"
VALIDATION = ROOT / "infrastructures" / "fidelity_validation_lab.yaml"
CALIBRATED = ROOT / "infrastructures" / "emulated_lab_calibrated.yaml"
SPEC = ROOT / "calibrations" / "emulated_lab.yaml"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_validation_lab_is_the_calibrated_one_on_the_measured_devices() -> None:
    validation, calibrated = _yaml(VALIDATION), _yaml(CALIBRATED)
    assert validation["sites"]["field"] == {"pool": {"edge_device": 5}}
    for raw in (validation, calibrated):
        del raw["name"], raw["sites"]["field"]
    assert validation == calibrated  # profiles, links and their citations
    ids = cited(load_infrastructure(VALIDATION))
    assert len(ids) == 1 and ids == cited(load_infrastructure(CALIBRATED))


def test_the_runs_use_the_hosts_and_cpus_the_calibration_measured() -> None:
    spec = load_spec(SPEC)
    measured = set(spec.train.hosts) | set(spec.aggregate.hosts)
    measured |= {h for pair in spec.links.pairs for h in pair}
    infra = load_infrastructure(VALIDATION)
    assert set(infra.hosts()) == measured
    # the calibration gives its CPUs to its hosts in name order
    free, expected = list(spec.pin.cpus), {}
    for host in sorted(measured):
        need = infra.profile_of(host).cpus or 1
        expected[host], free = free[:need], free[need:]
    pin = _yaml(EXPERIMENT)["backend"]["pin"]
    assert pin == {"cpus": expected, "others": spec.pin.others}


def test_the_edges_train_a_workload_the_calibration_measured() -> None:
    spec = load_spec(SPEC)
    config = load_experiment(EXPERIMENT)
    learning = config.dump()["learning"]
    workloads = [w.model_dump(mode="json") for w in spec.workloads]
    model = learning["model"]
    assert any(w["model"] == model for w in workloads)
    (calibrated,) = [w for w in workloads if w["model"] == model]
    assert learning["trainer"]["batch_size"] == calibrated["trainer"]["batch_size"]
    assert (
        config.dump()["data"]["datasets"]
        == spec.data.model_dump(mode="json")["datasets"]
    )
    assert config.infrastructure.name == VALIDATION.stem
