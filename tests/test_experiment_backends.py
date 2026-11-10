"""Tests for the backend contract and the ``sim`` and ``local`` backends (issue
#191, spec I1 §10). Edges use the stub trainer on a CSV format fixture, so
nothing is trained for real (docs/RULES.md)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from onion_fl.core.ids import config_id
from onion_fl.experiment.backends import LocalBackend, SimBackend
from onion_fl.experiment.config import ConfigError, experiment_schema, parse_experiment
from onion_fl.experiment.runner import run_scenario
from onion_fl.experiment.sweep import identity, scenarios
from onion_fl.infrastructure.backends import Backend, backends, execute
from tests.test_experiment import experiment, workspace  # noqa: F401


def test_sim_and_local_are_registered_and_exported() -> None:
    assert {"sim", "local"} <= set(backends.names())
    names = {p["name"] for p in experiment_schema()["plugins"]["backend"]}
    assert {"sim", "local"} <= names
    assert isinstance(SimBackend(), Backend) and isinstance(LocalBackend(), Backend)


def test_the_real_mode_is_an_alias_of_the_local_backend(
    workspace: Path,  # noqa: F811
) -> None:
    default = parse_experiment(experiment(workspace))
    real = parse_experiment(experiment(workspace, runtime={"mode": "real"}))
    local = parse_experiment(experiment(workspace, backend="local"))

    assert default.backend_ref == "sim" and default.simulated
    assert real.backend_ref == "local" and not real.simulated
    assert local.backend_ref == "local" and not local.simulated


def test_an_unset_backend_keeps_the_config_id(workspace: Path) -> None:  # noqa: F811
    config = parse_experiment(experiment(workspace))

    assert "backend" not in identity(config)
    assert config_id(identity(config)) == scenarios(config)[0].config_id


def test_the_real_mode_cannot_name_another_backend(
    workspace: Path,  # noqa: F811
) -> None:
    with pytest.raises(ConfigError, match="alias of backend: local"):
        parse_experiment(experiment(workspace, backend="sim", runtime={"mode": "real"}))


def test_an_unknown_backend_is_refused(workspace: Path) -> None:  # noqa: F811
    with pytest.raises(ConfigError, match="unknown backend plugin"):
        parse_experiment(experiment(workspace, backend="cluster"))


def test_the_local_backend_says_what_it_does_not_support() -> None:
    assert SimBackend().supports({"name": "lab"}) == []
    assert LocalBackend().supports(None) == []
    (reason,) = LocalBackend().supports({"name": "lab"})
    assert "simulated only for now" in reason


def test_a_scenario_runs_through_its_backend_and_records_it(
    workspace: Path,  # noqa: F811
) -> None:
    (scenario,) = scenarios(parse_experiment(experiment(workspace, backend="sim")))

    path = run_scenario(scenario)

    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert meta["backend"] == "sim" and meta["status"] == "finished"


def test_execute_prepares_runs_and_collects_in_order(tmp_path: Path) -> None:
    calls = []

    class Recorder:
        def supports(self, infrastructure):
            return []

        def prepare(self, scenario, infrastructure=None, assignment=None, **options):
            calls.append(("prepare", scenario, options))
            return {"scenario": scenario}

        def run(self, plan):
            calls.append(("run", plan["scenario"], {}))

        def collect(self, plan):
            calls.append(("collect", plan["scenario"], {}))
            return tmp_path

    assert execute(Recorder(), "s", hook=1) == tmp_path
    assert [c[0] for c in calls] == ["prepare", "run", "collect"]
    assert calls[0][2] == {"hook": 1}


def test_execute_refuses_what_a_backend_does_not_support() -> None:
    with pytest.raises(ValueError, match="simulated only"):
        execute(LocalBackend(), "s", {"name": "lab"})


def test_the_local_backend_cannot_take_a_custom_scorer(
    workspace: Path,  # noqa: F811
) -> None:
    (scenario,) = scenarios(parse_experiment(experiment(workspace, backend="local")))

    with pytest.raises(ValueError, match="node processes"):
        run_scenario(scenario, evaluate=lambda model, data: ({}, 0))
