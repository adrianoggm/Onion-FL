"""The triggers move from ``continuum:`` to ``continual:`` (M1, issue #193).

The old form is read for one version, with one deprecation warning per config,
and every config_id stays: the committed configs and results of C2-C8 load
unchanged and give the config_ids their runs were signed with. The shipped
experiments are in the new form since #309; ``old_form`` writes them back.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from onion_fl.experiment.config import (
    PACE,
    ConfigError,
    experiment_schema,
    from_continuum,
    parse_experiment,
)
from onion_fl.experiment.sweep import identity, scenarios

ROOT = Path(__file__).resolve().parents[1]
TRIGGERS = ROOT / "experiments" / "stream_triggers_swell_wesad.yaml"


def committed() -> dict[str, set[tuple[str, str]]]:
    """(scenario, config_id) of every committed run, by experiment."""
    runs = defaultdict(set)
    for meta in _metas():
        runs[meta["config"]["name"]].add((meta["scenario"], meta["config_id"]))
    return dict(runs)


def _metas() -> list[dict]:
    out = []
    for path in sorted((ROOT / "results").glob("*/*/run.json")):
        if path.parts[-3] != "legacy":
            out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def recorded(name: str) -> dict[tuple[str, str], dict]:
    """The config each committed run of ``name`` recorded, by (scenario, id)."""
    return {
        (m["scenario"], m["config_id"]): m["config"]
        for m in _metas()
        if m["config"]["name"] == name
    }


def new_form(raw: dict) -> dict:
    """A config written after M1: triggers in continual, sweeps on continual."""
    out, _ = from_continuum(raw)

    def path(p: str) -> str:
        return (
            "continual" + p.removeprefix("continuum")
            if p.startswith("continuum")
            else p
        )

    if "sweep" in out:
        out["sweep"] = {
            ",".join(path(p.strip()) for p in key.split(",")): values
            for key, values in out["sweep"].items()
        }
    return out


def old_form(raw: dict) -> dict:
    """The same config as written before M1: the triggers under ``continuum``."""
    out = dict(raw)
    continual = dict(out.pop("continual", {}))
    pace = {k: continual.pop(k) for k in PACE if k in continual}
    if continual:
        out["continual"] = continual
    if pace:
        out["continuum"] = pace

    def path(p: str) -> str:
        return "continuum." + p[10:] if p[10:] in PACE else p

    if "sweep" in out:
        out["sweep"] = {
            ", ".join(path(p.strip()) for p in key.split(",")): values
            for key, values in out["sweep"].items()
        }
    return out


def old_triggers() -> dict:
    return old_form(yaml.safe_load(TRIGGERS.read_text("utf-8")))


def quiet(raw: dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return parse_experiment(raw)


RUNS = committed()


def test_the_committed_continual_results_are_all_checked() -> None:
    names = {"stream_triggers_swell_wesad", "continuum_benchmark_sweeps"}
    assert names <= RUNS.keys()  # the old continuum section, and sweeps on it


@pytest.mark.parametrize("name", sorted(RUNS))
def test_a_committed_config_keeps_its_config_ids_in_either_form(name: str) -> None:
    shipped = yaml.safe_load((ROOT / "experiments" / f"{name}.yaml").read_text("utf-8"))
    raw = old_form(shipped)
    ids = {s.config_id for s in scenarios(quiet(raw))}
    rewritten = {s.config_id for s in scenarios(quiet(new_form(raw)))}
    assert {s.config_id for s in scenarios(parse_experiment(shipped))} == ids
    # Runs are matched by config_id, not by scenario name: moving a swept path
    # from continuum to continual (#309) renamed the scenarios, not their ids.
    # Runs a script made from the experiment with changes (C9's backends and
    # failures) are checked on the config they recorded instead.
    for (scenario, cid), config in recorded(name).items():
        if cid not in ids:
            for form in (config, new_form(config)):
                again = {
                    s.config_id for s in scenarios(quiet(form)) if s.name == scenario
                }
                assert cid in again, (scenario, cid)
            ids.add(cid)
            rewritten.add(cid)

    assert {cid for _, cid in RUNS[name]} <= ids
    assert {cid for _, cid in RUNS[name]} <= rewritten


def test_the_old_form_warns_once_per_config() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = parse_experiment(old_triggers())
        assert [w.category for w in caught] == [DeprecationWarning]
        assert "continual.trigger" in str(caught[0].message)

        (one, *_) = scenarios(config)  # nothing reads the old form again
        parse_experiment(config.dump())
        parse_experiment(identity(one.config), signed=True)  # a run.json's config
        parse_experiment(new_form(old_triggers()))
    assert len(caught) == 1


def test_the_config_is_written_in_the_new_form_and_signed_in_the_old() -> None:
    raw = old_triggers()
    config = quiet(raw)
    dumped = config.dump()

    assert "continuum" not in dumped
    assert dumped["continual"] == {
        "trigger": {"name": "schedule", "every": "10m"},
        "status_every": 600.0,
    }
    signed = identity(config)
    assert "continual" not in signed  # it had no replay part
    assert signed["continuum"] == raw["continuum"] | {"status_every": 600.0}


def test_an_empty_old_continual_section_stays_in_the_signature() -> None:
    raw = old_triggers()
    old = identity(quiet(raw | {"continual": {}}))
    new = identity(quiet(new_form(raw | {"continual": {}})))

    assert old == new
    assert old["continual"] == {"memory": "none", "replay_ratio": 0.25}


@pytest.mark.parametrize(
    "change, message",
    [
        (
            {"continual": {"trigger": {"name": "schedule", "every": 600}}},
            "continual.trigger is given too",
        ),
        ({"continuum": "schedule"}, "mapping"),
    ],
)
def test_the_two_forms_cannot_be_mixed(change: dict, message: str) -> None:
    raw = old_triggers() | change
    with pytest.raises(ConfigError, match=message):
        quiet(raw)


def test_a_local_trigger_needs_a_federated_one() -> None:
    raw = new_form(old_triggers())
    raw["continual"] = {"edge_trigger": {"name": "volume", "samples": 8}}
    with pytest.raises(ConfigError, match="continual.edge_trigger"):
        parse_experiment(raw)


def test_a_swept_continual_value_replaces_the_parts_it_names() -> None:
    raw = new_form(old_triggers()) | {"seeds": [0]}
    volume = {"trigger": {"name": "volume", "samples": 300}, "status_every": "5m"}
    raw["sweep"] = {
        "continual": [{"memory": "none"}, volume, volume | {"memory": "none"}]
    }
    memory, trigger, both = (
        s.config.continual for s in scenarios(parse_experiment(raw))
    )

    assert memory.trigger == {"name": "schedule", "every": "10m"} and memory.replays
    assert trigger.trigger == volume["trigger"] and not trigger.replays
    assert both.trigger == volume["trigger"] and both.replays


def test_the_schema_the_studio_reads_has_the_new_form_only() -> None:
    schema = experiment_schema()
    continual = schema["$defs"]["ContinualConfig"]["properties"]

    assert "continuum" not in schema["properties"]
    assert {"trigger", "edge_trigger", "status_every", "memory"} <= continual.keys()
