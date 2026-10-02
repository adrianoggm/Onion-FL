"""Tests for the diagnostics computed at each aggregator (issue #94).

Deltas and metrics are hand-picked numbers; the federation uses the stub
trainer. Nothing is trained (docs/RULES.md).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from onion_fl.core.topology import parse_topology
from onion_fl.learning.aggregators import Contribution
from onion_fl.learning.model import (
    DataShape,
    ModularMLP,
    ModularMLPConfig,
    state_arrays,
)
from onion_fl.learning.trainers import trainers
from onion_fl.observability.diagnostics import RoundView, diagnostics
from onion_fl.observability.run import Run
from onion_fl.roles import EdgeSpec, build_federation


def arr(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


SENT = {
    "trunk.0.weight": arr(0, 0),
    "adapter.a.0.weight": arr(0),
    "adapter.b.0.weight": arr(0),
}


def view(**overrides) -> RoundView:
    fields = {
        "round": 1,
        "sent": SENT,
        "contributions": {},
        "aggregated": {},
        "previous": None,
        "received": {},
        "datasets": {},
        "reports": {},
        "participants": [],
        "late": 0,
        "quorum_failed": False,
        "time_to_quorum": None,
    }
    return RoundView(**(fields | overrides))


def child(source: str, **state: np.ndarray) -> Contribution:
    state = {k.replace("__", "."): v for k, v in state.items()}
    return Contribution(source, state, dict.fromkeys(state, 1.0))


def run_plugin(name: str, round_view: RoundView) -> dict[tuple, float]:
    out = diagnostics.create(name).compute(round_view)
    return {(n, tuple(sorted(t.items()))): v for n, v, t in out}


# --- divergence ---------------------------------------------------------------------------


def test_divergence_of_orthogonal_deltas() -> None:
    contributions = {
        "e1": child("e1", trunk__0__weight=arr(1, 0)),
        "e2": child("e2", trunk__0__weight=arr(0, 1)),
    }

    out = run_plugin("divergence", view(contributions=contributions))

    assert out[("divergence_cos", (("group", "trunk"),))] == pytest.approx(0.0)
    assert out[("divergence_l2", (("group", "trunk"),))] == pytest.approx(
        math.sqrt(0.5)
    )


def test_identical_deltas_do_not_diverge() -> None:
    contributions = {
        "e1": child("e1", trunk__0__weight=arr(2, 2)),
        "e2": child("e2", trunk__0__weight=arr(2, 2)),
    }

    out = run_plugin("divergence", view(contributions=contributions))

    assert out[("divergence_cos", (("group", "trunk"),))] == pytest.approx(1.0)
    assert out[("divergence_l2", (("group", "trunk"),))] == pytest.approx(0.0)


def test_divergence_needs_two_children_holding_the_group() -> None:
    contributions = {
        "e1": child("e1", adapter__a__0__weight=arr(1)),
        "e2": child("e2", adapter__b__0__weight=arr(1)),
    }

    assert run_plugin("divergence", view(contributions=contributions)) == {}


# --- dataset conflict -----------------------------------------------------------------------


def test_opposite_datasets_conflict_in_the_shared_groups() -> None:
    contributions = {
        "e1": child("e1", trunk__0__weight=arr(1, 0), adapter__a__0__weight=arr(1)),
        "e2": child("e2", trunk__0__weight=arr(3, 0), adapter__a__0__weight=arr(1)),
        "e3": child("e3", trunk__0__weight=arr(-1, 0), adapter__b__0__weight=arr(1)),
    }
    datasets = {"e1": "a", "e2": "a", "e3": "b"}

    out = run_plugin(
        "dataset_conflict", view(contributions=contributions, datasets=datasets)
    )

    assert out == {
        ("dataset_conflict", (("datasets", "a|b"), ("group", "trunk"))): pytest.approx(
            -1.0
        )
    }


def test_one_dataset_has_no_conflict() -> None:
    contributions = {
        "e1": child("e1", trunk__0__weight=arr(1, 0)),
        "e2": child("e2", trunk__0__weight=arr(0, 1)),
    }

    out = run_plugin(
        "dataset_conflict",
        view(contributions=contributions, datasets={"e1": "a", "e2": "a"}),
    )

    assert out == {}


# --- drift ------------------------------------------------------------------------------------


def test_drift_between_rounds_and_from_the_received_model() -> None:
    out = run_plugin(
        "drift",
        view(
            aggregated={"trunk.0.weight": arr(3, 4)},
            previous={"trunk.0.weight": arr(0, 0)},
            received={"trunk.0.weight": arr(3, 0)},
        ),
    )

    assert out[("drift", (("group", "trunk"),))] == pytest.approx(5.0)
    assert out[("zone_distance", (("group", "trunk"),))] == pytest.approx(4.0)


def test_no_drift_in_the_first_round() -> None:
    out = run_plugin("drift", view(aggregated={"trunk.0.weight": arr(1, 1)}))

    assert out == {}


# --- participation and fairness -----------------------------------------------------------------


def test_participation() -> None:
    contributions = {"e1": child("e1", trunk__0__weight=arr(1, 1))}

    ((name, value, tags),) = diagnostics.create("participation").compute(
        view(
            contributions=contributions,
            participants=["e1", "e2"],
            late=1,
            time_to_quorum=2.5,
        )
    )

    assert (name, value) == ("participation", 0.5)
    assert tags == {
        "selected": 2,
        "responded": 1,
        "late": 1,
        "quorum_failed": False,
        "time_to_quorum": 2.5,
    }


def test_fairness_per_dataset() -> None:
    reports = {
        "e1": {"eval.local.accuracy": 0.5, "eval.local.samples": 3},
        "e2": {"eval.local.accuracy": 0.9, "eval.local.samples": 1},
        "e3": {"eval.local.accuracy": 0.7, "eval.local.samples": 2},
    }
    datasets = {"e1": "a", "e2": "a", "e3": "b"}

    out = diagnostics.create("fairness").compute(
        view(reports=reports, datasets=datasets)
    )

    by = {t["dataset"]: (v, t) for _, v, t in out}
    assert by["a"][0] == pytest.approx(0.2)
    assert (by["a"][1]["min"], by["a"][1]["max"], by["a"][1]["metric"]) == (
        0.5,
        0.9,
        "local.accuracy",
    )
    assert by["b"][0] == 0.0


def test_registry_lists_the_built_ins() -> None:
    assert diagnostics.names() == [
        "dataset_conflict",
        "divergence",
        "drift",
        "fairness",
        "participation",
    ]


# --- in a federation ----------------------------------------------------------------------------

A = DataShape(dataset="a", task="t", n_features=2, n_classes=2)
B = DataShape(dataset="b", task="t", n_features=3, n_classes=2)
CONFIG = ModularMLPConfig(adapter_width=2, trunk_hidden=[2], dropout=0.0)


def model(shape) -> ModularMLP:
    return ModularMLP(CONFIG, [shape], seed=0)


def federation(fog: dict | None = None):
    topology = parse_topology(
        {
            "name": "t",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {"defaults": fog or {}, "nodes": [{"id": "fog_0"}]},
        }
    )
    edges = {
        "fog_0": [
            EdgeSpec(
                "e1",
                model(A),
                trainer=trainers.create("stub", {"shift": 1}),
                tags={"dataset": "a"},
            ),
            EdgeSpec(
                "e2",
                model(B),
                trainer=trainers.create("stub", {"shift": -1}),
                tags={"dataset": "b"},
            ),
        ]
    }
    initial = state_arrays(ModularMLP(CONFIG, [A, B], seed=0))
    return topology, build_federation(topology, edges, initial_state=initial, rounds=2)


def events(fed, name: str) -> list[dict]:
    return [e for e in fed.runtime.events if e["name"] == name]


def test_aggregators_emit_diagnostics_every_round() -> None:
    _, fed = federation()
    fed.run()

    conflicts = events(fed, "diagnostic.dataset_conflict")
    assert {e["tags"]["round"] for e in conflicts} == {1, 2}
    trunk = [e for e in conflicts if e["tags"]["group"] == "trunk"]
    assert all(e["value"] == pytest.approx(-1.0) for e in trunk)  # +1 against -1
    assert all(
        e["node"] in ("fog_0", "cloud") for e in events(fed, "diagnostic.participation")
    )
    assert [
        e["tags"]["round"]
        for e in events(fed, "diagnostic.drift")
        if e["node"] == "fog_0"
    ] == [2, 2, 2, 2]


def test_diagnostics_can_be_chosen_or_switched_off() -> None:
    _, chosen = federation({"diagnostics": ["participation"]})
    _, off = federation({"diagnostics": []})
    chosen.run()
    off.run()

    names = {
        e["name"]
        for e in chosen.runtime.events
        if e["node"] == "fog_0" and e["name"].startswith("diagnostic.")
    }
    assert names == {"diagnostic.participation"}
    assert not [
        e
        for e in off.runtime.events
        if e["node"] == "fog_0" and e["name"].startswith("diagnostic.")
    ]


def test_the_run_reports_traffic_per_link_and_round(tmp_path: Path) -> None:
    topology, fed = federation()
    run = Run(tmp_path, config={}, topology=topology, seed=0)
    run.attach(fed)
    fed.run()
    run.finish()

    comm = [
        json.loads(line)
        for line in (run.path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"diagnostic.communication"' in line
    ]
    links = {(e["tags"]["src"], e["tags"]["dst"], e["round"]) for e in comm}
    assert ("e1", "fog_0", 1) in links and ("fog_0", "cloud", 2) in links
    assert all(e["value"] > 0 and e["tags"]["messages"] >= 1 for e in comm)


def test_participation_counts_late_updates_in_the_round_they_arrive() -> None:
    from onion_fl.runtime.devices import compute_models

    topology = parse_topology(
        {
            "name": "t",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {
                "defaults": {"quorum": 0.5, "deadline": 10},
                "nodes": [{"id": "fog_0"}],
            },
        }
    )
    slow = compute_models.create("samples_per_second", {"samples_per_second": 10 / 15})
    edges = {
        "fog_0": [
            EdgeSpec("fast", model(A), trainer=trainers.create("stub")),
            EdgeSpec("slow", model(A), trainer=trainers.create("stub"), compute=slow),
        ]
    }
    fed = build_federation(
        topology, edges, initial_state=state_arrays(model(A)), rounds=2
    )
    fed.run()

    late = {
        e["tags"]["round"]: e["tags"]["late"]
        for e in events(fed, "diagnostic.participation")
        if e["node"] == "fog_0"
    }
    assert late == {1: 0, 2: 1}
