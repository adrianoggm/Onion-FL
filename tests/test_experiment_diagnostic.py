"""Tests for the ex ante diagnostic of the deadline bias (issue #212, spec N8).

The bias and the star are checked against values worked out by hand. The runs
use the stub trainer on a CSV format fixture (docs/RULES.md): nothing is
trained or evaluated, and the diagnostic is computed before the first round.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from onion_fl.core.ids import config_id
from onion_fl.experiment.config import parse_experiment
from onion_fl.experiment.diagnostic import bias, diagnose
from onion_fl.experiment.predictor import Round, planned_round, rounds_of
from onion_fl.experiment.runner import _scenario_data, build_scenario
from onion_fl.experiment.sweep import Scenario
from onion_fl.observability.events import read_events
from tests.test_experiment import workspace  # noqa: F401
from tests.test_experiment_predictor import _run, _star, _topology

# --- by hand ---------------------------------------------------------------------------


def test_the_bias_is_the_covariance_normalised_by_the_means() -> None:
    # The only divergent edge misses: none of the divergence arrives.
    assert bias([1, 0], [0.5, 0.5], [0, 2]) == {"covariance": -0.25, "D": -1.0}
    # The only divergent edge arrives: twice its share of the edges.
    assert bias([1, 0], [0.5, 0.5], [2, 0]) == {"covariance": 0.25, "D": 1.0}
    # The same divergence everywhere: who arrives does not matter.
    assert bias([1, 0], [0.5, 0.5], [1, 1]) == {"covariance": 0.0, "D": 0.0}
    assert bias([1, 0.5], [0.5, 0.5], [0, 0])["D"] == 0.0  # nothing diverges
    assert bias([0, 0], [0.5, 0.5], [1, 3])["D"] is None  # nobody arrives
    # Weights scale what each edge brings: p = (1, .5), u = (.3, .2), so
    # mean(p u) = .2, mean(p) = .75 and mean(u) = .25.
    out = bias([1.0, 0.5], [0.75, 0.25], [0.4, 0.8])
    assert out["covariance"] == pytest.approx(0.2 - 0.75 * 0.25)
    assert out["D"] == pytest.approx(0.0125 / 0.1875)


def _round() -> Round:
    """One round of the star: ``a`` (30 examples) and ``b`` (10), 10 samples of
    work each, 100 bytes each way."""
    return Round(
        children={"c": ["a", "b"]},
        down={("c", "a"): 100, ("c", "b"): 100},
        up={("a", "c"): 100, ("b", "c"): 100},
        samples={"a": 10, "b": 10},
        examples={"a": 30, "b": 10},
    )


def test_the_star_loses_the_edge_that_misses_its_deadline() -> None:
    # b needs 5 s and has 5 - 2.2: it never arrives; a always does.
    out = diagnose(_star(), "c", _round(), {"label_divergence": {"a": 0.0, "b": 1.0}})

    assert out["statistic"] == "label_divergence"
    assert {e: v["p"] for e, v in out["edges"].items()} == {"a": 1.0, "b": 0.0}
    assert out["edges"]["a"]["w"] == 0.75 and out["edges"]["b"]["w"] == 0.25
    # u = (0, .25), mean(p) = .5, mean(u) = .125, mean(p u) = 0
    assert out["covariance"] == pytest.approx(-0.0625)
    assert out["D"] == pytest.approx(-1.0)
    assert out["participation"] == 0.5  # the first baseline: the mean rate alone
    assert out["floss"] == {
        "participation": 0.75,
        "rounds_factor": pytest.approx(4 / 3),
    }
    assert out["deadlines"] == ["c"]


def test_without_deadlines_everyone_arrives_and_nothing_is_biased() -> None:
    plan = _star()
    plan.deadlines = {"c": None}

    out = diagnose(plan, "c", _round(), {"gradient_norm": {"a": 0.1, "b": 9.0}})

    assert out["participation"] == 1.0 and out["D"] == 0.0
    assert out["floss"]["rounds_factor"] == 1.0 and out["deadlines"] == []


# --- on runs ---------------------------------------------------------------------------


def _metas(runs: Path) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(runs.glob("*/run.json"))
    ]


def test_every_run_of_a_sweep_saves_its_diagnostic_in_seconds(
    workspace: Path,  # noqa: F811
) -> None:
    # The fog rounds close at 1.5 s: 10 examples at 20 samples/s (0.5 s) make
    # it, at 5 samples/s (2 s) they never do.
    runs = _run(
        workspace,
        "runs",
        topology=_topology(deadline="1.5s"),
        rounds=2,
        sweep={"topology.edge.device.samples_per_second": [20.0, 5.0]},
    )

    found = {}
    for meta in _metas(runs):
        diagnostic = meta["diagnostic"]
        assert "error" not in diagnostic
        assert diagnostic["seconds"] < 10
        assert diagnostic["deadlines"] == ["fog_a", "fog_b"]
        assert set(diagnostic["statistics"]) == {"label_divergence", "gradient_norm"}
        assert all(e["gradient_norm"] > 0 for e in diagnostic["edges"].values())
        found[meta["scenario"]] = diagnostic["participation"]
    assert sorted(found.values()) == [0.0, 1.0]


def test_the_planned_round_is_the_one_the_run_sends(
    workspace: Path,  # noqa: F811
) -> None:
    runs = _run(workspace, "runs", topology=_topology(), rounds=3)
    path = next(runs.glob("*/run.json")).parent
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    root, rounds = rounds_of(read_events(path / "events.jsonl"))
    config = meta["config"]
    scenario = Scenario(
        meta["scenario"], meta["seed"], parse_experiment(config), config_id(config)
    )
    topology, split, placement, _ = _scenario_data(scenario, None)
    federation = build_scenario(scenario, topology, split, placement, parent=None)

    planned_root, planned = planned_round(federation)

    sent = rounds[2]  # past the bootstrap of round 1
    assert planned_root == root
    assert {k: sorted(v) for k, v in planned.children.items()} == {
        k: sorted(v) for k, v in sent.children.items()
    }
    assert planned.samples == sent.samples and planned.examples == sent.examples
    assert planned.down.keys() == sent.down.keys()
    for link, size in sent.down.items():  # JSON: the digits of the weights vary
        assert planned.down[link] == pytest.approx(size, rel=0.1)
    for link, size in sent.up.items():
        assert planned.up[link] == pytest.approx(size, rel=0.1)
    assert meta["diagnostic"]["participation"] == 1.0  # no deadline
