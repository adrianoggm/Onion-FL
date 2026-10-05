"""Stream experiments end to end (continuum C3, issue #158).

The dataset is a small CSV format fixture with a time column; edges use the
stub trainer and scores come from a stub scorer, so nothing is trained or
evaluated for real (docs/RULES.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from onion_fl.experiment.config import ConfigError, parse_experiment
from onion_fl.experiment.runner import plan, run_scenario
from onion_fl.experiment.sweep import scenarios

TOPOLOGY = {
    "name": "two_fogs",
    "levels": ["global", "fog", "edge"],
    "root": {"id": "cloud"},
    "fog": {
        "nodes": [{"id": "fog_a", "home": "demo"}, {"id": "fog_b", "home": "demo"}]
    },
}
STREAM = {"bootstrap": 120, "round_every": 300, "batch_size": 1, "speed": 60}


def descriptor(workspace: Path, name: str, timed: bool) -> None:
    lines = [
        f"name: {name}",
        f"root: {(workspace / 'raw').as_posix()}",
        "source: {reader: csv, path: table.csv}",
        "steps:",
        "  - subject: {column: pp}",
        *(["  - time: {column: sec}"] if timed else []),
        '  - label: {task: stress, column: cond, map: {"N": 0, "T": 1}}',
        "  - features: {exclude: [sec]}",
    ]
    path = workspace / "datasets" / f"{name}.yaml"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    rows = ["pp,sec,cond,f1,f2"]
    for subject in range(1, 13):
        for i in range(20):  # one row a minute
            rows.append(f"{subject},{i * 60},{'NT'[(i // 4) % 2]},{subject + i},{i}")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "table.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (tmp_path / "datasets").mkdir()
    descriptor(tmp_path, "demo", timed=True)
    descriptor(tmp_path, "plain", timed=False)
    return tmp_path


def experiment(workspace: Path, **overrides) -> dict:
    base = {
        "name": "stream_exp",
        "topology": TOPOLOGY,
        "data": {
            "datasets": {"demo": {}},
            "roles": {"test": 0.25, "val": 0.17, "seed": 0},
            "placement": {"name": "mixing", "alpha": 0.0},
        },
        "learning": {
            "model": {"name": "modular_mlp", "adapter_width": 4, "trunk_hidden": [4]},
            "trainer": "stub",
        },
        "rounds": 10,
        "stream": STREAM,
        "labels": {"fraction": 0.5, "delay": 60},
        "evaluation": {"global": {"every": 1}, "aggregators": {"every": 1}},
        "seeds": [0],
        "paths": {
            "datasets": str(workspace / "datasets"),
            "runs": str(workspace / "runs"),
            "cache": str(workspace / "cache"),
            "topologies": str(workspace / "topologies"),
        },
    }
    return base | overrides


def stub_score(model, data):
    return {"accuracy": 0.5}, data.n_samples


def run(workspace: Path, **overrides) -> Path:
    (scenario,) = scenarios(parse_experiment(experiment(workspace, **overrides)))
    return run_scenario(scenario, evaluate=stub_score)


def events(path: Path, name: str) -> list[dict]:
    lines = (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [e for e in map(json.loads, lines) if e["name"] == name]


def test_a_stream_run_lasts_until_the_horizon_with_its_volume_recorded(
    workspace: Path,
) -> None:
    path = run(workspace)

    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished"
    # rows from t0 = 120 s to 1140 s arrive by 1020 s: rounds at 0, 300, 600, 900, 1200
    assert summary["rounds"] == 5
    (stream,) = events(path, "data.stream")
    assert stream["value"] == 1020 and stream["tags"]["rounds"] == 5
    edges = stream["tags"]["edges"]
    assert all(e["rows"] == 20 and e["history"] == 2 for e in edges.values())
    arrived = events(path, "data.arrived")
    assert sum(e["value"] for e in arrived) == 20 * len(edges)


def test_the_same_seed_gives_the_same_stream_run(workspace: Path) -> None:
    def trace(path: Path) -> tuple:
        with np.load(path / "model.npz") as archive:
            model = {k: archive[k] for k in archive.files}
        arrived = [
            (e["node"], e["round"], e["value"], e["tags"]["labelled"])
            for e in events(path, "data.arrived")
        ]
        return model, arrived

    (first, arrived), (second, again) = trace(run(workspace)), trace(run(workspace))
    assert arrived == again
    for key, value in first.items():
        np.testing.assert_array_equal(second[key], value, err_msg=key)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"runtime": {"mode": "real"}}, "real"),
        (
            {
                "learning": {
                    "model": {"name": "modular_mlp"},
                    "trainer": "stub",
                    "init": {"name": "run", "run": "x"},
                }
            },
            "init",
        ),
        ({"roles": {"local_val": 0.2}}, "local_val"),
        ({"roles": {"scaler": "local"}}, "scaler"),
        (
            {
                "roles": {"subjects_per_client": 2},
                "stream": STREAM | {"start": {"staggered": 60}},
            },
            "subjects_per_client",
        ),
        (
            {
                "roles": {"subjects_per_client": 2},
                "stream": STREAM | {"bootstrap": {"samples": 2}},
            },
            "subjects_per_client",
        ),
        ({"stream": None}, "labels"),
    ],
    ids=[
        "real",
        "init_run",
        "local_val",
        "local_scaler",
        "staggered",
        "samples",
        "labels",
    ],
)
def test_what_a_stream_cannot_do_yet_is_refused(
    workspace: Path, change: dict, match: str
) -> None:
    raw = experiment(workspace)
    if "roles" in change:
        raw["data"] = raw["data"] | {
            "roles": raw["data"]["roles"] | change.pop("roles")
        }
    raw |= change

    with pytest.raises(ConfigError, match=match):
        plan(parse_experiment(raw))


def test_a_dataset_without_time_cannot_stream(workspace: Path) -> None:
    raw = experiment(workspace)
    raw["data"] = raw["data"] | {"datasets": {"plain": {}}}
    raw["topology"] = {
        **TOPOLOGY,
        "fog": {"nodes": [{"id": "fog_a", "home": "plain"}, {"id": "fog_b"}]},
    }

    with pytest.raises(ConfigError, match="time"):
        plan(parse_experiment(raw))


def test_a_run_without_stream_keeps_its_config_id(workspace: Path) -> None:
    raw = experiment(workspace)
    del raw["stream"], raw["labels"]

    dumped = parse_experiment(raw).dump()
    assert "stream" not in dumped and "labels" not in dumped


def test_the_plan_previews_the_stream(workspace: Path) -> None:
    (preview,) = plan(parse_experiment(experiment(workspace)))

    stream = preview["stream"]
    assert stream["horizon"] == 1020 and stream["rounds"] == 5
    assert all(e["rows"] == 20 for e in stream["edges"].values())
