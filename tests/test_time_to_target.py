"""Tests for the time to target and its comparison (issue #209, spec N5).

The curves are hand-written event logs with known crossing instants, to check
the arithmetic of the measure and of the censoring; they are not results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from onion_fl.observability.analysis import load_runs


def event(t, round, node, name, value, **tags) -> dict:
    return {
        "t_virtual": float(t),
        "t_wall": 0.0,
        "round": round,
        "level": "global" if node == "cloud" else "fog",
        "node": node,
        "role": "aggregator",
        "kind": "metric" if name.startswith("eval.") else "lifecycle",
        "name": name,
        "value": value,
        "tags": {"model": "global", **tags},
    }


def write_run(root: Path, run_id: str, topology: str, seed: int, events) -> None:
    path = root / run_id
    path.mkdir(parents=True)
    meta = {"run_id": run_id, "topology_id": topology, "seed": seed, "config": {}}
    (path / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    lines = [json.dumps(e | {"run_id": run_id, "seed": seed}) for e in events]
    (path / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def curve(points, end: float | None = None, name="eval.accuracy") -> list[dict]:
    """``(t, round, value)`` of the cloud, then the run's last event at ``end``."""
    events = [event(t, r, "cloud", name, v) for t, r, v in points]
    if end is not None:
        events.append(event(end, None, "cloud", "run.finished", None))
    return events


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    # T1 seed 0 crosses 0.8 in round 3 at t=30; seed 1 crosses in round 2 at t=12.
    write_run(
        tmp_path, "a0", "T1", 0, curve([(10, 1, 0.5), (20, 2, 0.7), (30, 3, 0.9)], 35)
    )
    write_run(tmp_path, "a1", "T1", 1, curve([(6, 1, 0.6), (12, 2, 0.85)], 18))
    # T2 seed 0 never reaches 0.8: censored at its last event, t=40.
    write_run(
        tmp_path, "b0", "T2", 0, curve([(10, 1, 0.5), (20, 2, 0.79), (30, 3, 0.7)], 40)
    )
    # T2 seed 1 reaches it exactly in round 1.
    write_run(tmp_path, "b1", "T2", 1, curve([(8, 1, 0.8), (16, 2, 0.9)], 20))
    return tmp_path


def test_the_first_instant_the_curve_reaches_the_target(runs_dir: Path) -> None:
    out = load_runs(runs_dir).time_to_target("global", "accuracy", 0.8)
    rows = out.set_index("run_id")

    assert rows.loc["a0", ["time", "round"]].tolist() == [30.0, 3]
    assert rows.loc["a1", ["time", "round"]].tolist() == [12.0, 2]
    assert rows.loc["b1", ["time", "round"]].tolist() == [8.0, 1]  # >= reaches it
    assert not rows.loc[["a0", "a1", "b1"], "censored"].any()


def test_a_run_that_never_reaches_it_is_censored_at_its_duration(
    runs_dir: Path,
) -> None:
    out = load_runs(runs_dir).time_to_target("global", "accuracy", 0.8)
    b0 = out.set_index("run_id").loc["b0"]

    assert b0["censored"] and not b0["reached"]
    assert b0["time"] == 40.0  # the run's last event, not its last evaluation
    assert math.isnan(b0["round"])


def test_the_nodes_of_a_level_are_averaged_before_the_threshold(
    tmp_path: Path,
) -> None:
    events = [
        event(10, 1, "fog_0", "eval.accuracy", 0.9),
        event(11, 1, "fog_1", "eval.accuracy", 0.5),  # mean 0.7: not yet
        event(20, 2, "fog_0", "eval.accuracy", 0.9),
        event(23, 2, "fog_1", "eval.accuracy", 0.8),  # mean 0.85, known at t=23
    ]
    write_run(tmp_path, "r", "T", 0, events)

    out = load_runs(tmp_path).time_to_target("fog", "accuracy", 0.8)

    assert out[["time", "round", "censored"]].iloc[0].tolist() == [23.0, 2, False]


def test_lower_is_better_for_losses(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "r",
        "T",
        0,
        curve([(5, 1, 1.0), (9, 2, 0.4), (14, 3, 0.2)], name="eval.loss"),
    )

    out = load_runs(tmp_path).time_to_target("global", "loss", 0.5, lower=True)

    assert out[["time", "round"]].iloc[0].tolist() == [9.0, 2]


def test_compare_gives_the_mean_time_with_a_bootstrap_interval(
    runs_dir: Path,
) -> None:
    runs = load_runs(runs_dir)
    out = runs.compare("global", "accuracy", by=["topology_id"], target=0.8)
    rows = out.set_index("topology_id")

    t1 = rows.loc["T1"]
    assert (t1["n"], t1["reached"], t1["censored"]) == (2, 2, 0)
    assert t1["mean"] == pytest.approx(21.0)
    assert 12.0 <= t1["ci_low"] < 21.0 < t1["ci_high"] <= 30.0

    t2 = rows.loc["T2"]  # the censored run counts at its duration, and is reported
    assert (t2["n"], t2["reached"], t2["censored"]) == (2, 1, 1)
    assert t2["mean"] == pytest.approx(24.0)

    again = runs.compare("global", "accuracy", by=["topology_id"], target=0.8)
    assert again.equals(out)  # the bootstrap is seeded


def test_compare_of_one_run_has_no_interval(runs_dir: Path) -> None:
    out = load_runs(runs_dir).compare("global", "accuracy", by=["run_id"], target=0.8)

    assert len(out) == 4 and out["ci_low"].map(math.isnan).all()


def test_time_to_target_of_nothing_is_empty(runs_dir: Path) -> None:
    runs = load_runs(runs_dir)

    assert runs.time_to_target("fog", "accuracy", 0.8).empty
    assert runs.compare("fog", "accuracy", target=0.8).empty
