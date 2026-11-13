"""Tests for the summary of the degradation x topology study (issue #290).

Hand-made configs and score tables only: nothing is trained, nothing reads data.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "degradation_topology_summary",
    ROOT / "scripts" / "degradation_topology_summary.py",
)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def config(name: str, infra: str, **fog) -> dict:
    return {
        "topology": {
            "name": name,
            "root": {"id": "cloud"},
            "fog": {"defaults": {"quorum": 0.5, **fog}},
            "edge": {"availability": fog.pop("availability", None)},
        },
        "infrastructure": {
            "name": infra,
            "assignment": {
                "root": "cloud-1",
                "aggregators": {"fog_a": "fog-a1", "hub": "cloud-1"},
                "edges": {"hosts": {"e1": "field_a-raspberry_pi-1", "e2": "gw-b"}},
            },
        },
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (config("dt_flat", "degradation_lab"), ("flat", "nominal")),
        (config("dt_two_levels", "degradation_lab_loss20"), ("two", "loss20")),
        (
            config("dt_three_levels", "degradation_lab_crash_cifar10"),
            ("three", "crash"),
        ),
        (
            config("dt_two_levels", "degradation_lab_partition", autonomy={"after": 1}),
            ("two", "partition_autonomy"),
        ),
    ],
)
def test_each_run_is_labelled_with_its_arm(raw: dict, expected: tuple) -> None:
    assert summary.arm(raw) == expected


def test_dropout_and_its_strict_quorum_are_told_apart() -> None:
    raw = config("dt_flat", "degradation_lab")
    raw["topology"]["edge"]["availability"] = {"name": "bernoulli", "p": 0.2}
    assert summary.arm(raw) == ("flat", "dropout")
    raw["topology"]["fog"]["defaults"]["quorum"] = 1.0
    assert summary.arm(raw) == ("flat", "dropout_k1")


def test_nodes_take_the_side_of_the_wan_their_host_is_on() -> None:
    zones = summary.zones_of(config("dt_flat", "degradation_lab"))
    assert zones == {"cloud": "dc", "fog_a": "a", "hub": "dc", "e1": "a", "e2": "b"}
    assert [summary.wan_links(*p) for p in [("a", "a"), ("a", "dc"), ("a", "b")]] == [
        0,
        1,
        2,
    ]


def rows(values: dict[tuple[str, str], list[float]]) -> list[dict]:
    return [
        {"topology": t, "level": level, "seed": s, "macro_f1": v}
        for (t, level), vs in values.items()
        for s, v in enumerate(vs)
    ]


def test_the_interaction_is_the_difference_of_the_slopes() -> None:
    table = {
        ("two", "nominal"): [0.7, 0.8, 0.9],
        ("two", "loss20"): [0.6, 0.7, 0.8],  # slope -0.1
        ("flat", "nominal"): [0.7, 0.8, 0.9],
        ("flat", "loss20"): [0.4, 0.5, 0.6],  # slope -0.3
        ("three", "nominal"): [0.7, 0.8, 0.9],
        ("three", "loss20"): [0.6, 0.7, 0.8],
    }
    out = summary.contrasts(rows(table), ["macro_f1"])

    (flat,) = [c for c in out["h2"] if c["contrast"] == "DiD flat - two"]
    (three,) = [c for c in out["h2"] if c["contrast"] == "DiD three - two"]
    assert flat["delta"] == pytest.approx(-0.2) and three["delta"] == pytest.approx(0)
    assert three["verdict"] == "equivalent"
    h1 = {c["contrast"]: c for c in out["h1"] if c["level"] == "loss20"}
    assert h1["flat - two"]["delta"] == pytest.approx(-0.2)
    assert h1["three - two"]["seeds_needed"] == 2  # no spread: certain


def test_the_seed_rule_takes_the_largest_need_within_its_bounds() -> None:
    h1 = [
        {
            "metric": "macro_f1",
            "level": "loss5",
            "contrast": "flat - two",
            "seeds_needed": 4,
            "margin": 0.02,
            "sd_diff": 0.01,
        },
        {
            "metric": "time_to_target_s",
            "level": "loss5",
            "contrast": "flat - two",
            "seeds_needed": 500,
            "margin": math.log(1.1),
            "sd_diff": 1.0,
        },
        {
            "metric": "accuracy",
            "level": "loss5",
            "contrast": "flat - two",
            "seeds_needed": 900,
            "margin": 0.02,
            "sd_diff": 1.0,
        },  # not primary
    ]
    out = summary.seed_rule(h1)
    assert out["n"] == 30 and out["largest"] == 500
    assert [u["metric"] for u in out["underpowered"]] == ["time_to_target_s"]
    assert summary.seed_rule(h1[:1])["n"] == 10
