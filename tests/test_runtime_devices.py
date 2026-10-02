"""Tests for the compute and availability models (issue #81)."""

from __future__ import annotations

import pytest

from onion_fl.core.registry import PluginError
from onion_fl.runtime.devices import availability_models, compute_models


def test_both_model_families_are_registered() -> None:
    assert compute_models.names() == ["measured", "samples_per_second"]
    assert availability_models.names() == [
        "always",
        "bernoulli",
        "crash_at",
        "schedule",
    ]


def test_samples_per_second_turns_work_into_virtual_seconds() -> None:
    model = compute_models.create(
        "samples_per_second", {"samples_per_second": 500, "overhead_s": 0.5}
    )

    assert model.duration(samples=1000, wall_s=99.0) == pytest.approx(2.5)


def test_overhead_only_applies_when_there_is_work() -> None:
    model = compute_models.create(
        "samples_per_second", {"samples_per_second": 500, "overhead_s": 0.5}
    )

    assert model.duration(samples=0, wall_s=1.0) == 0.0


def test_measured_scales_the_wall_time() -> None:
    model = compute_models.create("measured", {"factor": 2.0})

    assert model.duration(samples=0, wall_s=0.25) == pytest.approx(0.5)


def test_compute_params_are_validated() -> None:
    with pytest.raises(PluginError, match="samples_per_second"):
        compute_models.create("samples_per_second", {"samples_per_second": 0})


def test_always_is_up() -> None:
    model = availability_models.create("always")

    assert model.is_up(1e9, round=3, node_id="edge_1", seed=0)


def test_crash_at_takes_the_node_down_for_good() -> None:
    model = availability_models.create("crash_at", {"t": 5.0})

    assert model.is_up(4.99, round=None, node_id="edge_1", seed=0)
    assert not model.is_up(5.0, round=None, node_id="edge_1", seed=0)
    assert not model.is_up(1e6, round=None, node_id="edge_1", seed=0)


def test_schedule_takes_the_node_down_inside_its_windows() -> None:
    model = availability_models.create(
        "schedule", {"offline": [[1.0, 2.0], [5.0, 6.0]]}
    )

    up = [
        model.is_up(t, round=None, node_id="e", seed=0)
        for t in (0.5, 1.0, 1.5, 2.0, 5.5, 7.0)
    ]

    assert up == [True, False, False, True, False, True]


def test_schedule_windows_must_be_ordered() -> None:
    with pytest.raises(PluginError):
        availability_models.create("schedule", {"offline": [[2.0, 1.0]]})


def test_bernoulli_only_affects_messages_of_a_round() -> None:
    model = availability_models.create("bernoulli", {"p": 1.0})

    assert not model.is_up(0.0, round=1, node_id="edge_1", seed=0)
    assert model.is_up(0.0, round=None, node_id="edge_1", seed=0)


def test_bernoulli_is_deterministic_per_node_and_round() -> None:
    model = availability_models.create("bernoulli", {"p": 0.5})

    def outcomes(node: str, seed: int) -> list[bool]:
        return [model.is_up(0.0, round=r, node_id=node, seed=seed) for r in range(60)]

    assert outcomes("edge_1", 7) == outcomes("edge_1", 7)
    assert outcomes("edge_1", 7) != outcomes("edge_2", 7)
    assert outcomes("edge_1", 7) != outcomes("edge_1", 8)
    assert 10 < sum(outcomes("edge_1", 7)) < 50
