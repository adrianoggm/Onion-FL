"""Tests for the compute and availability models (issue #81)."""

from __future__ import annotations

import numpy as np
import pytest

from onion_fl.core.registry import PluginError
from onion_fl.runtime.devices import availability_models, compute_models


def test_both_model_families_are_registered() -> None:
    assert compute_models.names() == [
        "lognormal_rate",
        "measured",
        "samples_per_second",
        "trace",
    ]
    assert availability_models.names() == [
        "always",
        "bernoulli",
        "crash_at",
        "schedule",
        "windows",
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


def lognormal(sigma: float = 0.5, **kw):
    params = {"samples_per_second": 100, "sigma": sigma} | kw
    return compute_models.create("lognormal_rate", params)


def test_lognormal_rate_gives_the_same_times_for_the_same_stream() -> None:
    model = lognormal()

    def times(seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        return [model.duration(samples=100, wall_s=0.0, rng=rng) for _ in range(5)]

    assert times(3) == times(3)
    assert times(3) != times(4)
    assert len(set(times(3))) == 5


def test_lognormal_rate_keeps_the_mean_duration_of_its_rate() -> None:
    model, rng = lognormal(sigma=0.5, overhead_s=1.0), np.random.default_rng(0)

    times = [model.duration(samples=100, wall_s=0.0, rng=rng) for _ in range(20_000)]

    assert np.mean(times) == pytest.approx(2.0, rel=0.01)  # 1 s overhead + 100/100
    assert np.std(np.log(np.array(times) - 1.0)) == pytest.approx(0.5, rel=0.02)


def test_lognormal_rate_without_dispersion_is_samples_per_second() -> None:
    model, rng = lognormal(sigma=0.0), np.random.default_rng(0)

    assert model.duration(samples=250, wall_s=9.0, rng=rng) == pytest.approx(2.5)


def test_stochastic_models_draw_nothing_without_work() -> None:
    rng = np.random.default_rng(0)
    trace = compute_models.create("trace", {"durations": [1.0, 2.0]})

    assert lognormal().duration(samples=0, wall_s=1.0, rng=rng) == 0.0
    assert trace.duration(samples=0, wall_s=1.0, rng=rng) == 0.0
    assert rng.random() == np.random.default_rng(0).random()


def test_trace_replays_its_measured_durations() -> None:
    durations = [1.5, 4.0, 9.25]
    model, rng = (
        compute_models.create("trace", {"durations": durations}),
        (np.random.default_rng(1)),
    )

    times = [model.duration(samples=10, wall_s=0.0, rng=rng) for _ in range(3000)]

    assert set(times) == set(durations)
    for d in durations:  # uniform over the trace
        assert times.count(d) / len(times) == pytest.approx(1 / 3, abs=0.03)


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("lognormal_rate", {"samples_per_second": 1, "sigma": -0.1}),
        ("trace", {"durations": []}),
        ("trace", {"durations": [1.0, -2.0]}),
    ],
)
def test_stochastic_compute_params_are_validated(name: str, params: dict) -> None:
    with pytest.raises(PluginError):
        compute_models.create(name, params)


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
