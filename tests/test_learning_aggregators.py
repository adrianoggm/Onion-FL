"""Tests for per-key aggregators and server optimizers (issue #84).

Arrays here are hand-picked numbers to check arithmetic; nothing is trained.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from onion_fl.core.registry import PluginError
from onion_fl.learning.aggregators import (
    AggregationError,
    Contribution,
    aggregators,
    server_optimizers,
)


def c(
    source: str, weights: dict[str, float] | None = None, **state: list[float]
) -> Contribution:
    arrays = {
        key.replace("__", "."): np.asarray(v, dtype=np.float32)
        for key, v in state.items()
    }
    if weights is None:
        weights = dict.fromkeys(arrays, 1.0)
    return Contribution(source=source, state=arrays, weights=weights)


def agg(name: str, contributions, params=None) -> Contribution:
    return aggregators.create(name, params).aggregate(contributions, source="fog_0")


# --- fedavg --------------------------------------------------------------------


def test_fedavg_is_the_sample_weighted_mean() -> None:
    out = agg(
        "fedavg",
        [
            c("e1", {"trunk.w": 1}, trunk__w=[0.0, 4.0]),
            c("e2", {"trunk.w": 3}, trunk__w=[4.0, 8.0]),
        ],
    )

    np.testing.assert_allclose(out.state["trunk.w"], [3.0, 7.0])
    assert out.weights == {"trunk.w": 4}
    assert out.source == "fog_0"


def test_each_key_is_averaged_only_among_the_contributions_that_have_it() -> None:
    out = agg(
        "fedavg",
        [
            c(
                "e1",
                {"trunk.w": 2, "adapter.swell.w": 2},
                trunk__w=[1.0],
                adapter__swell__w=[10.0],
            ),
            c(
                "e2",
                {"trunk.w": 2, "adapter.sweet.w": 6},
                trunk__w=[3.0],
                adapter__sweet__w=[20.0],
            ),
        ],
    )

    np.testing.assert_allclose(out.state["trunk.w"], [2.0])
    np.testing.assert_allclose(out.state["adapter.swell.w"], [10.0])
    np.testing.assert_allclose(out.state["adapter.sweet.w"], [20.0])
    assert out.weights == {"trunk.w": 4, "adapter.swell.w": 2, "adapter.sweet.w": 6}


def test_fedavg_needs_a_weight_for_every_key() -> None:
    contribution = Contribution(
        source="e1", state={"w": np.ones(1, np.float32)}, weights={}
    )

    with pytest.raises(AggregationError, match="weight"):
        agg("fedavg", [contribution])


def test_zero_total_weight_falls_back_to_the_plain_mean() -> None:
    out = agg("fedavg", [c("e1", {"w": 0}, w=[2.0]), c("e2", {"w": 0}, w=[4.0])])

    np.testing.assert_allclose(out.state["w"], [3.0])
    assert out.weights == {"w": 0}


# --- unweighted and robust aggregators ------------------------------------------


def test_mean_ignores_the_sample_counts() -> None:
    out = agg("mean", [c("e1", {"w": 1}, w=[0.0]), c("e2", {"w": 99}, w=[10.0])])

    np.testing.assert_allclose(out.state["w"], [5.0])
    assert out.weights == {"w": 100}


def test_median_is_coordinate_wise() -> None:
    out = agg(
        "median",
        [c("e1", w=[1.0, 9.0]), c("e2", w=[2.0, 7.0]), c("e3", w=[100.0, 8.0])],
    )

    np.testing.assert_allclose(out.state["w"], [2.0, 8.0])


def test_trimmed_mean_drops_the_extremes() -> None:
    values = [[0.0], [1.0], [2.0], [3.0], [1000.0]]
    out = agg(
        "trimmed_mean", [c(f"e{i}", w=v) for i, v in enumerate(values)], {"beta": 0.2}
    )

    np.testing.assert_allclose(out.state["w"], [2.0])


def test_trimmed_mean_with_too_few_contributions_keeps_the_middle() -> None:
    out = agg("trimmed_mean", [c("e1", w=[1.0]), c("e2", w=[5.0])], {"beta": 0.49})

    np.testing.assert_allclose(out.state["w"], [3.0])


@pytest.mark.parametrize("beta", [-0.1, 0.5, 0.9])
def test_trimmed_mean_beta_is_validated(beta: float) -> None:
    with pytest.raises(PluginError, match="beta"):
        aggregators.create("trimmed_mean", {"beta": beta})


# --- guarantees shared by every aggregator ----------------------------------------

ALL = [
    ("fedavg", None),
    ("mean", None),
    ("median", None),
    ("trimmed_mean", {"beta": 0.2}),
]


def _updates() -> list[Contribution]:
    rng = np.random.default_rng(
        0
    )  # arbitrary numbers to check arithmetic, not training data
    return [
        c(
            f"edge_{i}",
            {"trunk.w": i + 1, "head.h": 2 * i + 1},
            trunk__w=rng.normal(size=6),
            head__h=rng.normal(size=3),
        )
        for i in range(5)
    ]


@pytest.mark.parametrize("name, params", ALL)
def test_the_result_does_not_depend_on_arrival_order(name: str, params) -> None:
    updates = _updates()
    reference = agg(name, updates, params)

    for order in itertools.islice(itertools.permutations(updates), 1, 25):
        out = agg(name, list(order), params)
        for key, value in reference.state.items():
            assert np.array_equal(out.state[key], value), (name, key)


@pytest.mark.parametrize("name", ["fedavg", "mean"])
def test_rounding_does_not_depend_on_arrival_order(name: str) -> None:
    # With catastrophic cancellation the summation order changes the result even
    # after rounding to float32; sorting by source makes it reproducible.
    updates = [c("e1", w=[1e20]), c("e2", w=[1.0]), c("e3", w=[-1e20])]
    results = {
        agg(name, list(order)).state["w"].tobytes()
        for order in itertools.permutations(updates)
    }

    assert len(results) == 1


@pytest.mark.parametrize("name, params", ALL)
def test_dtype_and_shape_are_preserved(name: str, params) -> None:
    out = agg(name, _updates(), params)

    assert out.state["trunk.w"].dtype == np.float32
    assert out.state["trunk.w"].shape == (6,)


@pytest.mark.parametrize("name, params", ALL)
def test_shape_mismatches_are_rejected(name: str, params) -> None:
    with pytest.raises(AggregationError, match="trunk.w"):
        agg(name, [c("e1", trunk__w=[1.0]), c("e2", trunk__w=[1.0, 2.0])], params)


@pytest.mark.parametrize("name, params", ALL)
def test_nothing_to_aggregate_is_an_error(name: str, params) -> None:
    with pytest.raises(AggregationError, match="no contributions"):
        agg(name, [], params)


def test_hierarchical_fedavg_equals_flat_fedavg() -> None:
    updates = _updates()
    # edges 0-1 under fog_a, 2-4 under fog_b; edge_4 also carries a private adapter
    updates[4] = Contribution(
        source="edge_4",
        state={**updates[4].state, "adapter.x.w": np.full(2, 7.0, np.float32)},
        weights={**updates[4].weights, "adapter.x.w": 9},
    )
    fedavg = aggregators.create("fedavg")
    fog_a = fedavg.aggregate(updates[:2], source="fog_a")
    fog_b = fedavg.aggregate(updates[2:], source="fog_b")

    hierarchical = fedavg.aggregate([fog_a, fog_b], source="cloud")
    flat = fedavg.aggregate(updates, source="cloud")

    assert hierarchical.weights == flat.weights
    for key, value in flat.state.items():
        np.testing.assert_allclose(hierarchical.state[key], value, rtol=1e-6, atol=1e-7)


# --- server optimizers ---------------------------------------------------------------


def g(**state: list[float]) -> dict[str, np.ndarray]:
    return {key: np.asarray(v, dtype=np.float32) for key, v in state.items()}


def test_replace_takes_the_aggregate_and_keeps_untouched_keys() -> None:
    opt = server_optimizers.create("replace")

    new = opt.apply(g(w=[1.0], h=[5.0]), g(w=[3.0]))

    np.testing.assert_allclose(new["w"], [3.0])
    np.testing.assert_allclose(new["h"], [5.0])


def test_fedavgm_without_momentum_and_unit_lr_is_replace() -> None:
    opt = server_optimizers.create("fedavgm", {"server_lr": 1.0, "momentum": 0.0})

    np.testing.assert_allclose(opt.apply(g(w=[1.0]), g(w=[3.0]))["w"], [3.0])


def test_fedavgm_accumulates_momentum_across_rounds() -> None:
    opt = server_optimizers.create("fedavgm", {"server_lr": 1.0, "momentum": 0.5})

    first = opt.apply(g(w=[0.0]), g(w=[1.0]))  # v = 1        -> w = 1
    second = opt.apply(first, g(w=[2.0]))  # v = 0.5 + 1 -> w = 2.5

    np.testing.assert_allclose(first["w"], [1.0])
    np.testing.assert_allclose(second["w"], [2.5])


def test_fedadam_follows_its_update_rule() -> None:
    opt = server_optimizers.create(
        "fedadam", {"server_lr": 0.1, "beta1": 0.9, "beta2": 0.99, "tau": 0.001}
    )

    new = opt.apply(g(w=[0.0]), g(w=[2.0]))

    delta = 2.0
    m = 0.1 * delta
    v = 0.01 * delta**2
    np.testing.assert_allclose(new["w"], [0.1 * m / (np.sqrt(v) + 0.001)], rtol=1e-6)


def test_optimizers_keep_the_dtype() -> None:
    for name in server_optimizers.names():
        stats = {"train_steps": 1.0, "train_edges": 1.0, "edges_total": 1.0}
        optimizer = server_optimizers.create(name)
        out = optimizer.apply(g(w=[0.0, 1.0]), g(w=[1.0, 2.0]), stats)
        assert out["w"].dtype == np.float32, name


def test_registries_list_the_built_ins() -> None:
    assert aggregators.names() == ["fedavg", "mean", "median", "trimmed_mean"]
    assert server_optimizers.names() == ["fedadam", "fedavgm", "fednova", "replace"]


@pytest.mark.parametrize("name", ["fedavgm", "fedadam"])
def test_server_optimizers_replace_auxiliary_arrays(name: str) -> None:
    optimizer = server_optimizers.create(name)
    global_state = {"w": np.zeros(2), "scaffold/w": np.zeros(2)}
    aggregated = {"w": np.ones(2), "scaffold/w": np.full(2, 7.0)}

    out = optimizer.apply(global_state, aggregated)
    out = optimizer.apply(out, aggregated)  # a second step: momentum would show

    np.testing.assert_array_equal(out["scaffold/w"], [7.0, 7.0])


def test_fednova_with_equal_steps_is_fedavg() -> None:
    x = np.zeros(2)
    ys = [np.array([1.0, 2.0]), np.array([3.0, 6.0])]
    d = np.mean([(y - x) / 4 for y in ys], axis=0)

    out = server_optimizers.create("fednova").apply(
        {"w": x}, {"w": np.mean(ys, axis=0), "fednova/w": d}, {"train_steps": 4.0}
    )

    np.testing.assert_allclose(out["w"], np.mean(ys, axis=0))
    assert "fednova/w" not in out


def test_fednova_scales_the_normalised_update_by_the_mean_steps() -> None:
    out = server_optimizers.create("fednova").apply(
        {"w": np.ones(1)},
        {"w": np.full(1, 9.0), "fednova/w": np.full(1, 0.5)},
        {"train_steps": 3.0},
    )

    np.testing.assert_allclose(out["w"], [2.5])  # 1 + 3·0.5


def test_fednova_names_the_missing_steps() -> None:
    with pytest.raises(ValueError, match="train_steps"):
        server_optimizers.create("fednova").apply(
            {"w": np.zeros(1)}, {"w": np.zeros(1), "fednova/w": np.zeros(1)}, {}
        )
