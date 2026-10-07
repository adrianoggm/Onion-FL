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
    assert aggregators.names() == [
        "bulyan",
        "dp_fedavg",
        "fedavg",
        "geometric_median",
        "krum",
        "mean",
        "median",
        "multi_krum",
        "norm_clip",
        "trimmed_mean",
    ]
    assert server_optimizers.names() == [
        "fedadagrad",
        "fedadam",
        "fedasync_mix",
        "fedavgm",
        "feddyn",
        "fednova",
        "fedyogi",
        "replace",
        "scaffold",
    ]


@pytest.mark.parametrize(
    "name", ["fedavgm", "fedadam", "fedyogi", "fedadagrad", "fedasync_mix"]
)
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


def test_feddyn_moves_the_global_model_against_its_drift_term() -> None:
    optimizer = server_optimizers.create("feddyn", {"alpha": 0.5})
    stats = {"train_edges": 2.0, "edges_total": 4.0}

    first = optimizer.apply({"w": np.zeros(1)}, {"w": np.full(1, 2.0)}, stats)
    # h = 0 − 0.5·(2/4)·(2 − 0) = −0.5 ; w = 2 − h/0.5 = 3
    np.testing.assert_allclose(first["w"], [3.0])

    second = optimizer.apply(first, {"w": np.full(1, 3.0)}, stats)
    # h = −0.5 − 0.5·0.5·(3 − 3) = −0.5 ; w = 3 + 1 = 4
    np.testing.assert_allclose(second["w"], [4.0])


def test_fednova_scales_each_key_by_the_steps_of_its_holders() -> None:
    out = server_optimizers.create("fednova").apply(
        {"a": np.zeros(1), "b": np.zeros(1)},
        {
            "a": np.zeros(1),
            "fednova/a": np.full(1, 0.5),
            "fednova_steps/a": np.full(1, 20.0),
            "b": np.zeros(1),
            "fednova/b": np.full(1, 0.5),
            "fednova_steps/b": np.full(1, 40.0),
        },
        {"train_steps": 35.0},  # the round mean: right for neither key
    )

    np.testing.assert_allclose(out["a"], [10.0])
    np.testing.assert_allclose(out["b"], [20.0])
    assert not [k for k in out if k.startswith("fednova")]


def vec(source: str, w, n: float = 1.0, **extra) -> Contribution:
    state = {"w": np.asarray(w, dtype=np.float64)}
    state |= {k: np.asarray(v, dtype=np.float64) for k, v in extra.items()}
    return Contribution(source, state, dict.fromkeys(state, n))


HONEST = [vec(f"h{i}", [1.0 + 0.1 * i, 1.0]) for i in range(5)]
BYZANTINE = [vec("b0", [50.0, -50.0]), vec("b1", [60.0, -40.0])]


def test_krum_picks_an_honest_update() -> None:
    krum = aggregators.create("krum", {"f": 2})

    out = krum.aggregate(HONEST + BYZANTINE, "fog")

    assert out.state["w"][0] < 2.0
    (_, _, tags), _ = krum.report()  # the selection, then the scored keys
    assert {"b0", "b1"} <= set(tags["dropped"]) and tags["f_used"] == 2


def test_multi_krum_averages_the_m_best() -> None:
    multi = aggregators.create("multi_krum", {"f": 2, "m": 5})

    out = multi.aggregate(HONEST + BYZANTINE, "fog")

    expected = np.mean([h.state["w"] for h in HONEST], axis=0)
    np.testing.assert_allclose(out.state["w"], expected)


def test_too_few_children_lower_f_instead_of_failing() -> None:
    krum = aggregators.create("krum", {"f": 2})

    krum.aggregate(HONEST[:3], "fog")  # n=3 fits f=0 only (n >= 2f + 3)

    assert krum.report()[0][2]["f_used"] == 0


def test_the_geometric_median_resists_an_outlier() -> None:
    points = [
        vec("a", [0.0, 0.0]),
        vec("b", [1.0, 0.0]),
        vec("c", [0.0, 1.0]),
        vec("d", [100.0, 100.0]),
    ]

    out = aggregators.create("geometric_median").aggregate(points, "fog")

    assert np.linalg.norm(out.state["w"]) < 1.0


def test_bulyan_bounds_an_extreme_coordinate() -> None:
    children = [*HONEST, vec("h5", [1.2, 1.0]), vec("b0", [1000.0, 1.0])]  # n=7: f=1

    bulyan = aggregators.create("bulyan", {"f": 1})
    out = bulyan.aggregate(children, "fog")

    assert out.state["w"][0] < 2.0
    assert bulyan.report()[0][2]["f_used"] == 1


def test_selection_scores_common_keys_and_combines_every_key() -> None:
    children = [
        vec("s0", [1.0], adapter_s=[1.0]),
        vec("s1", [1.1], adapter_s=[3.0]),
        vec("t0", [0.9], adapter_t=[5.0]),
        vec("t1", [90.0], adapter_t=[7.0]),
    ]

    out = aggregators.create("multi_krum", {"f": 1, "m": 3}).aggregate(children, "fog")

    np.testing.assert_allclose(out.state["adapter_s"], [2.0])  # both holders kept
    np.testing.assert_allclose(out.state["adapter_t"], [5.0])  # only the honest one


def test_auxiliary_arrays_follow_the_selected_children_unscored() -> None:
    children = [*HONEST, vec("b0", [50.0, -50.0])]
    for child in children:
        child.state["scaffold/w"] = np.full(2, 500.0 if child.source == "b0" else 1.0)
        child.weights["scaffold/w"] = 1.0

    out = aggregators.create("multi_krum", {"f": 1, "m": 5}).aggregate(children, "fog")

    np.testing.assert_allclose(out.state["scaffold/w"], [1.0, 1.0])


def test_norm_clip_bounds_each_update_against_the_reference() -> None:
    reference = {"w": np.zeros(2)}
    children = [vec("a", [3.0, 4.0]), vec("b", [0.3, 0.4])]  # norms 5 and 0.5
    clip = aggregators.create("norm_clip", {"bound": 1.0})

    out = clip.aggregate(children, "fog", reference=reference)

    np.testing.assert_allclose(out.state["w"], [(0.6 + 0.3) / 2, (0.8 + 0.4) / 2])
    assert clip.report() == [("diagnostic.clipped", 1.0, {})]


def test_dp_fedavg_adds_seeded_noise_and_reports_epsilon() -> None:
    reference = {"w": np.zeros(3)}
    children = [vec("a", [1.0, 1.0, 1.0]), vec("b", [1.0, 1.0, 1.0])]
    params = {"clip": 10.0, "sigma": 1.0, "delta": 1e-5}

    dp = aggregators.create("dp_fedavg", params)
    first = dp.aggregate(
        children, "fog", reference=reference, rng=np.random.default_rng(0)
    )
    second = aggregators.create("dp_fedavg", params).aggregate(
        children, "fog", reference=reference, rng=np.random.default_rng(0)
    )

    np.testing.assert_array_equal(first.state["w"], second.state["w"])
    assert not np.allclose(first.state["w"], [1.0, 1.0, 1.0])  # noise σ·C/m = 5
    ((name, value, tags),) = dp.report()
    assert name == "diagnostic.privacy_epsilon" and tags == {"mechanism": "central"}
    assert value == pytest.approx(5.298, abs=0.01)  # one round at σ=1


def test_dp_fedavg_composes_its_budget_across_a_change_of_sigma() -> None:
    from onion_fl.learning.privacy import ORDERS

    reference = {"w": np.zeros(3)}
    children = [vec("a", [1.0, 1.0, 1.0]), vec("b", [1.0, 1.0, 1.0])]
    before = aggregators.create("dp_fedavg", {"clip": 10.0, "sigma": 0.5})
    before.aggregate(children, "fog", reference=reference, rng=np.random.default_rng(0))
    after = aggregators.create("dp_fedavg", {"clip": 10.0, "sigma": 1.0})
    after.load_state(before.state())
    after.aggregate(children, "fog", reference=reference, rng=np.random.default_rng(0))

    rdp = ORDERS / (2 * 0.5**2) + ORDERS / (2 * 1.0**2)
    expected = float((rdp + np.log(1e5) / (ORDERS - 1)).min())
    ((_, value, _),) = after.report()
    assert value == pytest.approx(expected)


def test_dp_fedavg_leaves_auxiliary_arrays_unclipped_and_unnoised() -> None:
    reference = {"w": np.zeros(1), "scaffold/w": np.zeros(1)}
    children = [vec("a", [1.0]), vec("b", [1.0])]
    for child in children:
        child.state["scaffold/w"] = np.full(1, 9.0)
        child.weights["scaffold/w"] = 1.0

    out = aggregators.create("dp_fedavg", {"clip": 0.1, "sigma": 1.0}).aggregate(
        children, "fog", reference=reference, rng=np.random.default_rng(0)
    )

    np.testing.assert_allclose(out.state["scaffold/w"], [9.0])


SWELL_ONLY = [
    vec(f"s{i}", [1.0 + 0.02 * i]) for i in range(6)
]  # 7 children: Bulyan keeps f = 1
OUTLIER_WITH_ITS_OWN_KEY = vec("t0", [9.0], adapter_t=[7.0])


@pytest.mark.parametrize("name", ["krum", "multi_krum", "bulyan"])
def test_a_key_whose_holders_were_all_dropped_comes_from_its_best_holder(
    name: str,
) -> None:
    selector = aggregators.create(name, {"f": 1})

    out = selector.aggregate([*SWELL_ONLY, OUTLIER_WITH_ITS_OWN_KEY], "fog")

    report = selector.report()[0][2]
    assert "t0" in report["dropped"]
    np.testing.assert_allclose(out.state["adapter_t"], [7.0])
    # t0 lost the selection, but its own key still shaped the result.
    assert report["rescued"] == {"adapter_t": ["t0"]}
    assert "t0" not in report["excluded"]


def test_bulyan_averages_auxiliary_arrays_of_the_selected_children() -> None:
    children = [vec(f"h{i}", [1.0 + 0.01 * i]) for i in range(7)]
    for i, child in enumerate(children):
        child.state["scaffold/w"] = np.full(1, float(i**2))  # skewed: mean != median
        child.weights["scaffold/w"] = 1.0
    bulyan = aggregators.create("bulyan", {"f": 1})

    out = bulyan.aggregate(children, "fog")

    kept = [c for c in children if c.source not in bulyan.report()[0][2]["dropped"]]
    expected = np.mean([c.state["scaffold/w"] for c in kept])
    np.testing.assert_allclose(out.state["scaffold/w"], [expected])


def test_a_selection_reports_before_any_round() -> None:
    assert aggregators.create("krum").report()[0][1] == 0.0


@pytest.mark.parametrize("name", ["krum", "multi_krum", "bulyan", "geometric_median"])
def test_a_robust_aggregator_reports_how_many_keys_it_scored(name: str) -> None:
    # Two datasets' adapters under independent sharing: nothing in common.
    apart = [c(f"e{i}", **{f"adapter__{d}": [float(i)]}) for i, d in enumerate("abab")]
    together = [c(f"e{i}", w=[float(i)]) for i in range(4)]
    robust = aggregators.create(name)

    scored = []
    for children in (apart, together):
        robust.aggregate(children, "fog")
        scored += [v for n, v, _ in robust.report() if n == "diagnostic.scored_keys"]

    assert scored == [0.0, 1.0]  # with 0 the scores cannot tell children apart


def test_dp_fedavg_refuses_to_noise_without_a_stream() -> None:
    dp = aggregators.create("dp_fedavg")

    # A fixed fallback stream would repeat the same noise every round.
    with pytest.raises(ValueError, match="random stream"):
        dp.aggregate(HONEST, "fog", reference={"w": np.zeros(2)})


def test_scaffold_moves_c_by_the_share_of_its_holders_that_trained() -> None:
    # Two holders, one trained with Δc = 2: c = 0 + (1/2)·2 (Karimireddy et al.).
    out = server_optimizers.create("scaffold").apply(
        {"trunk.0.weight": np.zeros(1), "scaffold/trunk.0.weight": np.zeros(1)},
        {"trunk.0.weight": np.full(1, 5.0), "scaffold/trunk.0.weight": np.full(1, 2.0)},
        {"train_edges/trunk": 1.0, "edges_total/trunk": 2.0},
    )

    np.testing.assert_allclose(out["scaffold/trunk.0.weight"], [1.0])
    np.testing.assert_allclose(out["trunk.0.weight"], [5.0])  # the model: replaced


def test_scaffold_accumulates_c_across_rounds() -> None:
    optimizer = server_optimizers.create("scaffold")
    stats = {"train_edges/trunk": 2.0, "edges_total/trunk": 2.0}
    state = {"trunk.0.weight": np.zeros(1)}

    for delta in (1.0, 3.0):
        aggregated = {
            "trunk.0.weight": np.zeros(1),
            "scaffold/trunk.0.weight": np.full(1, delta),
        }
        state = optimizer.apply(state, aggregated, stats)

    np.testing.assert_allclose(state["scaffold/trunk.0.weight"], [4.0])


def test_feddyn_uses_the_share_of_each_keys_holders() -> None:
    optimizer = server_optimizers.create("feddyn", {"alpha": 0.5})
    stats = {
        "train_edges": 3.0,
        "edges_total": 4.0,
        "train_edges/adapter.a": 1.0,
        "edges_total/adapter.a": 2.0,
    }

    out = optimizer.apply(
        {"adapter.a.0.weight": np.zeros(1)},
        {"adapter.a.0.weight": np.full(1, 2.0)},
        stats,
    )

    # h = −0.5·(1/2)·2 = −0.5 ; w = 2 + 1 = 3 (the round's 3/4 would give 3.5)
    np.testing.assert_allclose(out["adapter.a.0.weight"], [3.0])


@pytest.mark.parametrize("name", ["krum", "multi_krum", "bulyan"])
def test_a_selection_names_the_children_it_excluded_entirely(name: str) -> None:
    selector = aggregators.create(name, {"f": 1})

    selector.aggregate([*SWELL_ONLY, vec("t0", [9.0])], "fog")

    report = selector.report()[0][2]
    assert "t0" in report["dropped"] and "t0" in report["excluded"]
    assert report["rescued"] == {}


def test_bulyan_selects_recursively_with_krum() -> None:
    # Krum (4 nearest of 7, then 3 of 6, 2 of 5, 1...) picks 3, 2, 8, 0, 7 one at a
    # time; a single Krum ranking would keep 1 and drop 8 instead.
    values = [0.0, 1.0, 2.0, 3.0, 7.0, 8.0, 10.0]
    children = [vec(f"c{i}", [v]) for i, v in enumerate(values)]
    bulyan = aggregators.create("bulyan", {"f": 1})

    bulyan.aggregate(children, "fog")

    assert bulyan.report()[0][2]["dropped"] == ["c1", "c6"]


ADAPTIVE = {"server_lr": 0.1, "beta1": 0.9, "tau": 0.001}


def test_fedadagrad_accumulates_squared_pseudo_gradients() -> None:
    # Reddi et al. (2021), Algorithm 2: v starts at tau², then v += Δ².
    opt = server_optimizers.create("fedadagrad", ADAPTIVE)

    first = opt.apply(g(w=[0.0]), g(w=[2.0]))
    second = opt.apply(first, first | g(w=[float(first["w"][0]) + 1.0]))

    m1, v1 = 0.1 * 2.0, 1e-6 + 2.0**2
    x1 = 0.1 * m1 / (np.sqrt(v1) + 0.001)
    np.testing.assert_allclose(first["w"], [x1], rtol=1e-6)
    m2, v2 = 0.9 * m1 + 0.1 * 1.0, v1 + 1.0**2
    np.testing.assert_allclose(
        second["w"], [x1 + 0.1 * m2 / (np.sqrt(v2) + 0.001)], rtol=1e-5
    )


def test_fedyogi_moves_v_additively_by_the_sign_of_its_gap() -> None:
    # v ← v − (1 − β2)·Δ²·sign(v − Δ²): up by 0.01·Δ² while v < Δ², then down.
    opt = server_optimizers.create("fedyogi", ADAPTIVE | {"beta2": 0.99})

    first = opt.apply(g(w=[0.0]), g(w=[2.0]))
    second = opt.apply(first, first | g(w=[float(first["w"][0]) + 0.1]))

    m1, v1 = 0.1 * 2.0, 1e-6 + 0.01 * 4.0  # v < Δ²: grows
    x1 = 0.1 * m1 / (np.sqrt(v1) + 0.001)
    np.testing.assert_allclose(first["w"], [x1], rtol=1e-6)
    m2, v2 = 0.9 * m1 + 0.1 * 0.1, v1 - 0.01 * 0.01  # v > Δ²: shrinks by 0.01·Δ²
    np.testing.assert_allclose(
        second["w"], [x1 + 0.1 * m2 / (np.sqrt(v2) + 0.001)], rtol=1e-5
    )


def test_fedasync_mixes_by_a_staleness_discounted_weight() -> None:
    opt = server_optimizers.create("fedasync_mix", {"alpha": 0.5, "a": 0.5})

    fresh = opt.apply(g(w=[0.0]), g(w=[4.0]))
    stale = opt.apply(g(w=[0.0]), g(w=[4.0]), {"staleness": 3.0})

    np.testing.assert_allclose(fresh["w"], [2.0])  # α_s = 0.5
    np.testing.assert_allclose(stale["w"], [1.0])  # α_s = 0.5·(1 + 3)^-0.5 = 0.25


@pytest.mark.parametrize(
    "name, params",
    [
        ("fedavgm", {"server_lr": 1.0, "momentum": 0.5}),
        ("fedadam", {"server_lr": 0.1}),
        ("fedyogi", {"server_lr": 0.1}),
        ("fedadagrad", {"server_lr": 0.1}),
        ("feddyn", {"alpha": 0.5}),
    ],
)
def test_server_optimizer_state_survives_save_and_load(name: str, params: dict) -> None:
    stats = {"train_edges": 2.0, "edges_total": 4.0}
    one = server_optimizers.create(name, params)
    first = one.apply(g(w=[0.0, 1.0]), g(w=[2.0, 3.0]), stats)

    two = server_optimizers.create(name, params)
    two.load_state(one.state())

    np.testing.assert_array_equal(
        one.apply(first, g(w=[1.0, 5.0]), stats)["w"],
        two.apply(first, g(w=[1.0, 5.0]), stats)["w"],
    )
    assert all(isinstance(v, np.ndarray) for v in one.state().values())


def test_stateless_server_optimizers_have_an_empty_state() -> None:
    for name in ("replace", "fednova", "scaffold", "fedasync_mix"):
        assert server_optimizers.create(name).state() == {}
