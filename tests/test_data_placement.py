"""Tests for the placement plugins and the scenario composition (issue #89).

Subjects are hand-picked arrays to check bookkeeping; nothing is trained.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from onion_fl.core.registry import PluginError
from onion_fl.core.topology import parse_topology
from onion_fl.data.contract import DataError, SubjectData
from onion_fl.data.placement import largest_remainder, place, placements
from onion_fl.data.roles import RolesConfig, split_subjects


def subject(dataset: str, name: str, rows: int = 2, label: int | None = None):
    labels = [label] * rows if label is not None else [i % 2 for i in range(rows)]
    return SubjectData(
        X=np.arange(rows, dtype=np.float32).reshape(-1, 1),
        y=np.asarray(labels),
        dataset=dataset,
        subject=name,
        task="stress",
        n_classes=2,
        feature_names=["f0"],
    )


def split(n_swell: int = 8, n_sweet: int = 8, val: float = 0.0, rows: int = 2):
    subjects = [subject("swell", str(i), rows) for i in range(1, n_swell + 1)]
    subjects += [subject("sweet", f"u{i}", rows) for i in range(1, n_sweet + 1)]
    config = RolesConfig(test=0.0, val=val, scaler="none", drop_constant=False)
    return split_subjects(subjects, config)


def topology(homes: list[str | None], parent_home: str | None = None):
    fog = {
        "nodes": [
            {"id": f"fog_{i}"} | ({"home": h} if h else {}) for i, h in enumerate(homes)
        ]
    }
    root = {"id": "cloud"} | ({"home": parent_home} if parent_home else {})
    return parse_topology(
        {"name": "t", "levels": ["global", "fog", "edge"], "root": root, "fog": fog}
    )


FOUR = ["swell", "swell", "sweet", "sweet"]


def datasets_per_leaf(placement) -> dict[str, Counter]:
    return {
        leaf: Counter(c.dataset for c in clients)
        for leaf, clients in placement.edges.items()
    }


# --- mixing ---------------------------------------------------------------------------


def test_alpha_zero_segregates_by_home_dataset() -> None:
    placement = place(split(), topology(FOUR), "mixing", {"alpha": 0.0})

    mix = datasets_per_leaf(placement)
    assert mix["fog_0"] == Counter(swell=4) and mix["fog_1"] == Counter(swell=4)
    assert mix["fog_2"] == Counter(sweet=4) and mix["fog_3"] == Counter(sweet=4)


def test_alpha_one_gives_every_leaf_the_same_share_of_each_dataset() -> None:
    placement = place(split(), topology(FOUR), "mixing", {"alpha": 1.0})

    for counts in datasets_per_leaf(placement).values():
        assert counts == Counter(swell=2, sweet=2)


def test_intermediate_alpha_follows_the_weights() -> None:
    # w_home = 0.5 * 1/2 + 0.5/4 = 0.375, w_other = 0.125 -> of 8: 3, 3, 1, 1
    placement = place(split(), topology(FOUR), "mixing", {"alpha": 0.5})

    swell = [datasets_per_leaf(placement)[f"fog_{i}"]["swell"] for i in range(4)]
    assert swell == [3, 3, 1, 1]


def test_a_dataset_without_home_leaves_is_spread_uniformly() -> None:
    placement = place(split(), topology(["swell"] * 4), "mixing", {"alpha": 0.0})

    for counts in datasets_per_leaf(placement).values():
        assert counts["sweet"] == 2 and counts["swell"] == 2


def test_homes_are_inherited_from_ancestors() -> None:
    placement = place(
        split(n_sweet=0),
        topology([None, None], parent_home="swell"),
        "mixing",
        {"alpha": 0.0},
    )

    assert all(c["swell"] == 4 for c in datasets_per_leaf(placement).values())


def test_missing_homes_are_assigned_in_turns() -> None:
    placement = place(split(), topology([None] * 4), "mixing", {"alpha": 0.0})

    mix = datasets_per_leaf(placement)
    assert [set(mix[f"fog_{i}"]) for i in range(4)] == [
        {"sweet"},
        {"swell"},
        {"sweet"},
        {"swell"},
    ]


def test_a_home_dataset_that_is_not_loaded_is_an_error() -> None:
    with pytest.raises(DataError, match="wesad"):
        place(split(), topology(["swell", "wesad"]), "mixing", {"alpha": 0.0})


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_alpha_is_validated(alpha: float) -> None:
    with pytest.raises(PluginError, match="alpha"):
        placements.create("mixing", {"alpha": alpha})


# --- guarantees shared by every plugin -------------------------------------------------

ALL = [
    ("mixing", {"alpha": 0.3}),
    ("dirichlet", {"beta": 0.5}),
    ("label_skew", {"beta": 0.5}),
    ("pooled", {}),
]


@pytest.mark.parametrize("name, params", ALL)
def test_no_subject_is_lost_or_duplicated(name: str, params: dict) -> None:
    data = split(n_swell=7, n_sweet=5, val=0.3)

    placement = place(data, topology(FOUR), name, params, seed=1)

    placed = [
        s for clients in placement.edges.values() for c in clients for s in c.subjects
    ]
    assert Counter(placed) == Counter(s for c in data.clients for s in c.subjects)
    evaluators = [e.subject for es in placement.zone_evaluators.values() for e in es]
    assert sorted(evaluators) == sorted(e.subject for e in data.val)


@pytest.mark.parametrize("name, params", ALL)
def test_placement_is_deterministic(name: str, params: dict) -> None:
    a = place(split(), topology(FOUR), name, params, seed=4)
    b = place(split(), topology(FOUR), name, params, seed=4)

    assert a.describe() == b.describe()


def test_the_seed_changes_a_random_placement() -> None:
    a = place(split(), topology(FOUR), "dirichlet", {"beta": 0.5}, seed=0)
    b = place(split(), topology(FOUR), "dirichlet", {"beta": 0.5}, seed=1)

    assert a.describe() != b.describe()


def test_leaf_order_in_the_file_does_not_matter() -> None:
    a = topology(FOUR)
    raw = {
        "name": "t",
        "levels": ["global", "fog", "edge"],
        "root": {"id": "cloud"},
        "fog": {"nodes": [{"id": f"fog_{i}", "home": FOUR[i]} for i in (3, 1, 0, 2)]},
    }
    b = parse_topology(raw)

    assert a.topology_id == b.topology_id
    assert (
        place(split(), a, "dirichlet", {"beta": 1.0}).describe()
        == place(split(), b, "dirichlet", {"beta": 1.0}).describe()
    )


# --- dirichlet and label skew -----------------------------------------------------------


def test_dirichlet_with_a_large_beta_is_close_to_uniform() -> None:
    placement = place(split(40, 40), topology(FOUR), "dirichlet", {"beta": 1000.0})

    for counts in datasets_per_leaf(placement).values():
        assert 7 <= counts["swell"] <= 13 and 7 <= counts["sweet"] <= 13


def test_dirichlet_with_a_small_beta_concentrates_each_dataset() -> None:
    placement = place(
        split(40, 40), topology(FOUR), "dirichlet", {"beta": 0.01}, seed=2
    )

    largest = max(c["swell"] for c in datasets_per_leaf(placement).values())
    assert largest >= 30


@pytest.mark.parametrize("name", ["dirichlet", "label_skew"])
def test_beta_must_be_positive(name: str) -> None:
    with pytest.raises(PluginError, match="beta"):
        placements.create(name, {"beta": 0})


def test_label_skew_separates_classes_when_targets_are_extreme() -> None:
    subjects = [subject("swell", f"a{i}", 4, label=0) for i in range(6)]
    subjects += [subject("swell", f"b{i}", 4, label=1) for i in range(6)]
    data = split_subjects(
        subjects, RolesConfig(test=0.0, scaler="none", drop_constant=False)
    )

    placement = place(
        data, topology([None, None]), "label_skew", {"beta": 0.05}, seed=3
    )

    purity = [
        max(counts) / sum(counts)
        for counts in (
            np.sum([c.train.class_counts for c in clients], axis=0)
            for clients in placement.edges.values()
        )
    ]
    assert min(purity) >= 0.75


def test_label_skew_keeps_leaves_balanced_in_size() -> None:
    placement = place(
        split(12, 0, rows=4), topology([None] * 4), "label_skew", {"beta": 0.1}
    )

    sizes = [sum(c.train.n_samples for c in cs) for cs in placement.edges.values()]
    assert max(sizes) - min(sizes) <= 4


# --- pooled and explicit ---------------------------------------------------------------------


def test_pooled_puts_each_dataset_in_one_edge() -> None:
    data = split(val=0.25)

    placement = place(data, topology(FOUR), "pooled")

    (leaf,) = [leaf for leaf, cs in placement.edges.items() if cs]
    assert leaf == "fog_0"
    edges = placement.edges[leaf]
    assert [c.id for c in edges] == ["sweet-pooled", "swell-pooled"]
    swell = edges[1]
    assert swell.train.n_samples == sum(
        c.train.n_samples for c in data.clients if c.dataset == "swell"
    )
    assert len(placement.zone_evaluators["fog_0"]) == len(data.val)


def test_pooled_can_name_its_leaf() -> None:
    placement = place(split(), topology(FOUR), "pooled", {"leaf": "fog_2"})

    assert [leaf for leaf, cs in placement.edges.items() if cs] == ["fog_2"]


def test_explicit_lists_per_leaf() -> None:
    data = split(n_swell=2, n_sweet=2)
    ids = [c.id for c in data.clients]

    placement = place(
        data,
        topology([None, None]),
        "explicit",
        {"assignment": {"fog_0": ids[:3], "fog_1": ids[3:]}},
    )

    assert [c.id for c in placement.edges["fog_0"]] == ids[:3]


@pytest.mark.parametrize(
    "assignment, message",
    [
        ({"fog_0": ["swell-1", "swell-2", "sweet-u1"]}, "sweet-u2"),
        ({"fog_0": ["swell-1", "swell-2", "sweet-u1", "sweet-u2", "swell-1"]}, "twice"),
        ({"fog_0": ["swell-1", "swell-2", "sweet-u1", "sweet-u2", "ghost"]}, "ghost"),
        ({"fog_9": ["swell-1", "swell-2", "sweet-u1", "sweet-u2"]}, "fog_9"),
    ],
)
def test_explicit_assignments_are_checked(assignment: dict, message: str) -> None:
    with pytest.raises(DataError, match=message):
        place(
            split(2, 2), topology([None, None]), "explicit", {"assignment": assignment}
        )


# --- largest remainder and composition ------------------------------------------------------


def test_largest_remainder_keeps_the_total_and_breaks_ties_by_rng() -> None:
    counts = {
        tuple(largest_remainder(np.array([0.5, 0.5]), 5, np.random.default_rng(s)))
        for s in range(20)
    }

    assert counts == {(3, 2), (2, 3)}
    assert largest_remainder(
        np.array([0.7, 0.2, 0.1]), 10, np.random.default_rng(0)
    ).tolist() == [7, 2, 1]


def test_composition_reports_samples_classes_and_mix_entropy() -> None:
    segregated = place(split(), topology(FOUR), "mixing", {"alpha": 0.0}).composition()
    uniform = place(split(), topology(FOUR), "mixing", {"alpha": 1.0}).composition()

    leaf = segregated["fog_0"]
    assert leaf["clients"] == 4 and leaf["subjects"] == 4 and leaf["samples"] == 8
    assert leaf["class_counts"] == [4, 4]
    assert leaf["datasets"] == {"swell": 8}
    assert leaf["entropy"] == 0.0
    assert uniform["fog_0"]["entropy"] == pytest.approx(1.0)


def test_describe_lists_the_assignment() -> None:
    placement = place(
        split(2, 2, val=0.5), topology([None, None]), "mixing", {"alpha": 1.0}
    )

    described = placement.describe()

    assert described["placement"] == "mixing"
    assert set(described["edges"]) == {"fog_0", "fog_1"}
    assert set(described["zone_evaluators"]) == {"fog_0", "fog_1"}
    assert described["composition"] == placement.composition()


def test_home_weights_are_shared_among_the_home_leaves() -> None:
    # w_home = 0.5 / 2 + 0.5 / 4 = 0.375 and w_other = 0.125: of 24, 9 9 3 3
    placement = place(split(24, 24), topology(FOUR), "mixing", {"alpha": 0.5})

    swell = [datasets_per_leaf(placement)[f"fog_{i}"]["swell"] for i in range(4)]
    assert swell == [9, 9, 3, 3]


def test_the_seed_changes_which_subjects_go_where() -> None:
    a = place(split(), topology(FOUR), "mixing", {"alpha": 1.0}, seed=0)
    b = place(split(), topology(FOUR), "mixing", {"alpha": 1.0}, seed=1)

    assert a.describe()["edges"] != b.describe()["edges"]


def test_homes_come_from_the_nearest_ancestor() -> None:
    regions = parse_topology(
        {
            "name": "r",
            "levels": ["global", "region", "fog", "edge"],
            "root": {"id": "cloud"},
            "region": {
                "nodes": [
                    {"id": "north", "home": "swell"},
                    {"id": "south", "home": "sweet"},
                ]
            },
            "fog": {
                "nodes": [
                    {"id": "fog_1", "parent": "north"},
                    {"id": "fog_2", "parent": "north"},
                    {"id": "fog_3", "parent": "south"},
                ]
            },
        }
    )

    mix = datasets_per_leaf(place(split(), regions, "mixing", {"alpha": 0.0}))

    assert set(mix["fog_1"]) == set(mix["fog_2"]) == {"swell"}
    assert set(mix["fog_3"]) == {"sweet"}
