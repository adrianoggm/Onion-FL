"""Tests for subject roles, clients and train-only preprocessing (issue #88).

The arrays are hand-picked numbers to check bookkeeping and arithmetic;
nothing is trained.
"""

from __future__ import annotations

import numpy as np
import pytest

from onion_fl.data.contract import DataError, SubjectData
from onion_fl.data.roles import RolesConfig, split_subjects


def subject(name: str, X, y=None, dataset: str = "swell") -> SubjectData:
    X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    return SubjectData(
        X=X,
        y=np.zeros(len(X), np.int64) if y is None else np.asarray(y),
        dataset=dataset,
        subject=name,
        task="stress",
        n_classes=2,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
    )


def cohort(n: int = 10, dataset: str = "swell") -> list[SubjectData]:
    return [subject(str(i), [[i], [i + 1]], dataset=dataset) for i in range(1, n + 1)]


def roles_of(split, dataset: str = "swell") -> dict[str, list[str]]:
    return split.roles[dataset]


# --- assignment ---------------------------------------------------------------------


def test_proportions_split_the_subjects_without_overlap() -> None:
    split = split_subjects(cohort(10), RolesConfig(test=0.2, val=0.1, scaler="none"))
    roles = roles_of(split)

    assert (len(roles["test"]), len(roles["val"]), len(roles["train"])) == (2, 1, 7)
    every = roles["test"] + roles["val"] + roles["train"]
    assert sorted(every, key=int) == [str(i) for i in range(1, 11)]
    assert [s.subject for s in split.test] == roles["test"]
    assert [s.subject for s in split.val] == roles["val"]


def test_assignment_ignores_the_input_order() -> None:
    config = RolesConfig(test=0.3, val=0.2)
    subjects = cohort(10)

    assert roles_of(split_subjects(subjects, config)) == roles_of(
        split_subjects(subjects[::-1], config)
    )


def test_the_seed_changes_the_assignment() -> None:
    a = roles_of(split_subjects(cohort(10), RolesConfig(test=0.2, seed=0)))
    b = roles_of(split_subjects(cohort(10), RolesConfig(test=0.2, seed=1)))

    assert a["test"] != b["test"]


@pytest.mark.parametrize(
    "scenario",
    [
        {"subjects_per_client": 3},
        {"local_val": 0.5},
        {"scaler": "local"},
        {"scaler": "none", "impute": "median"},
    ],
)
def test_test_subjects_are_the_same_in_every_scenario(scenario: dict) -> None:
    base = RolesConfig(test=0.2, val=0.2, seed=7)
    reference = roles_of(split_subjects(cohort(10), base))

    other = roles_of(split_subjects(cohort(10), base.model_copy(update=scenario)))

    assert other["test"] == reference["test"]
    assert other["val"] == reference["val"]


def test_adding_a_dataset_does_not_move_another_datasets_roles() -> None:
    config = RolesConfig(test=0.2, val=0.1, seed=3)
    alone = roles_of(split_subjects(cohort(10), config))

    mixed = split_subjects(cohort(10) + cohort(6, dataset="sweet"), config)

    assert roles_of(mixed) == alone
    assert len(roles_of(mixed, "sweet")["test"]) == 1


def test_a_positive_proportion_keeps_at_least_one_subject() -> None:
    roles = roles_of(split_subjects(cohort(3), RolesConfig(test=0.01, val=0.0)))

    assert len(roles["test"]) == 1 and roles["val"] == []


def test_explicit_lists_per_dataset() -> None:
    config = RolesConfig(overrides={"swell": {"test": ["3", "1"], "val": ["10"]}})

    roles = roles_of(split_subjects(cohort(10), config))

    assert roles["test"] == ["1", "3"] and roles["val"] == ["10"]
    assert "1" not in roles["train"]


def test_proportions_count_every_subject_of_the_dataset() -> None:
    config = RolesConfig(val=0.5, overrides={"swell": {"test": ["1", "2"]}})

    roles = roles_of(split_subjects(cohort(10), config))

    assert len(roles["val"]) == 5 and not {"1", "2"} & set(roles["val"])


@pytest.mark.parametrize(
    "override, message",
    [
        ({"test": ["99"]}, "99"),
        ({"test": ["1"], "val": ["1"]}, "both"),
        ({"test": 0.5, "val": 0.5}, "no training"),
    ],
)
def test_impossible_assignments_are_rejected(override: dict, message: str) -> None:
    with pytest.raises(DataError, match=message):
        split_subjects(cohort(4), RolesConfig(overrides={"swell": override}))


def test_overrides_must_name_a_known_dataset() -> None:
    with pytest.raises(DataError, match="wesad"):
        split_subjects(cohort(4), RolesConfig(overrides={"wesad": {"test": 0.5}}))


def test_proportions_are_validated() -> None:
    with pytest.raises(ValueError):
        RolesConfig(test=1.0)
    with pytest.raises(ValueError):
        RolesConfig(local_val=1.0)


# --- clients ---------------------------------------------------------------------------


def test_subjects_are_grouped_into_clients() -> None:
    split = split_subjects(
        cohort(7), RolesConfig(test=0.0, subjects_per_client=3, scaler="none")
    )

    assert sorted(len(c.subjects) for c in split.clients) == [1, 3, 3]
    first = next(c for c in split.clients if len(c.subjects) == 3)
    assert first.id == "swell-" + "-".join(first.subjects)
    assert first.dataset == "swell"
    assert first.train.n_samples == 6
    assert sorted(s for c in split.clients for s in c.subjects) == sorted(
        roles_of(split)["train"]
    )


def test_one_subject_per_client_by_default() -> None:
    split = split_subjects(cohort(5), RolesConfig(test=0.0, scaler="none"))

    assert [c.subjects for c in split.clients] == [
        (s,) for s in roles_of(split)["train"]
    ]


def test_local_val_is_the_tail_of_each_subject() -> None:
    subjects = [subject("1", [[1], [2], [3], [4]], [0, 0, 1, 1])]

    (client,) = split_subjects(
        subjects, RolesConfig(test=0.0, local_val=0.25, scaler="none")
    ).clients

    assert client.train.X[:, 0].tolist() == [1, 2, 3]
    assert client.local_val.X[:, 0].tolist() == [4]
    assert client.local_val.y.tolist() == [1]


def test_no_local_val_by_default() -> None:
    (client,) = split_subjects(cohort(1), RolesConfig(test=0.0)).clients

    assert client.local_val is None


def test_class_tail_holds_out_the_last_rows_of_each_class() -> None:
    subjects = [subject("1", [[i] for i in range(1, 11)], [0] * 5 + [1] * 5)]

    (client,) = split_subjects(
        subjects,
        RolesConfig(
            test=0.0, local_val=0.2, local_val_split="class_tail", scaler="none"
        ),
    ).clients

    assert client.local_val.X[:, 0].tolist() == [5, 10]
    assert client.local_val.y.tolist() == [0, 1]
    assert client.train.X[:, 0].tolist() == [1, 2, 3, 4, 6, 7, 8, 9]


def test_class_tail_always_leaves_a_training_row() -> None:
    subjects = [subject("1", [[1], [2]], [0, 1])]

    (client,) = split_subjects(
        subjects,
        RolesConfig(
            test=0.0,
            local_val=0.9,
            local_val_split="class_tail",
            scaler="none",
            drop_constant=False,  # one training row makes every feature constant
        ),
    ).clients

    assert client.train.X[:, 0].tolist() == [1]
    assert client.local_val.X[:, 0].tolist() == [2]


def test_class_tail_with_one_class_is_that_class_tail() -> None:
    subjects = [subject("1", [[1], [2], [3], [4]], [1, 1, 1, 1])]

    (client,) = split_subjects(
        subjects,
        RolesConfig(
            test=0.0, local_val=0.25, local_val_split="class_tail", scaler="none"
        ),
    ).clients

    assert client.local_val.X[:, 0].tolist() == [4]


def test_the_default_split_stays_out_of_the_config() -> None:
    assert "local_val_split" not in RolesConfig().model_dump()
    dumped = RolesConfig(local_val_split="class_tail").model_dump()
    assert dumped["local_val_split"] == "class_tail"


# --- preprocessing fitted on the training bag only ---------------------------------------


def two_way(test_X, scaler: str = "global", local_val: float = 0.0, **extra):
    train = [subject("1", [[0.0], [2.0]]), subject("2", [[4.0], [6.0]])]
    held = [subject("3", test_X)]
    config = RolesConfig(
        scaler=scaler,
        local_val=local_val,
        overrides={"swell": {"test": ["3"]}},
        **extra,
    )
    return split_subjects(train + held, config)


def test_global_scaling_uses_the_training_bag_statistics() -> None:
    split = two_way([[3.0], [8.0]])

    # train bag: mean 3, std sqrt(5)
    np.testing.assert_allclose(split.test[0].X[:, 0], [0.0, 5 / np.sqrt(5)], rtol=1e-6)
    pooled = np.concatenate([c.train.X[:, 0] for c in split.clients])
    np.testing.assert_allclose([pooled.mean(), pooled.std()], [0.0, 1.0], atol=1e-6)


def test_missing_values_take_the_training_bag_mean() -> None:
    split = two_way([[np.nan], [3.0]], scaler="none")

    assert split.test[0].X[:, 0].tolist() == [3.0, 3.0]


def test_median_imputation() -> None:
    train = [subject("1", [[0.0], [1.0], [np.nan]]), subject("2", [[10.0], [1.0]])]

    split = split_subjects(train, RolesConfig(test=0.0, scaler="none", impute="median"))

    assert split.clients[0].train.X[2, 0] == 1.0


def test_local_val_does_not_feed_the_statistics() -> None:
    # The tail of subject 1 (1000) is local validation: it must not shift the mean.
    train = [
        subject("1", [[0.0], [2.0], [1000.0]]),
        subject("2", [[4.0], [6.0], [-5.0]]),
    ]
    held = [subject("3", [[3.0]])]
    config = RolesConfig(
        scaler="global", local_val=0.34, overrides={"swell": {"test": ["3"]}}
    )

    split = split_subjects(train + held, config)

    assert split.test[0].X[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_local_scaling_standardises_each_holder_with_its_own_statistics() -> None:
    split = two_way([[10.0], [20.0]], scaler="local")

    for client in split.clients:
        np.testing.assert_allclose(client.train.X[:, 0], [-1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(split.test[0].X[:, 0], [-1.0, 1.0], rtol=1e-6)


def test_no_scaling_keeps_the_values() -> None:
    split = two_way([[3.0], [8.0]], scaler="none")

    assert split.test[0].X[:, 0].tolist() == [3.0, 8.0]


def test_constant_training_features_are_dropped_everywhere() -> None:
    train = [subject("1", [[0, 5], [2, 5]]), subject("2", [[4, 5], [6, 5]])]
    held = [subject("3", [[1, 7], [2, 9]])]  # varies in test, still constant in train
    config = RolesConfig(scaler="none", overrides={"swell": {"test": ["3"]}})

    split = split_subjects(train + held, config)

    assert split.test[0].feature_names == ["f0"]
    assert all(c.train.feature_names == ["f0"] for c in split.clients)


def test_constant_features_can_be_kept() -> None:
    train = [subject("1", [[0, 5], [2, 5]])]

    split = split_subjects(train, RolesConfig(test=0.0, drop_constant=False))

    assert split.clients[0].train.feature_names == ["f0", "f1"]


def test_datasets_are_preprocessed_separately() -> None:
    swell = [subject("1", [[0.0], [2.0]])]
    sweet = [subject("1", [[100.0], [300.0]], dataset="sweet")]

    split = split_subjects(swell + sweet, RolesConfig(test=0.0))

    for client in split.clients:
        np.testing.assert_allclose(client.train.X[:, 0], [-1.0, 1.0], rtol=1e-6)


def test_describe_lists_roles_and_clients() -> None:
    split = split_subjects(cohort(4), RolesConfig(test=0.25, scaler="none"))

    described = split.describe()

    assert described["roles"] == split.roles
    assert described["clients"][0]["id"] == split.clients[0].id
    assert {"subjects", "samples", "class_counts"} <= set(described["clients"][0])


def test_every_feature_constant_is_an_error() -> None:
    with pytest.raises(DataError, match="constant"):
        split_subjects([subject("1", [[5.0], [5.0]])], RolesConfig(test=0.0))


def test_an_empty_evaluator_is_kept_with_local_scaling() -> None:
    train = [subject("1", [[0.0], [2.0]])]
    empty = SubjectData(
        X=np.zeros((0, 1), np.float32),
        y=np.zeros(0, np.int64),
        dataset="swell",
        subject="2",
        task="stress",
        n_classes=2,
        feature_names=["f0"],
    )
    config = RolesConfig(scaler="local", overrides={"swell": {"test": ["2"]}})

    split = split_subjects(train + [empty], config)

    assert split.test[0].n_samples == 0


def test_excluded_subjects_take_no_role() -> None:
    config = RolesConfig(
        test=0.0, overrides={"swell": {"test": ["1"], "exclude": ["8", "9"]}}
    )

    split = split_subjects(cohort(10), config)
    roles = roles_of(split)

    assert roles["excluded"] == ["8", "9"]
    assert not {"8", "9"} & set(roles["test"] + roles["val"] + roles["train"])
    assert all("8" not in c.subjects for c in split.clients)


def test_excluding_an_unknown_subject_is_an_error() -> None:
    with pytest.raises(DataError, match="99"):
        split_subjects(cohort(4), RolesConfig(overrides={"swell": {"exclude": ["99"]}}))


def test_a_test_subject_cannot_also_be_excluded() -> None:
    config = RolesConfig(overrides={"swell": {"test": ["1"], "exclude": ["1"]}})

    with pytest.raises(DataError, match="1"):
        split_subjects(cohort(4), config)


# --- preprocessing as an artifact ----------------------------------------------------


def test_the_split_records_its_preprocessing_per_dataset() -> None:
    import json

    split = two_way([[3.0], [8.0]])

    prep = split.preprocessing["swell"]
    assert prep["features"] == ["f0"] and prep["scaler"] == "global"
    np.testing.assert_allclose(prep["mean"], [3.0])
    np.testing.assert_allclose(prep["std"], [np.sqrt(5)])
    assert json.loads(json.dumps(split.preprocessing)) == split.preprocessing


def test_a_frozen_preprocessing_is_applied_instead_of_refitted() -> None:
    parent = two_way([[3.0], [8.0]])
    shifted = [subject("1", [[100.0], [102.0]]), subject("2", [[104.0], [106.0]])]
    held = [subject("3", [[3.0], [8.0]])]
    config = RolesConfig(overrides={"swell": {"test": ["3"]}})

    child = split_subjects(shifted + held, config, frozen=parent.preprocessing)

    np.testing.assert_array_equal(child.test[0].X, parent.test[0].X)
    assert child.preprocessing["swell"] == parent.preprocessing["swell"]


def test_a_frozen_preprocessing_keeps_its_features_and_new_datasets_are_fitted() -> (
    None
):
    flat = [
        subject("1", [[0.0, 1.0], [2.0, 1.0]]),
        subject("2", [[4.0, 1.0], [6.0, 1.0]]),
    ]
    parent = split_subjects(flat, RolesConfig(test=0.0))  # f1 is constant: dropped
    varied = [
        subject("1", [[0.0, 5.0], [2.0, 1.0]]),
        subject("2", [[4.0, 9.0], [6.0, 3.0]]),
    ]
    wesad = [subject(str(i), [[i], [i + 2.0]], dataset="wesad") for i in (1, 2)]

    child = split_subjects(
        varied + wesad, RolesConfig(test=0.0), frozen=parent.preprocessing
    )

    swell = [c for c in child.clients if c.dataset == "swell"]
    assert all(c.train.feature_names == ["f0"] for c in swell)
    assert child.preprocessing["swell"] == parent.preprocessing["swell"]
    np.testing.assert_allclose(child.preprocessing["wesad"]["mean"], [2.5])


def test_a_frozen_feature_the_data_lacks_is_an_error() -> None:
    parent = two_way([[3.0], [8.0]])
    renamed = dict(parent.preprocessing["swell"], features=["missing"])

    with pytest.raises(DataError, match="missing"):
        split_subjects(cohort(4), RolesConfig(test=0.0), frozen={"swell": renamed})


def test_a_frozen_preprocessing_reproduces_the_parents_arrays_bit_for_bit() -> None:
    # Heavy-tailed values, where x - mean is inexact in float32 (HRV, EDA, ...).
    values = np.random.default_rng(0).lognormal(3, 2, size=(5, 40, 2))
    data = [subject(str(i), v) for i, v in enumerate(values, 1)]
    parent = split_subjects(data, RolesConfig(test=0.2))

    child = split_subjects(data, RolesConfig(test=0.2), frozen=parent.preprocessing)

    for mine, theirs in zip(child.clients, parent.clients, strict=True):
        assert mine.train.X.dtype == theirs.train.X.dtype
        np.testing.assert_array_equal(mine.train.X, theirs.train.X)
    np.testing.assert_array_equal(child.test[0].X, parent.test[0].X)
