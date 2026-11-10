"""The shipped dataset descriptors (issue #87).

The descriptors are checked without data. Parity with the previous loaders
needs the real datasets under ``data/`` and is skipped when they are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from onion_fl.data.ingest import check_spec, ingest, load_spec

DESCRIPTORS = Path("datasets")


def descriptor(name: str):
    return load_spec(DESCRIPTORS / f"{name}.yaml")


@pytest.mark.parametrize("name", ["swell", "sweet", "wesad", "femnist", "cifar10"])
def test_descriptors_are_valid(name: str) -> None:
    spec = descriptor(name)

    assert spec.name == name
    check_spec(spec)


@pytest.mark.parametrize(
    "name, options",
    [
        ("swell", {"facial": True, "physiology": True}),
        ("swell", {"label": "binary_no_r"}),
        ("sweet", {"label": "ordinal", "selection": "selection2/users"}),
        ("sweet", {"label": "three_class"}),
        ("wesad", {"location": "chest", "signals": "ECG,EDA", "label": "three_class"}),
        ("cifar10", {"partition": "iid"}),
        ("cifar10", {"partition": "label_skew", "k": 3, "clients": 20}),
        ("cifar10", {"alpha": 0.1, "partition_seed": 4}),
    ],
)
def test_descriptor_options_are_valid(name: str, options: dict) -> None:
    check_spec(descriptor(name), options)


def _same_where_present(new, old_X: np.ndarray, old_names: list[str]) -> None:
    for j, name in enumerate(new.feature_names):
        if name not in old_names:
            continue
        values, legacy = new.X[:, j], old_X[:, old_names.index(name)]
        present = ~np.isnan(values)  # the old loaders imputed over the whole dataset
        np.testing.assert_allclose(
            values[present], legacy[present], rtol=1e-5, err_msg=name
        )


@pytest.mark.skipif(not Path("data/SWELL").exists(), reason="data/SWELL not available")
def test_swell_reads_like_the_previous_loader() -> None:
    from onion_fl.datasets.swell import load_swell_all_samples

    X, y, subject_ids, info = load_swell_all_samples(modalities=["computer"])
    new = {s.subject: s for s in ingest(descriptor("swell"))}

    assert set(new) == {str(s) for s in np.unique(subject_ids)}
    for subject, data in new.items():
        mask = subject_ids.astype(str) == subject
        np.testing.assert_array_equal(data.y, y[mask])
        _same_where_present(data, X[mask], list(info["feature_names"]))


@pytest.mark.skipif(not Path("data/SWELL").exists(), reason="data/SWELL not available")
@pytest.mark.parametrize(
    "name, options, columns",
    [
        ("swell", {"facial": True}, ["squality", "sneutral", "svalence"]),
        ("swell", {"physiology": True}, ["hr", "rmssd", "scl"]),
        ("swell_physiology", {}, ["hr", "rmssd", "scl"]),
    ],
)
def test_swell_modalities_read_999_as_missing(name, options, columns) -> None:
    subjects = ingest(descriptor(name), options)

    assert len(subjects) == 25
    for data in subjects:
        idx = [data.feature_names.index(c) for c in columns]
        assert not (data.X[:, idx] == 999).any(), data.subject


@pytest.mark.skipif(
    not Path("data/SWEET/sample_subjects").exists(), reason="SWEET sample not available"
)
def test_sweet_reads_like_the_previous_loader() -> None:
    from onion_fl.datasets.sweet_samples import load_sweet_sample_full

    X, y, groups, names = load_sweet_sample_full()
    new = {s.subject: s for s in ingest(descriptor("sweet"))}

    assert set(new) == set(np.unique(groups))
    for subject, data in new.items():
        mask = groups == subject
        assert data.n_samples == int(mask.sum()), subject
        np.testing.assert_array_equal(np.sort(data.y), np.sort(y[mask]))


@pytest.mark.skipif(not Path("data/WESAD").exists(), reason="data/WESAD not available")
def test_wesad_chest_reads_like_the_previous_loader() -> None:
    from onion_fl.datasets.wesad import _load_subject_data

    # Not RESP or TEMP: the old loader looked them up in capitals, missed the
    # published "Resp" and "Temp", and returned no window at all.
    X, y, names = _load_subject_data(
        Path("data/WESAD"),
        "S2",
        ["ECG", "EDA", "EMG"],
        "chest",
        ["baseline", "stress"],
        60,
        0.5,
    )
    subjects = ingest(
        descriptor("wesad"), {"location": "chest", "signals": "ECG,EDA,EMG"}
    )
    new = {s.subject: s for s in subjects}["S2"]

    assert new.feature_names == names
    np.testing.assert_array_equal(new.y, y)
    np.testing.assert_allclose(new.X, X, rtol=1e-5)


def test_femnist_declares_a_fixed_subsample_of_hundreds_of_writers() -> None:
    writers = descriptor("femnist").source.reader_params()["writers"]

    assert len(writers) == len(set(writers)) == 200
    assert writers == sorted(writers)


@pytest.mark.skipif(
    not Path("data/FEMNIST/all_data").exists(), reason="data/FEMNIST not available"
)
def test_femnist_reads_one_subject_per_declared_writer() -> None:
    spec = descriptor("femnist")
    subjects = ingest(spec)

    assert [s.subject for s in subjects] == spec.source.reader_params()["writers"]
    assert sum(s.n_samples for s in subjects) == 44451  # flwrlabs/femnist, 2026-11-05
    labels = np.concatenate([s.y for s in subjects])
    assert set(np.unique(labels)) == set(range(62))
    for data in subjects:
        assert data.n_classes == 62 and data.task == "character"
        assert data.feature_names == [f"x_{i}" for i in range(784)]  # no writer id
        assert data.X.min() >= 0 and data.X.max() <= 1, data.subject
    # No class is a near-constant answer: the majority is '1' with under 6%.
    assert np.bincount(labels.astype(int)).max() / len(labels) < 0.06


CIFAR10 = Path("data/CIFAR10/cifar-10-batches-py")


@pytest.mark.skipif(not CIFAR10.exists(), reason="data/CIFAR10 not available")
@pytest.mark.parametrize("partition", ["iid", "dirichlet_label", "label_skew"])
def test_cifar10_reads_the_training_batches_into_clients_and_the_official_test(
    partition: str,
) -> None:
    subjects = {
        s.subject: s for s in ingest(descriptor("cifar10"), {"partition": partition})
    }

    assert sorted(subjects) == [f"client_{i}" for i in range(10)] + ["test"]
    test = subjects.pop("test")
    assert test.n_samples == 10000
    assert np.bincount(test.y.astype(int)).tolist() == [1000] * 10
    labels = np.concatenate([s.y for s in subjects.values()])
    if partition != "label_skew":  # k=2 of 10 classes may leave one out
        assert len(labels) == 50000
        assert np.bincount(labels.astype(int)).tolist() == [5000] * 10
    for data in [test, *subjects.values()]:
        assert data.n_classes == 10 and data.task == "object"
        assert data.feature_names == [f"x_{i}" for i in range(3072)]  # no batch
        assert data.X.min() >= 0 and data.X.max() <= 1, data.subject
    sizes = [s.n_samples for s in subjects.values()]
    if partition == "iid":
        assert sizes == [5000] * 10


@pytest.mark.skipif(not CIFAR10.exists(), reason="data/CIFAR10 not available")
def test_cifar10_training_and_test_images_do_not_overlap() -> None:
    subjects = ingest(descriptor("cifar10"), {"partition": "iid"})
    test = {s.subject: s for s in subjects}["test"]
    train = np.concatenate([s.X for s in subjects if s.subject != "test"])

    seen = {row.tobytes() for row in train}
    assert sum(row.tobytes() in seen for row in test.X) == 0


# --- per-row time on the real recordings (continuum C3) ----------------------------


@pytest.mark.skipif(not Path("data/SWELL").exists(), reason="data/SWELL not available")
def test_swell_rows_are_timed_in_order() -> None:
    for data in ingest(descriptor("swell")):
        assert data.t[0] == 0 and np.all(np.diff(data.t) >= 0), data.subject
        assert data.t[-1] > 3600, data.subject  # the sessions last about three hours


@pytest.mark.skipif(not Path("data/WESAD").exists(), reason="data/WESAD not available")
def test_wesad_windows_are_timed_in_order() -> None:
    for data in ingest(descriptor("wesad")):
        steps = np.diff(data.t)
        assert data.t[0] == 0 and np.all(steps > 0), data.subject
        assert np.isclose(steps.min(), 30.0), data.subject  # 60 s windows, half overlap
