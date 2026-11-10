"""The shipped dataset descriptors (issue #87).

The descriptors are checked without data. Parity with the previous loaders
needs the real datasets under ``data/`` and is skipped when they are absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from onion_fl.data.ingest import check_spec, ingest, load_spec

DESCRIPTORS = Path("datasets")


def descriptor(name: str):
    return load_spec(DESCRIPTORS / f"{name}.yaml")


@pytest.mark.parametrize(
    "name", ["swell", "sweet", "wesad", "femnist", "heart_disease"]
)
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


# Fed-Heart-Disease (FLamby): train/test per center, from its README.
FLAMBY_COUNTS = {
    "cleveland": (199, 104),
    "hungarian": (172, 89),
    "switzerland": (30, 16),
    "va": (85, 45),
}
FLAMBY_FEATURES = [
    "age", "sex", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak",
    "cp_2", "cp_3", "cp_4", "restecg_1", "restecg_2",
]  # fmt: skip


@pytest.mark.skipif(
    not Path("data/HeartDisease/flamby").exists(),
    reason="data/HeartDisease/flamby not available",
)
def test_heart_disease_has_flambys_split_per_center() -> None:
    subjects = {s.subject: s for s in ingest(descriptor("heart_disease"))}

    expected = {
        name: n
        for center, (train, test) in FLAMBY_COUNTS.items()
        for name, n in ((center, train), (f"{center}_test", test))
    }
    assert {name: s.n_samples for name, s in subjects.items()} == expected
    assert sum(expected.values()) == 740
    for data in subjects.values():
        assert data.feature_names == FLAMBY_FEATURES  # never the center
        assert data.task == "heart_disease" and data.n_classes == 2
    # FLamby's labels: Switzerland trains on disease only.
    assert list(subjects["switzerland"].class_counts) == [0, 30]
    assert list(subjects["cleveland"].class_counts) == [108, 91]


@pytest.mark.skipif(
    not Path("data/HeartDisease/raw").exists(),
    reason="data/HeartDisease/raw not available",
)
def test_heart_disease_script_reproduces_flambys_counts() -> None:
    spec = importlib.util.spec_from_file_location(
        "heart_disease_flamby", "scripts/heart_disease_flamby.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    for center, (train, test) in FLAMBY_COUNTS.items():
        df = script.read_center(Path(f"data/HeartDisease/raw/processed.{center}.data"))
        held = script.split(df)
        assert (int((~held).sum()), int(held.sum())) == (train, test), center
        table = script.one_hot(df)
        assert list(table.columns) == [*FLAMBY_FEATURES, "num"]


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
