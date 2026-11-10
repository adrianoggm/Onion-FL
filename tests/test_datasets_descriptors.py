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


@pytest.mark.parametrize(
    "name", ["swell", "sweet", "wesad", "wesad_e4", "stress_predict", "femnist"]
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
        (
            "wesad_e4",
            {"signals": "EDA,TEMP", "window_seconds": 30, "label": "three_class"},
        ),
        ("stress_predict", {"signals": "BVP,EDA", "window_seconds": 30}),
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


@pytest.mark.skipif(not Path("data/WESAD").exists(), reason="data/WESAD not available")
def test_wesad_wrist_reads_the_same_windows_at_native_rates() -> None:
    """The acceptance of #239, on S2 (wesad_e4.yaml records the 15 subjects)."""
    one = {"path": "{subject}/S2.pkl"}
    old_spec, new_spec = descriptor("wesad"), descriptor("wesad_e4")
    old_spec.source = old_spec.source.model_copy(update=one)
    new_spec.source = new_spec.source.model_copy(update=one)
    (old,) = ingest(old_spec)
    (new,) = ingest(new_spec)

    np.testing.assert_array_equal(new.y, old.y)
    np.testing.assert_allclose(new.t, old.t)
    renamed = {"acc_0": "acc_x", "acc_1": "acc_y", "acc_2": "acc_z"}
    renamed |= {"bvp_0": "bvp", "eda_0": "eda", "temp_0": "temp"}
    legacy = {
        renamed[name.rsplit("_", 1)[0]] + "_" + name.rsplit("_", 1)[1]: old.X[:, j]
        for j, name in enumerate(old.feature_names)
    }
    assert sorted(legacy) == sorted(new.feature_names)
    for j, name in enumerate(new.feature_names):
        channel, stat = name.rsplit("_", 1)
        span = new.X[:, new.feature_names.index(f"{channel}_max")]
        span = span - new.X[:, new.feature_names.index(f"{channel}_min")]
        error = np.abs(new.X[:, j] - legacy[name])
        if stat in ("min", "max") or channel in ("eda", "temp"):
            np.testing.assert_allclose(
                new.X[:, j], legacy[name], rtol=1e-6, err_msg=name
            )
        else:
            tolerance = 0.04 if stat == "median" else 3e-4
            assert (error <= tolerance * span + 1e-6).all(), name


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


# --- Stress-Predict (issue #242) ----------------------------------------------------

STRESS_PREDICT = Path("data/StressPredict")
E4_HARMONIZED = [  # the names wesad_e4.yaml gives too
    f"{channel}_{stat}"
    for channel in ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp"]
    for stat in ["mean", "std", "min", "max", "median"]
]


@pytest.mark.skipif(
    not STRESS_PREDICT.exists(), reason="data/StressPredict not available"
)
def test_stress_predict_labels_match_the_authors_csv() -> None:
    """Time_logs.xlsx against the authors' labels, second by second.

    The authors took the boundaries from the button presses, to the second, so
    the minute resolution of the time log leaves 3.7% of the seconds different
    (2026-11-10: 107,315 seconds with a class, 3,952 of them different).
    """
    pytest.importorskip("openpyxl")
    import pandas as pd

    from onion_fl.data.ingest import readers

    spec = descriptor("stress_predict")
    (label,) = [s["label"] for s in spec.steps if "label" in s]
    logs = readers.create("stress_predict_time_logs", {}).read(
        STRESS_PREDICT / "Processed_data/Time_logs.xlsx"
    )
    csv = pd.read_csv(
        STRESS_PREDICT / "Processed_data/Improved_All_Combined_hr_rsp_binary.csv"
    )
    subjects = csv["Participant"].map(lambda p: f"S{p:02d}").to_numpy()
    seconds = csv["Time(sec)"].to_numpy(np.float64)
    ours = np.full(len(csv), np.nan)
    for phase in logs.itertuples():
        inside = (subjects == phase.subject) & (seconds >= phase.start)
        inside &= seconds < phase.end
        ours[inside] = label["map"].get(phase.phase, np.nan)

    # The CSV has every subject but S01, which the descriptor leaves out.
    assert sorted(set(subjects)) == sorted(set(logs["subject"]) - {"S01"})
    held = ~np.isnan(ours)
    assert held.mean() > 0.95  # the rest falls between two phases
    assert (ours[held] == csv["Label"].to_numpy()[held]).mean() > 0.96


SAMPLE = Path("data/samples/stress_predict")  # scripts/stress_predict_sample.py


@pytest.mark.skipif(not SAMPLE.exists(), reason="data/samples/stress_predict missing")
def test_stress_predict_reads_the_harmonized_e4_features() -> None:
    pytest.importorskip("openpyxl")

    subjects = ingest(descriptor("stress_predict"), root=SAMPLE)

    assert [s.subject for s in subjects] == ["S02", "S03", "S04"]
    counts = [np.bincount(s.y.astype(int)).tolist() for s in subjects]
    assert counts == [[76, 35], [69, 31], [80, 34]]  # as on the whole dataset
    for data in subjects:
        assert data.n_classes == 2 and data.feature_names == E4_HARMONIZED
        assert data.t[0] == 0 and np.isclose(np.diff(data.t).min(), 30.0)
        assert not np.isnan(data.X).any()
