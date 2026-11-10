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
    "name", ["swell", "sweet", "wesad", "wesad_e4", "nurse_stress", "femnist"]
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
        ("nurse_stress", {"label": "high", "signals": "EDA,TEMP,HR"}),
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


NURSE = Path("data/NurseStress")


def _survey_events():
    """The survey events in Unix UTC, computed here apart from the descriptor."""
    import pandas as pd

    survey = pd.read_excel(NURSE / "SurveyResults.xlsx", na_values=["na"])
    day = survey["date"].dt.strftime("%Y-%m-%d") + " "

    def utc(column: str) -> pd.Series:
        local = pd.to_datetime(day + survey[column].astype(str))
        stamps = local.dt.tz_localize("America/Chicago").dt.tz_convert("UTC")
        return (stamps - pd.Timestamp(0, tz="UTC")).dt.total_seconds()

    start, end = utc("Start time"), utc("End time")
    end = end.where(end > start, end + 86400)
    return survey.assign(subject=survey["ID"].astype(str), start=start, end=end)


@pytest.mark.skipif(
    not (NURSE / "Stress_dataset").exists(), reason="data/NurseStress not available"
)
def test_nurse_stress_matches_the_published_files() -> None:
    sessions = sorted((NURSE / "Stress_dataset").glob("*/*/EDA.csv"))
    nurses = sorted({s.parent.parent.name for s in sessions})
    survey = _survey_events()

    assert len(nurses) == 15 and len(sessions) == 609
    assert all(s.parent.name.startswith(s.parent.parent.name + "_") for s in sessions)
    assert len(survey) == 358 and sorted(survey["subject"].unique()) == nurses
    levels = survey["Stress level"].value_counts(dropna=False).to_dict()
    assert {k: v for k, v in levels.items() if k == k} == {2: 179, 1: 20, 0: 46}
    assert survey["Stress level"].isna().sum() == 113


def _covered(events, times: np.ndarray) -> np.ndarray:
    return np.array([((events.start <= x) & (events.end > x)).any() for x in times])


@pytest.mark.skipif(
    not (NURSE / "Stress_dataset").exists(), reason="data/NurseStress not available"
)
def test_nurse_stress_labels_follow_the_survey() -> None:
    """Nurse 6D: each sample's label is the one the survey gives its time."""
    from onion_fl.data.ingest import IngestContext, readers, steps

    spec = descriptor("nurse_stress")
    events = _survey_events().query("subject == '6D'")
    stress = events[events["Stress level"] >= 1]
    reader = readers.create("empatica_e4_csv", {})
    ctx = IngestContext(root=NURSE, options={})
    labelled = [steps.create(*next(iter(s.items()))) for s in spec.steps[:3]]
    surveyed = 0
    for session in sorted((NURSE / "Stress_dataset" / "6D").iterdir()):
        df = reader.read(session).assign(subject="6D")
        for step in labelled:
            df = step.apply(df, ctx)
        t = df["time"].to_numpy()[::97]  # about one sample a second
        anywhere, in_stress = _covered(events, t), _covered(stress, t)
        expected = np.where(in_stress, 1.0, np.where(anywhere, np.nan, 0.0))
        if not anywhere.any():
            expected[:] = np.nan  # a session without an event is dropped
        surveyed += bool(anywhere.any())
        np.testing.assert_array_equal(df["label"].to_numpy()[::97], expected)
    assert surveyed == 1  # 6D has one session with events (2026-11-10)

    one = {"path": "Stress_dataset/{subject}/6D_*/EDA.csv"}
    spec.source = spec.source.model_copy(update=one)
    (data,) = ingest(spec)
    assert data.feature_names[:2] == ["acc_x_mean", "acc_x_std"]
    assert np.bincount(data.y.astype(int)).tolist() == [63, 92]


NURSE_SAMPLE = Path("data/samples/NurseStress")


@pytest.mark.skipif(
    not NURSE_SAMPLE.exists(), reason="data/samples/NurseStress not available"
)
def test_nurse_stress_sample_reads_with_both_classes() -> None:
    """The real extract of scripts/nurse_stress_sample.py: four nurses."""
    subjects = {
        s.subject: s for s in ingest(descriptor("nurse_stress"), root=NURSE_SAMPLE)
    }

    counts = {k: np.bincount(s.y.astype(int)).tolist() for k, s in subjects.items()}
    assert counts == {"7E": [55, 52], "BG": [55, 36], "E4": [46, 48], "F5": [75, 24]}
    for data in subjects.values():
        assert data.task == "stress_binary" and data.n_classes == 2
        assert len(data.feature_names) == 30  # 6 E4 channels x 5 statistics


@pytest.mark.skipif(
    not (NURSE_SAMPLE.exists() and (NURSE / "Stress_dataset").exists()),
    reason="data/NurseStress or its sample not available",
)
def test_nurse_stress_sample_matches_the_full_dataset() -> None:
    spec = descriptor("nurse_stress")
    sample = {s.subject: s for s in ingest(spec, root=NURSE_SAMPLE)}
    for session in ["E4_1593538007", "BG_1607735684"]:
        nurse = session.split("_")[0]
        spec.source = spec.source.model_copy(
            update={"path": f"Stress_dataset/{{subject}}/{session}/EDA.csv"}
        )
        (full,) = ingest(spec)
        np.testing.assert_array_equal(sample[nurse].y, full.y)
        np.testing.assert_array_equal(sample[nurse].X, full.X)


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
