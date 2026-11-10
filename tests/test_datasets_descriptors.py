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
    "name",
    [
        "swell",
        "sweet",
        "wesad",
        "wesad_e4",
        "physionet_e4",
        "nurse_stress",
        "stress_predict",
        "femnist",
    ],
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
        ("physionet_e4", {"signals": "EDA,HR", "window_seconds": 30}),
        ("physionet_e4", {"label": "three_class"}),
        ("nurse_stress", {"label": "high", "signals": "EDA,TEMP,HR"}),
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


# --- PhysioNet Wearable Device Dataset (issue #240) ---------------------------------

PHYSIONET = Path("data/PhysioNetE4/Wearable_Dataset")
PHYSIONET_SAMPLE = Path("data/samples/physionet_e4")
DROPPED = {  # data_constraints.txt, as the descriptor's select step applies it
    "STRESS/f07",
    "AEROBIC/S03",
    "AEROBIC/S07",
    "AEROBIC/S11_a",
    "AEROBIC/S11_b",
    "ANAEROBIC/S06",
    "ANAEROBIC/S16_a",
    "ANAEROBIC/S16_b",
}
TAGS = {  # tags per complete session, as the protocol figures of the notebook
    ("STRESS", "V1"): 13,
    ("STRESS", "V2"): 9,
    ("AEROBIC", "V1"): 12,
    ("AEROBIC", "V2"): 8,
    ("ANAEROBIC", "V1"): 7,
    ("ANAEROBIC", "V2"): 11,
}
SHORT_STAGES = {"V1": [7, 9, 11], "V2": [4, 6, 8]}  # opinions and subtract, 30 s


def _tag_seconds(folder: Path) -> np.ndarray:
    import pandas as pd

    start = pd.Timestamp(folder.joinpath("EDA.csv").read_text().splitlines()[0])
    lines = folder.joinpath("tags.csv").read_text().split()
    stamps = [" ".join(pair) for pair in zip(lines[::2], lines[1::2], strict=True)]
    return np.array([(pd.Timestamp(t) - start).total_seconds() for t in stamps])


@pytest.mark.skipif(not PHYSIONET.exists(), reason="data/PhysioNetE4 not available")
def test_physionet_tags_follow_the_protocol_the_labels_assume() -> None:
    """Every session the descriptor labels has its protocol's tags in place."""
    unsorted = set()
    for folder in sorted(PHYSIONET.glob("*/*")):
        session = f"{folder.parent.name}/{folder.name}"
        if session in DROPPED or session == "STRESS/f14_a":  # f14_a: no tags
            continue
        version = "V1" if folder.name.startswith("S") else "V2"
        tags = _tag_seconds(folder)
        assert len(tags) == TAGS[folder.parent.name, version], session
        if np.any(np.diff(tags) <= 0):
            unsorted.add(session)
        tags = np.sort(tags)  # as the reader does
        assert np.all(np.diff(tags) > 0), session
        if folder.parent.name == "STRESS":
            # The 30 s stages last 30 s: the tag k is where the map says.
            for k in SHORT_STAGES[version]:
                assert 15 <= tags[k] - tags[k - 1] <= 75, (session, k)
    # Its tags 5 and 6 (the 90/95 rpm block) were written 12 s apart, swapped.
    assert unsorted == {"AEROBIC/f07"}


@pytest.mark.skipif(not PHYSIONET.exists(), reason="data/PhysioNetE4 not available")
def test_physionet_s02_repeat_starts_where_the_descriptor_cuts_it() -> None:
    import pandas as pd

    folder = PHYSIONET / "STRESS" / "S02"
    eda = pd.read_csv(folder / "EDA.csv", header=None, skiprows=2)[0].to_numpy()
    start = pd.Timestamp(folder.joinpath("EDA.csv").read_text().splitlines()[0])

    half = 1548 * 4  # data_constraints.txt: EDA row 6,195, after the two header rows
    np.testing.assert_array_equal(eda[half:], eda[: len(eda) - half])
    assert (start - pd.Timestamp(0)).total_seconds() + 1548 == 1361387717
    assert _tag_seconds(folder)[-1] < 1548  # the protocol ends before the repeat


@pytest.mark.skipif(not PHYSIONET.exists(), reason="data/PhysioNetE4 not available")
@pytest.mark.parametrize(
    "label, n_subjects, counts",
    [("binary", 35, [1243, 203]), ("three_class", 36, [1243, 203, 1197])],
)
def test_physionet_reads_one_subject_per_participant(label, n_subjects, counts):
    subjects = ingest(descriptor("physionet_e4"), {"label": label})

    assert len(subjects) == n_subjects  # 36 participants; f07 has no valid stress
    assert ("f07" in {s.subject for s in subjects}) == (label == "three_class")
    labels = np.concatenate([s.y for s in subjects]).astype(int)
    assert np.bincount(labels).tolist() == counts  # 2026-11-10, 60 s windows
    for data in subjects:
        assert data.feature_names == descriptor_features(), data.subject
        # A participant's sessions come in path order (AEROBIC, ANAEROBIC,
        # STRESS), not in time order; their windows never share a time.
        assert data.t.min() == 0 and len(np.unique(data.t)) == len(data.t)


def descriptor_features() -> list[str]:
    channels = ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp"]
    stats = ["mean", "std", "min", "max", "median"]
    return [f"{c}_{s}" for c in channels for s in stats]  # as wesad_e4.yaml


@pytest.mark.skipif(
    not PHYSIONET_SAMPLE.exists(), reason="data/samples/physionet_e4 not available"
)
def test_physionet_sample_reads_its_participants() -> None:
    """The real extract: S02 (duplicated), f07 (invalid PPG) and f14 (split)."""
    spec = descriptor("physionet_e4")
    binary = {s.subject: s for s in ingest(spec, root=PHYSIONET_SAMPLE)}
    three = {
        s.subject: s
        for s in ingest(spec, {"label": "three_class"}, root=PHYSIONET_SAMPLE)
    }

    assert sorted(binary) == ["S02", "f14"]
    assert np.bincount(binary["S02"].y.astype(int)).tolist() == [20, 4]
    assert np.bincount(binary["f14"].y.astype(int)).tolist() == [42, 6]
    assert sorted(three) == ["S02", "f07", "f14"]
    assert set(three["f07"].y.tolist()) == {2}  # its exercise sessions are valid


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
