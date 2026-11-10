"""Empatica E4 readers, multi-rate windows and interval labels (issue #239).

Every file here is a tiny format fixture written to ``tmp_path`` to check
parsing: the layout of an E4 CSV export and of a WESAD subject file, with
rates far below the device's to keep them readable. Nothing is trained or
evaluated with them.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from onion_fl.data.contract import DataError
from onion_fl.data.ingest import (
    TIME,
    DatasetSpec,
    IngestContext,
    ingest,
    readers,
    steps,
)

START = 1_600_000_000.0  # a Unix time, as in the first row of every E4 CSV
SECONDS = 10
HARMONIZED_CHANNELS = ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp"]
HARMONIZED = [
    f"{channel}_{stat}"
    for channel in HARMONIZED_CHANNELS
    for stat in ["mean", "std", "min", "max", "median"]
]


def _csv(path: Path, starts: list[float], rates: list[float], rows) -> None:
    lines = [
        ", ".join(f"{s:.6f}" for s in starts),
        ", ".join(f"{r:.6f}" for r in rates),
    ]
    lines += [", ".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def e4_session(folder: Path) -> Path:
    """ACC at 2 Hz, BVP at 4 Hz, EDA and TEMP at 1 Hz, HR from second 2, and IBI."""
    folder.mkdir(parents=True, exist_ok=True)
    acc = [(i, 10 + i, 20 + i) for i in range(2 * SECONDS)]
    _csv(folder / "ACC.csv", [START] * 3, [2.0] * 3, acc)
    _csv(folder / "BVP.csv", [START], [4.0], [(i,) for i in range(4 * SECONDS)])
    _csv(folder / "EDA.csv", [START], [1.0], [(0.1 * i,) for i in range(SECONDS)])
    _csv(folder / "TEMP.csv", [START], [1.0], [(30 + i,) for i in range(SECONDS)])
    _csv(folder / "HR.csv", [START + 2], [1.0], [(60 + i,) for i in range(SECONDS - 2)])
    ibi = [(0.5, 0.8), (1.3, 0.8), (6.0, 0.9)]
    (folder / "IBI.csv").write_text(
        f"{START:.6f}, IBI\n" + "".join(f"{o},{v}\n" for o, v in ibi),
        encoding="utf-8",
    )
    (folder / "tags.csv").write_text(f"{START + 4:.6f}\n", encoding="utf-8")
    return folder


def _ctx(root: Path) -> IngestContext:
    return IngestContext(root=root, options={})


# --- empatica_e4_csv ------------------------------------------------------------


def test_e4_reader_keeps_each_signal_at_its_rate(tmp_path: Path) -> None:
    df = readers.create("empatica_e4_csv", {}).read(e4_session(tmp_path / "S1"))

    assert list(df.columns) == ["time", *HARMONIZED_CHANNELS]
    assert len(df) == 2 * SECONDS + 4 * SECONDS + SECONDS + SECONDS
    assert np.all(np.diff(df["time"]) >= 0)  # one table, in time order
    acc = df.dropna(subset=["acc_x"])
    np.testing.assert_allclose(acc["time"] - START, np.arange(2 * SECONDS) / 2)
    assert acc["acc_z"].tolist() == [20 + i for i in range(2 * SECONDS)]
    eda = df.dropna(subset=["eda"])
    np.testing.assert_allclose(eda["time"] - START, np.arange(SECONDS))
    assert eda[["acc_x", "bvp", "temp"]].isna().all().all()  # a row is one signal


def test_e4_reader_emits_channels_in_one_order(tmp_path: Path) -> None:
    folder = e4_session(tmp_path / "S1")
    reader = readers.create("empatica_e4_csv", {"signals": "IBI,TEMP,HR,EDA"})

    df = reader.read(folder)

    assert list(df.columns) == ["time", "eda", "temp", "hr", "ibi"]
    hr = df.dropna(subset=["hr"])
    assert hr["time"].iloc[0] == START + 2  # HR starts later, with its own stamp
    ibi = df.dropna(subset=["ibi"])
    np.testing.assert_allclose(ibi["time"] - START, [0.5, 1.3, 6.0])
    assert ibi["ibi"].tolist() == [0.8, 0.8, 0.9]


def test_e4_reader_takes_the_folder_or_a_file_in_it(tmp_path: Path) -> None:
    folder = e4_session(tmp_path / "S1")
    reader = readers.create("empatica_e4_csv", {"signals": ["EDA"]})

    pd.testing.assert_frame_equal(reader.read(folder), reader.read(folder / "EDA.csv"))


def test_e4_reader_names_a_missing_signal_file(tmp_path: Path) -> None:
    folder = e4_session(tmp_path / "S1")
    (folder / "BVP.csv").unlink()

    with pytest.raises(DataError, match="BVP.csv"):
        readers.create("empatica_e4_csv", {}).read(folder)


def test_e4_reader_rejects_columns_with_different_rates(tmp_path: Path) -> None:
    folder = e4_session(tmp_path / "S1")
    _csv(folder / "ACC.csv", [START] * 3, [32.0, 32.0, 16.0], [(1, 2, 3)])

    with pytest.raises(DataError, match="different starts or rates"):
        readers.create("empatica_e4_csv", {"signals": "ACC"}).read(folder)


def test_e4_reader_rejects_unknown_or_repeated_signals() -> None:
    with pytest.raises(ValueError, match="GSR"):
        readers.create("empatica_e4_csv", {"signals": "EDA,GSR"})
    with pytest.raises(ValueError, match="repeated"):
        readers.create("empatica_e4_csv", {"signals": "EDA,EDA"})


# --- wesad_e4 (the structure of a WESAD subject file) --------------------------------


def wesad_file(path: Path, seconds: int = 4) -> Path:
    n = 700 * seconds
    data = {
        "label": np.repeat([1, 2], n // 2),
        "signal": {
            "chest": {"ECG": np.zeros((n, 1))},
            "wrist": {
                "ACC": np.arange(32 * seconds * 3, dtype=float).reshape(-1, 3),
                "BVP": np.arange(64 * seconds, dtype=float).reshape(-1, 1),
                "EDA": np.arange(4 * seconds, dtype=float).reshape(-1, 1),
                "TEMP": np.arange(4 * seconds, dtype=float).reshape(-1, 1),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(data))
    return path


def test_wesad_e4_reader_reads_the_wrist_like_the_e4_reader(tmp_path: Path) -> None:
    df = readers.create("wesad_e4", {}).read(wesad_file(tmp_path / "S2" / "S2.pkl"))

    assert list(df.columns) == ["time", *HARMONIZED_CHANNELS, "condition"]
    assert len(df) == (32 + 64 + 4 + 4 + 700) * 4
    bvp = df.dropna(subset=["bvp"])
    np.testing.assert_allclose(bvp["time"], np.arange(64 * 4) / 64)
    acc = df.dropna(subset=["acc_y"])
    assert acc["acc_y"].iloc[32] == 32 * 3 + 1  # second 1 of the 32 Hz signal
    condition = df.dropna(subset=["condition"])
    np.testing.assert_allclose(condition["time"], np.arange(700 * 4) / 700)
    assert condition["condition"].iloc[[0, -1]].tolist() == [1, 2]


def test_wesad_e4_reader_names_a_signal_the_file_lacks(tmp_path: Path) -> None:
    path = wesad_file(tmp_path / "S2" / "S2.pkl")
    data = pickle.loads(path.read_bytes())
    del data["signal"]["wrist"]["TEMP"]
    path.write_bytes(pickle.dumps(data))

    with pytest.raises(DataError, match="TEMP"):
        readers.create("wesad_e4", {}).read(path)


def test_wesad_e4_reader_has_no_device_derived_signals() -> None:
    with pytest.raises(ValueError, match="HR"):
        readers.create("wesad_e4", {"signals": "EDA,HR"})


# --- multirate_window ---------------------------------------------------------------


def _session_table(tmp_path: Path, signals: str = "ACC,BVP,EDA,TEMP") -> pd.DataFrame:
    df = readers.create("empatica_e4_csv", {"signals": signals}).read(
        e4_session(tmp_path / "S1")
    )
    return df.assign(subject="S1", **{TIME: df["time"]})


def window(params: dict, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
    return steps.create("multirate_window", params).apply(df, ctx)


def test_multirate_window_reads_each_channel_at_its_rate(tmp_path: Path) -> None:
    out = window(
        {"seconds": 4, "overlap": 0.5, "columns": ["eda", "bvp"]},
        _session_table(tmp_path),
    )

    assert out[TIME].tolist() == [START + 4, START + 6, START + 8, START + 10]
    assert out["eda_mean"].iloc[0] == pytest.approx(0.15)  # 4 samples at 1 Hz
    assert out["bvp_max"].iloc[0] == 15  # 16 samples at 4 Hz
    assert out["bvp_min"].iloc[1] == 8  # the second window starts at second 2
    assert list(out.columns[:2]) == ["subject", TIME]


def test_multirate_window_spans_only_what_every_channel_covers(
    tmp_path: Path,
) -> None:
    df = _session_table(tmp_path, "EDA,HR")  # HR covers seconds 2 to 10

    out = window({"seconds": 4, "overlap": 0.5, "columns": ["eda", "hr"]}, df)

    assert out[TIME].tolist() == [START + 6, START + 8, START + 10]
    assert out["hr_min"].iloc[0] == 60


def test_multirate_window_drops_windows_where_a_channel_is_empty(
    tmp_path: Path,
) -> None:
    df = _session_table(tmp_path, "EDA,IBI")  # IBI has no beat in seconds 2-6

    out = window({"seconds": 2, "columns": ["eda", "ibi"]}, df)

    # IBI starts at 0.5 s, so the windows do too.
    assert out[TIME].tolist() == [START + 2.5, START + 6.5]


def test_multirate_window_labels_from_the_label_rows_only(tmp_path: Path) -> None:
    df = readers.create("wesad_e4", {"signals": "EDA"}).read(
        wesad_file(tmp_path / "S2" / "S2.pkl")
    )
    labels = df["condition"].map({1: 0.0, 2: 1.0})
    df = df.assign(subject="S2", label=labels, **{TIME: df["time"]})
    ctx = _ctx(tmp_path)
    ctx.meta_columns |= {"time", "condition"}

    out = window(
        {"seconds": 2, "label": "mode", "label_rows": "condition"},
        df,
        ctx,
    )

    # The EDA rows carry no label sample: they do not drop the windows.
    assert out["label"].tolist() == [0, 1]
    assert [c for c in out.columns if c.startswith("eda")] == [
        f"eda_{s}" for s in ["mean", "std", "min", "max", "median"]
    ]


def test_multirate_window_drops_windows_with_a_label_sample_without_class(
    tmp_path: Path,
) -> None:
    df = _session_table(tmp_path, "EDA")
    label = np.where(df["time"] - START < 6, 0.0, np.nan)

    out = window({"seconds": 2, "columns": ["eda"]}, df.assign(label=label))

    assert out[TIME].tolist() == [START + 2, START + 4, START + 6]


def test_multirate_window_needs_time_and_subject(tmp_path: Path) -> None:
    df = _session_table(tmp_path, "EDA")

    with pytest.raises(DataError, match="time step"):
        window({"seconds": 2}, df.drop(columns=[TIME]))
    with pytest.raises(DataError, match="subject"):
        window({"seconds": 2}, df.drop(columns=["subject"]))


# --- intervals ----------------------------------------------------------------------


def _intervals(root: Path, rows: list[tuple]) -> None:
    lines = ["subject,start,end,class"] + [",".join(map(str, r)) for r in rows]
    (root / "protocol.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def intervals(root: Path, df: pd.DataFrame, **params) -> pd.DataFrame:
    params = {"source": {"reader": "csv", "path": "protocol.csv"}, **params}
    return steps.create("intervals", params).apply(df, _ctx(root))


def test_intervals_give_each_row_the_class_that_contains_it(tmp_path: Path) -> None:
    _intervals(
        tmp_path,
        [("S1", START, START + 4, "rest"), ("S1", START + 6, START + 10, "tsst")],
    )
    df = pd.DataFrame({"subject": "S1", TIME: START + np.array([0, 3.9, 4, 5, 6, 9.5])})

    out = intervals(tmp_path, df, by=["subject"])

    assert out["interval"].tolist()[:2] == ["rest", "rest"]
    assert out["interval"].isna().tolist() == [False, False, True, True, False, False]
    assert out["interval"].iloc[-1] == "tsst"


def test_intervals_are_matched_by_key(tmp_path: Path) -> None:
    _intervals(tmp_path, [("S1", 0, 10, "rest"), ("S2", 0, 10, "tsst")])
    df = pd.DataFrame({"subject": ["S1", "S2", "S3"], TIME: [5.0, 5.0, 5.0]})

    out = intervals(tmp_path, df, by=["subject"])

    assert out["interval"].tolist()[:2] == ["rest", "tsst"]
    assert pd.isna(out["interval"].iloc[2])


def test_intervals_read_dates(tmp_path: Path) -> None:
    (tmp_path / "protocol.csv").write_text(
        "start,end,class\n2020-09-13 12:00:00,2020-09-13 12:05:00,1\n",
        encoding="utf-8",
    )
    noon = (pd.Timestamp("2020-09-13 12:00:00") - pd.Timestamp(0)).total_seconds()
    df = pd.DataFrame({"subject": "S1", TIME: [noon - 1, noon, noon + 299]})

    out = intervals(tmp_path, df)

    assert out["interval"].isna().tolist() == [True, False, False]


def test_intervals_must_not_overlap(tmp_path: Path) -> None:
    _intervals(tmp_path, [("S1", 0, 10, "rest"), ("S1", 5, 15, "tsst")])
    df = pd.DataFrame({"subject": "S1", TIME: [1.0]})

    with pytest.raises(DataError, match="overlapping"):
        intervals(tmp_path, df, by=["subject"])


def test_intervals_need_a_time_step(tmp_path: Path) -> None:
    _intervals(tmp_path, [("S1", 0, 10, "rest")])

    with pytest.raises(DataError, match="time step"):
        intervals(tmp_path, pd.DataFrame({"subject": ["S1"]}))


# --- the whole E4 path --------------------------------------------------------------


def e4_spec(root: Path, source: dict, label: dict, label_rows=None) -> DatasetSpec:
    return DatasetSpec(
        name="demo_e4",
        root=str(root),
        source=source,
        steps=[
            {"time": {"column": "time"}},
            *label,
            {"label": {"task": "stress_binary", **label_rows_rule(label_rows)}},
            {
                "multirate_window": {
                    "seconds": 4,
                    "overlap": 0.5,
                    "label": "mode",
                    "label_rows": label_rows,
                }
            },
            {"features": {"regex": "_(mean|std|min|max|median)$"}},
        ],
    )


def label_rows_rule(label_rows) -> dict:
    if label_rows == "condition":
        return {"column": "condition", "map": {1: 0, 2: 1}}
    return {"column": "interval", "map": {"rest": 0, "tsst": 1}}


def test_e4_sessions_ingest_with_interval_labels(tmp_path: Path) -> None:
    e4_session(tmp_path / "S1")
    e4_session(tmp_path / "S2")
    _intervals(
        tmp_path,
        [
            ("S1", START, START + 4, "rest"),
            ("S1", START + 6, START + 10, "tsst"),
            ("S2", START, START + 10, "tsst"),
        ],
    )
    steps_intervals = [
        {
            "intervals": {
                "source": {"reader": "csv", "path": "protocol.csv"},
                "by": ["subject"],
            }
        }
    ]
    source = {"reader": "empatica_e4_csv", "path": "{subject}/EDA.csv"}

    subjects = ingest(e4_spec(tmp_path, source, steps_intervals))

    s1, s2 = subjects
    assert (s1.subject, s2.subject) == ("S1", "S2")
    assert s1.y.tolist() == [0, 1]  # the windows across the gap are dropped
    assert s2.y.tolist() == [1, 1, 1, 1]
    np.testing.assert_allclose(s1.t, [0, 6])
    # No timestamp, subject or interval column reaches the features.
    assert s1.feature_names == HARMONIZED


def test_wesad_and_e4_csv_share_feature_names(tmp_path: Path) -> None:
    wesad_file(tmp_path / "S2" / "S2.pkl", seconds=8)
    source = {"reader": "wesad_e4", "path": "{subject}/{subject}.pkl"}

    (data,) = ingest(e4_spec(tmp_path, source, [], label_rows="condition"))

    assert data.feature_names == HARMONIZED
    assert len(data.y) == 3 and (data.y[0], data.y[-1]) == (0, 1)


def test_a_subject_left_out_by_a_select_adds_no_rows(tmp_path: Path) -> None:
    e4_session(tmp_path / "S1")
    e4_session(tmp_path / "S2")
    _intervals(tmp_path, [("S1", START, START + 10, "rest")])
    spec = e4_spec(
        tmp_path,
        {"reader": "empatica_e4_csv", "path": "{subject}/EDA.csv"},
        [
            {"select": {"query": "subject != 'S2'"}},
            {
                "intervals": {
                    "source": {"reader": "csv", "path": "protocol.csv"},
                    "by": ["subject"],
                }
            },
        ],
    )

    (data,) = ingest(spec)  # no warning from an empty frame (filterwarnings=error)

    assert data.subject == "S1" and data.y.tolist() == [0, 0, 0, 0]


# --- stress_predict_time_logs (issue #242) -----------------------------------------


def time_logs(path: Path, rows: list[tuple[str, list[str], str]]) -> Path:
    """Time_logs.xlsx's layout: a two-row header, a (Start, End) pair per phase."""
    pytest.importorskip("openpyxl")
    import datetime as dt

    phases = ["Consent", "Baseline/Questionniare", "Stroop Test", "Relax"]
    phases += ["Interview", "Relax", "Hyperventilation", "Relax"]
    phases += ["Questionniare", "Relax/Baseline"]
    head = ["S. ID.", "Start Time", "End Time", "Session ID"]
    head += [x for p in phases for x in (p, None)] + [None, "Date"]
    sub = [None] * 4 + ["Start", "End"] * len(phases) + [None, None]
    body = []
    for subject, clock, day in rows:
        times = [dt.time(*map(int, c.split(":"))) for c in clock]
        body.append(
            [subject, times[0], times[-1], 1]
            + times
            + ["Day 1", dt.datetime.fromisoformat(day)]
        )
    _sheet(path, [head, sub, *body])
    return path


def _sheet(path: Path, rows: list[list]) -> None:
    from openpyxl import Workbook

    book = Workbook()
    for row in rows:
        book.active.append(row)
    book.save(path)


def unix(text: str) -> float:
    return (pd.Timestamp(text) - pd.Timestamp(0)).total_seconds()


MORNING = [  # 20 clock times: 10 phases, the last ones after noon on a 12-hour clock
    "11:00", "11:05", "11:05", "11:15", "11:15", "11:20", "11:20", "11:25",
    "11:25", "11:35", "11:35", "11:40", "11:40", "11:42", "11:42", "11:47",
    "11:47", "11:52", "12:52", "01:02",
]  # fmt: skip


def test_time_logs_give_one_row_per_phase_in_unix_seconds(tmp_path: Path) -> None:
    path = time_logs(tmp_path / "logs.xlsx", [("S02", MORNING, "2022-02-07")])

    out = readers.create("stress_predict_time_logs", {}).read(path)

    assert out.columns.tolist() == ["subject", "phase", "start", "end"]
    assert out["phase"].tolist()[:3] == ["consent", "baseline", "stroop"]
    stroop = out.set_index("phase").loc["stroop"]
    assert (stroop.start, stroop.end) == (
        unix("2022-02-07 11:15"),
        unix("2022-02-07 11:20"),
    )
    last = out.set_index("phase").loc["relax_4"]  # 12:52 to 01:02 is past noon
    assert (last.start, last.end) == (
        unix("2022-02-07 12:52"),
        unix("2022-02-07 13:02"),
    )


def test_time_logs_end_a_phase_where_the_next_begins(tmp_path: Path) -> None:
    clock = list(MORNING)
    clock[3] = "11:16"  # the baseline ends a minute after the Stroop test starts
    path = time_logs(tmp_path / "logs.xlsx", [("S33", clock, "2022-03-09")])

    out = readers.create("stress_predict_time_logs", {}).read(path)

    baseline = out.set_index("phase").loc["baseline"]
    assert baseline.end == unix("2022-03-09 11:15")


def test_time_logs_need_the_phase_columns(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "logs.xlsx"
    _sheet(path, [["S. ID.", "Date"], ["", "Start"], ["S02", None]])

    with pytest.raises(DataError, match=r"\(Start, End\) pairs"):
        readers.create("stress_predict_time_logs", {}).read(path)
