"""Tests for declarative ingestion: readers and steps (issue #86).

Every file here is a tiny format fixture written to ``tmp_path`` to check
parsing; nothing is trained or evaluated with it.
"""

from __future__ import annotations

import pickle
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from onion_fl.data.contract import DataError
from onion_fl.data.ingest import DatasetSpec, ingest, load_spec, readers, steps


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


def spec(root: Path, source: dict, steps: list[dict], **extra) -> DatasetSpec:
    return DatasetSpec(name="demo", root=str(root), source=source, steps=steps, **extra)


BASIC = [
    {"subject": {"column": "pp"}},
    {"label": {"task": "stress", "column": "cond", "map": {"N": 0, "T": 1}}},
    {"features": {"exclude": ["blok"]}},
]


@pytest.fixture
def table(tmp_path: Path) -> Path:
    write(
        tmp_path / "table.csv",
        """
        pp,blok,cond,keys,mouse
        1,1,N,10,0.5
        1,2,T,20,0.7
        2,1,N,30,0.1
        2,2,T,40,0.2
        2,3,X,50,0.3
        """,
    )
    return tmp_path


# --- the whole pipeline -------------------------------------------------------------


def test_a_table_becomes_one_subject_data_per_subject(table: Path) -> None:
    subjects = ingest(spec(table, {"reader": "csv", "path": "table.csv"}, BASIC))

    assert [s.subject for s in subjects] == ["1", "2"]
    first = subjects[0]
    assert first.feature_names == ["keys", "mouse"]
    np.testing.assert_allclose(first.X, [[10, 0.5], [20, 0.7]])
    assert first.y.tolist() == [0, 1]
    assert first.meta["n_classes"] == 2 and first.task == "stress"


def test_rows_without_a_class_are_dropped(table: Path) -> None:
    subjects = ingest(spec(table, {"reader": "csv", "path": "table.csv"}, BASIC))

    assert subjects[1].n_samples == 2  # the 'X' row has no class in the map


def test_a_spec_loads_from_yaml(table: Path) -> None:
    path = write(
        table / "demo.yaml",
        f"""
        name: demo
        root: {table.as_posix()}
        source: {{reader: csv, path: table.csv}}
        steps:
          - subject: {{column: pp}}
          - label: {{task: stress, column: cond, map: {{N: 0, T: 1}}}}
          - features: {{include: [keys]}}
        """,
    )

    (first, _) = ingest(load_spec(path))

    assert first.feature_names == ["keys"]


def test_the_root_can_be_overridden(table: Path, tmp_path_factory) -> None:
    elsewhere = spec(
        tmp_path_factory.mktemp("x"), {"reader": "csv", "path": "table.csv"}, BASIC
    )

    assert len(ingest(elsewhere, root=table)) == 2


def test_a_features_step_is_required(table: Path) -> None:
    with pytest.raises(DataError, match="features"):
        ingest(spec(table, {"reader": "csv", "path": "table.csv"}, BASIC[:2]))


def test_a_subject_is_required(table: Path) -> None:
    with pytest.raises(DataError, match="subject"):
        ingest(spec(table, {"reader": "csv", "path": "table.csv"}, BASIC[1:]))


def test_a_step_entry_has_exactly_one_step(table: Path) -> None:
    with pytest.raises(DataError, match="one step"):
        ingest(
            spec(
                table,
                {"reader": "csv", "path": "table.csv"},
                [{"subject": {"column": "pp"}, "features": {}}],
            )
        )


def test_unknown_steps_and_readers_are_plugin_errors(table: Path) -> None:
    with pytest.raises(ValueError, match="explode"):
        ingest(spec(table, {"reader": "csv", "path": "table.csv"}, [{"explode": {}}]))
    with pytest.raises(ValueError, match="xml"):
        ingest(spec(table, {"reader": "xml", "path": "table.csv"}, BASIC))


# --- sources -----------------------------------------------------------------------


def test_subject_placeholders_read_one_file_per_subject(tmp_path: Path) -> None:
    for user in ("user01", "user02"):
        for day in (1, 2):
            write(
                tmp_path / user / f"{user}_Day{day}.csv",
                f"""
                hr,stress
                {day},1
                """,
            )
    write(tmp_path / "user03" / "other_Day1.csv", "hr,stress\n9,1\n")  # name mismatch

    subjects = ingest(
        spec(
            tmp_path,
            {"reader": "csv", "path": "{subject}/{subject}_Day*.csv"},
            [
                {"label": {"task": "s", "column": "stress", "n_classes": 2}},
                {"features": {"include": ["hr"]}},
            ],
        )
    )

    assert [s.subject for s in subjects] == ["user01", "user02"]
    assert sorted(subjects[0].X[:, 0].tolist()) == [1.0, 2.0]


def test_a_source_that_matches_nothing_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="missing.csv"):
        ingest(spec(tmp_path, {"reader": "csv", "path": "missing.csv"}, BASIC))


def test_csv_reader_options(tmp_path: Path) -> None:
    path = write(tmp_path / "t.csv", "a;b\n1,5;x\n")

    df = readers.create("csv", {"sep": ";", "decimal": ","}).read(path)

    assert df["a"].tolist() == [1.5]


def test_npz_reader_expands_matrices_into_columns(tmp_path: Path) -> None:
    path = tmp_path / "s.npz"
    np.savez(path, acc=np.arange(6).reshape(3, 2), label=np.array([1, 1, 2]))

    df = readers.create("npz").read(path)

    assert list(df.columns) == ["acc_0", "acc_1", "label"]
    assert df["acc_1"].tolist() == [1, 3, 5]


def test_pickle_reader_follows_a_key_path(tmp_path: Path) -> None:
    path = tmp_path / "s.pkl"
    frame = pd.DataFrame({"a": [1, 2]})
    path.write_bytes(pickle.dumps({"computer": {"data": frame}}))

    df = readers.create("pickle", {"key": "computer.data"}).read(path)

    assert df["a"].tolist() == [1, 2]


def test_pickle_reader_needs_a_table(tmp_path: Path) -> None:
    path = tmp_path / "s.pkl"
    path.write_bytes(pickle.dumps({"computer": [1, 2]}))

    with pytest.raises(DataError, match="computer"):
        readers.create("pickle", {"key": "computer"}).read(path)


def test_excel_reader(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "t.xlsx"
    pd.DataFrame({"a": [1, 2]}).to_excel(path, index=False)

    assert readers.create("excel").read(path)["a"].tolist() == [1, 2]


def test_parquet_reader(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "t.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(path)

    assert readers.create("parquet").read(path)["a"].tolist() == [1, 2]


# --- steps, one by one ---------------------------------------------------------------


def run(step: str, params: dict, df: pd.DataFrame) -> pd.DataFrame:
    return steps.create(step, params).apply(df, ctx=None)


def test_subject_step_extracts_with_a_regex() -> None:
    df = run(
        "subject",
        {"column": "pp", "regex": r"(\d+)"},
        pd.DataFrame({"pp": ["PP1", "pp12"]}),
    )

    assert df["subject"].tolist() == ["1", "12"]


def test_subject_step_rejects_values_that_do_not_match() -> None:
    with pytest.raises(DataError, match="PPx"):
        run(
            "subject",
            {"column": "pp", "regex": r"(\d+)"},
            pd.DataFrame({"pp": ["PPx"]}),
        )


def test_replace_turns_codes_into_missing_values() -> None:
    df = pd.DataFrame({"a": [999, 1], "b": ["#VALUE!", "2"]})

    out = run("replace", {"values": {999: None, "#VALUE!": None}}, df)

    assert out["a"].isna().tolist() == [True, False]
    assert out["b"].isna().tolist() == [True, False]


def test_replace_can_be_limited_to_some_columns() -> None:
    df = pd.DataFrame({"a": [999], "b": [999]})

    out = run("replace", {"values": {999: None}, "columns": ["a"]}, df)

    assert out["a"].isna().all() and out["b"].tolist() == [999]


def test_select_keeps_the_rows_of_a_query() -> None:
    out = run(
        "select", {"query": "cond != 'X'"}, pd.DataFrame({"cond": ["N", "X", "T"]})
    )

    assert out["cond"].tolist() == ["N", "T"]


@pytest.mark.parametrize(
    "rule, values, expected",
    [
        ({"map": {"N": 0, "T": 1, "I": 1}}, ["N", "I", "?"], [0, 1, None]),
        ({"map": {1: 0, 2: 1}}, ["1", "2", "3"], [0, 1, None]),  # keys match text too
        ({"threshold": 3}, [1, 3, 5], [0, 1, 1]),
        ({"bins": [1.5, 2.5]}, [1, 2, 4], [0, 1, 2]),
        ({"round": True, "map": {1: 0, 2: 1, 3: 2}}, [1.2, 2.4, 2.6], [0, 1, 2]),
        ({"n_classes": 3}, [0, 2, 1], [0, 2, 1]),
    ],
)
def test_label_rules(rule: dict, values: list, expected: list) -> None:
    out = run(
        "label", {"task": "t", "column": "c", **rule}, pd.DataFrame({"c": values})
    )

    got = [None if pd.isna(v) else int(v) for v in out["label"]]
    assert got == expected


def test_label_needs_a_rule_or_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        steps.create("label", {"task": "t", "column": "c"})


def test_label_strategies_are_chosen_by_option(table: Path) -> None:
    label = {
        "label": {
            "default": "binary",
            "strategies": {
                "binary": {"task": "stress", "column": "keys", "threshold": 25},
                "three": {"task": "stress_3", "column": "keys", "bins": [15, 35]},
            },
        }
    }
    s = spec(
        table,
        {"reader": "csv", "path": "table.csv"},
        [BASIC[0], label, {"features": {"include": ["mouse"]}}],
    )

    binary = ingest(s)
    three = ingest(s, options={"label": "three"})

    assert binary[1].y.tolist() == [1, 1, 1] and binary[1].n_classes == 2
    assert three[1].y.tolist() == [1, 2, 2] and three[1].task == "stress_3"
    with pytest.raises(DataError, match="ordinal"):
        ingest(s, options={"label": "ordinal"})


def test_features_include_exclude_and_regex() -> None:
    df = pd.DataFrame(
        {"subject": ["1"], "label": [0], "hr_mean": [1], "hr_std": [2], "blok": [3]}
    )

    assert list(run("features", {"regex": "^hr_"}, df).columns) == [
        "subject",
        "label",
        "hr_mean",
        "hr_std",
    ]
    assert list(run("features", {"exclude": ["blok"]}, df).columns)[2:] == [
        "hr_mean",
        "hr_std",
    ]
    with pytest.raises(DataError, match="nope"):
        run("features", {"include": ["nope"]}, df)


def test_features_reject_timestamps() -> None:
    df = pd.DataFrame(
        {"subject": ["1"], "label": [0], "t": pd.to_datetime(["2020-01-01"])}
    )

    with pytest.raises(DataError, match="'t'"):
        run("features", {}, df)


def test_features_reject_text_columns() -> None:
    df = pd.DataFrame({"subject": ["1"], "label": [0], "note": ["hello"]})

    with pytest.raises(DataError, match="note"):
        run("features", {}, df)


def test_features_keep_missing_values_for_later_imputation() -> None:
    df = pd.DataFrame({"subject": ["1", "1"], "label": [0, 1], "hr": [1.0, None]})

    assert run("features", {}, df)["hr"].isna().tolist() == [False, True]


# --- join ------------------------------------------------------------------------------


def test_join_on_keys(tmp_path: Path) -> None:
    write(tmp_path / "a.csv", "pp,blok,keys\n1,1,10\n1,2,20\n")
    write(tmp_path / "b.csv", "pp,blok,hr\n1,1,60\n1,2,70\n")
    out = steps.create(
        "join", {"source": {"reader": "csv", "path": "b.csv"}, "by": ["pp", "blok"]}
    ).apply(pd.read_csv(tmp_path / "a.csv"), ctx=_ctx(tmp_path))

    assert out["hr"].tolist() == [60, 70]


def test_join_pads_keys_that_lost_their_trailing_zeros(tmp_path: Path) -> None:
    # A sheet stored 20120918T131600000 as 20120918T1316 on one side only.
    write(tmp_path / "a.csv", "pp,timestamp,keys\n1,20120918T131600000,10\n")
    write(tmp_path / "b.csv", "pp,timestamp,hr\n1,20120918T1316,60\n")
    params = {"source": {"reader": "csv", "path": "b.csv"}, "by": ["pp", "timestamp"]}
    left = pd.read_csv(tmp_path / "a.csv")

    out = steps.create("join", params | {"pad": {"timestamp": 18}}).apply(
        left, ctx=_ctx(tmp_path)
    )

    assert out["hr"].tolist() == [60]
    assert out["timestamp"].tolist() == ["20120918T131600000"]


def test_join_by_time_floors_both_sides(tmp_path: Path) -> None:
    write(
        tmp_path / "u1" / "labels.csv",
        "TS,stress\n2020-01-01 10:00:40,3\n2020-01-01 10:00:50,4\n",
    )
    left = pd.DataFrame(
        {
            "subject": ["u1", "u1"],
            "timestamp": ["2020-01-01 10:00:05", "2020-01-01 10:01:05"],
            "hr": [1, 2],
        }
    )

    out = steps.create(
        "join",
        {
            "source": {"reader": "csv", "path": "{subject}/labels.csv"},
            "by": ["subject"],
            "time": {"left": "timestamp", "right": "TS", "floor": "min"},
            "deduplicate": True,
        },
    ).apply(left, ctx=_ctx(tmp_path))

    assert out["hr"].tolist() == [1]
    assert out["stress"].tolist() == [3]  # the first annotation of that minute


def _ctx(root: Path):
    from onion_fl.data.ingest import IngestContext

    return IngestContext(root=root, options={})


# --- window -------------------------------------------------------------------------------


def test_window_computes_stats_per_channel_and_subject() -> None:
    df = pd.DataFrame(
        {
            "subject": ["a"] * 4 + ["b"] * 4,
            "x": [1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0],
            "label": [0, 0, 1, 1, 1, 1, 1, 1],
        }
    )

    out = run(
        "window",
        {"size": 2, "overlap": 0.5, "stats": ["mean", "max"], "columns": ["x"]},
        df,
    )

    assert list(out.columns) == ["subject", "label", "x_mean", "x_max"]
    a = out[out["subject"] == "a"]
    assert a["x_mean"].tolist() == [1.5, 2.5, 3.5]
    assert a["label"].tolist() == [0, 0, 1]  # round(mean), half to even
    assert len(out[out["subject"] == "b"]) == 3


def test_window_drops_windows_with_rows_outside_the_classes() -> None:
    df = pd.DataFrame(
        {"subject": ["a"] * 4, "x": [1.0, 2.0, 3.0, 4.0], "label": [0, None, 1, 1]}
    )

    out = run("window", {"size": 2, "overlap": 0.0, "columns": ["x"]}, df)

    assert out["x_mean"].tolist() == [3.5]


def test_window_stat_names_follow_channel_then_stat() -> None:
    df = pd.DataFrame(
        {"subject": ["a"] * 2, "x": [1.0, 3.0], "y": [0.0, 0.0], "label": [0, 0]}
    )

    out = run("window", {"size": 2}, df)

    assert list(out.columns)[2:] == [
        f"{c}_{s}" for c in ("x", "y") for s in ("mean", "std", "min", "max", "median")
    ]
    assert out["x_std"].tolist() == [1.0]  # population std, as before


def test_window_never_summarises_the_label_source(tmp_path: Path) -> None:
    # The raw label column is numeric: without care it would become condition_mean.
    write(tmp_path / "S2" / "s.csv", "x,condition\n1,1\n2,1\n3,2\n4,2\n")

    (s,) = ingest(
        spec(
            tmp_path,
            {"reader": "csv", "path": "{subject}/s.csv"},
            [
                {"label": {"task": "t", "column": "condition", "map": {1: 0, 2: 1}}},
                {"window": {"size": 2, "stats": ["mean"]}},
                {"features": {}},
            ],
        )
    )

    assert s.feature_names == ["x_mean"]


def test_window_needs_subjects() -> None:
    with pytest.raises(DataError, match="subject"):
        run("window", {"size": 2}, pd.DataFrame({"x": [1.0, 2.0]}))


# --- options and when -------------------------------------------------------------------


def test_steps_can_depend_on_options(table: Path) -> None:
    s = spec(
        table,
        {"reader": "csv", "path": "table.csv"},
        [
            {"select": {"query": "pp == 2"}, "when": {"only_two": True}},
            *BASIC,
        ],
        options={"only_two": False},
    )

    assert len(ingest(s)) == 2
    assert [x.subject for x in ingest(s, options={"only_two": True})] == ["2"]


def test_when_accepts_a_list_of_values(table: Path) -> None:
    s = spec(
        table,
        {"reader": "csv", "path": "table.csv"},
        [{"select": {"query": "pp == 2"}, "when": {"mode": ["b", "c"]}}, *BASIC],
        options={"mode": "a"},
    )

    assert len(ingest(s, options={"mode": "c"})) == 1


def test_unknown_options_are_rejected(table: Path) -> None:
    with pytest.raises(DataError, match="colour"):
        ingest(
            spec(table, {"reader": "csv", "path": "table.csv"}, BASIC),
            options={"colour": 1},
        )


def test_when_must_name_a_declared_option(table: Path) -> None:
    with pytest.raises(DataError, match="mode"):
        ingest(
            spec(
                table,
                {"reader": "csv", "path": "table.csv"},
                [{"select": {"query": "pp == 2"}, "when": {"mode": "b"}}, *BASIC],
            )
        )


def test_subjects_come_in_natural_order(tmp_path: Path) -> None:
    write(tmp_path / "t.csv", "pp,cond,keys\n10,N,1\n2,T,2\n")

    subjects = ingest(spec(tmp_path, {"reader": "csv", "path": "t.csv"}, BASIC))

    assert [s.subject for s in subjects] == ["2", "10"]


# --- source extras, per-file processing and descriptor checks (issue #87) ---------------


def test_source_can_normalize_column_names(tmp_path: Path) -> None:
    write(tmp_path / "t.csv", " PP ,Condition,Key Strokes\n1,N,3\n")

    (s,) = ingest(
        spec(
            tmp_path,
            {"reader": "csv", "path": "t.csv", "normalize_columns": True},
            [
                {"subject": {"column": "pp"}},
                {
                    "label": {
                        "task": "t",
                        "column": "condition",
                        "map": {"N": 0, "T": 1},
                    }
                },
                {"features": {}},
            ],
        )
    )

    assert s.feature_names == ["key_strokes"]


def test_options_fill_placeholders_in_root_and_paths(tmp_path: Path) -> None:
    write(tmp_path / "selection2" / "t.csv", "pp,cond,keys\n1,N,3\n")
    s = DatasetSpec(
        name="demo",
        root=str(tmp_path / "{selection}"),
        source={"reader": "csv", "path": "t.csv"},
        options={"selection": "selection1"},
        steps=BASIC,
    )

    assert len(ingest(s, options={"selection": "selection2"})) == 1
    with pytest.raises(DataError, match="selection1"):
        ingest(s)


def test_each_file_runs_the_steps_on_its_own(tmp_path: Path) -> None:
    # Windows never span two files: each one is processed separately.
    for day in (1, 2):
        write(tmp_path / "u1" / f"d{day}.csv", "x,label\n1,0\n2,0\n3,0\n")

    (s,) = ingest(
        spec(
            tmp_path,
            {"reader": "csv", "path": "{subject}/d*.csv"},
            [
                {"label": {"task": "t", "column": "label", "n_classes": 2}},
                {"window": {"size": 2, "overlap": 0.5, "stats": ["mean"]}},
                {"features": {}},
            ],
        )
    )

    assert s.X[:, 0].tolist() == [1.5, 2.5, 1.5, 2.5]


def test_files_must_end_with_the_same_columns(tmp_path: Path) -> None:
    write(tmp_path / "a" / "f.csv", "pp,cond,keys\n1,N,3\n")
    write(tmp_path / "b" / "f.csv", "pp,cond,mouse\n2,N,3\n")

    with pytest.raises(DataError, match="mouse"):
        ingest(spec(tmp_path, {"reader": "csv", "path": "*/f.csv"}, BASIC))


def test_subjects_with_too_few_samples_are_dropped(table: Path) -> None:
    s = spec(
        table,
        {"reader": "csv", "path": "table.csv"},
        BASIC,
        min_samples_per_subject=3,
    )

    assert ingest(s) == []


def test_check_spec_builds_every_step_without_data(tmp_path: Path) -> None:
    from onion_fl.data.ingest import check_spec

    good = spec(tmp_path, {"reader": "csv", "path": "nothing.csv"}, BASIC)
    bad = spec(tmp_path, {"reader": "csv", "path": "x.csv"}, [{"window": {"size": 0}}])

    check_spec(good)
    with pytest.raises(ValueError, match="size"):
        check_spec(bad)


def test_option_names_cannot_clash_with_subject(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="subject"):
        ingest(
            spec(
                tmp_path,
                {"reader": "csv", "path": "t.csv"},
                BASIC,
                options={"subject": 1},
            )
        )


# --- wesad_pickle (format fixture: the structure of a WESAD subject file) ----------------


def wesad_file(path: Path, seconds: int = 2) -> Path:
    n = 700 * seconds
    data = {
        "label": np.repeat([1, 2], n // 2),
        "signal": {
            "chest": {  # the published files spell these two "Resp" and "Temp"
                "ECG": np.arange(n, dtype=float).reshape(-1, 1),
                "Resp": np.arange(n, dtype=float).reshape(-1, 1),
            },
            "wrist": {
                "ACC": np.arange(32 * seconds * 3, dtype=float).reshape(-1, 3),
                "EDA": np.arange(4 * seconds, dtype=float).reshape(-1, 1),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(data))
    return path


def test_wesad_reader_holds_wrist_signals_at_the_label_rate(tmp_path: Path) -> None:
    path = wesad_file(tmp_path / "S2" / "S2.pkl")
    reader = readers.create(
        "wesad_pickle", {"location": "wrist", "signals": ["ACC", "EDA"]}
    )

    df = reader.read(path)

    assert list(df.columns) == ["acc_0", "acc_1", "acc_2", "eda_0", "condition"]
    assert len(df) == 1400
    assert df["eda_0"].iloc[[0, 174, 175, 1399]].tolist() == [0, 0, 1, 7]  # 4 Hz held
    assert df["acc_1"].iloc[700] == 32 * 3 + 1  # sample 32 of the 32 Hz signal
    assert df["condition"].iloc[[0, 1399]].tolist() == [1, 2]


def test_wesad_reader_chest_is_sample_aligned(tmp_path: Path) -> None:
    path = wesad_file(tmp_path / "S2" / "S2.pkl")
    reader = readers.create("wesad_pickle", {"location": "chest", "signals": ["ECG"]})

    assert reader.read(path)["ecg_0"].tolist() == list(range(1400))


def test_wesad_reader_finds_signals_whatever_their_case(tmp_path: Path) -> None:
    path = wesad_file(tmp_path / "S2" / "S2.pkl")
    reader = readers.create("wesad_pickle", {"location": "chest", "signals": ["RESP"]})

    assert reader.read(path)["resp_0"].tolist() == list(range(1400))


def test_wesad_reader_names_a_signal_the_file_lacks(tmp_path: Path) -> None:
    path = wesad_file(tmp_path / "S2" / "S2.pkl")
    reader = readers.create("wesad_pickle", {"location": "chest", "signals": ["EMG"]})

    with pytest.raises(DataError, match="EMG"):
        reader.read(path)


def test_wesad_reader_truncates_to_the_shortest_signal(tmp_path: Path) -> None:
    path = wesad_file(tmp_path / "S2" / "S2.pkl")
    data = pickle.loads(path.read_bytes())
    data["signal"]["wrist"]["EDA"] = data["signal"]["wrist"]["EDA"][:6]  # 1.5 s
    path.write_bytes(pickle.dumps(data))

    df = readers.create("wesad_pickle", {"signals": ["EDA"]}).read(path)

    assert len(df) == 1050


def test_wesad_reader_rejects_unknown_signals() -> None:
    with pytest.raises(ValueError, match="BVP"):
        readers.create("wesad_pickle", {"location": "chest", "signals": ["BVP"]})


# --- time (continuum C3) ------------------------------------------------------------


STAMPS = """
pp,cond,stamp,keys
1,N,20120918T131600000,10
1,T,20120918T131700000,20
1,T,20120918T131900000,30
2,N,20120919T090000000,40
2,T,20120919T090100000,50
"""


def test_a_time_column_becomes_seconds_and_never_a_feature(tmp_path: Path) -> None:
    write(tmp_path / "table.csv", STAMPS)
    time = {"time": {"column": "stamp", "format": "%Y%m%dT%H%M%S%f"}}

    first, second = ingest(
        spec(
            tmp_path,
            {"reader": "csv", "path": "table.csv"},
            [BASIC[0], time, BASIC[1], {"features": {}}],
        )
    )

    assert first.feature_names == ["keys"]  # the stamp is meta: never a feature
    assert first.t.tolist() == [0.0, 60.0, 180.0]  # from each subject's first row
    assert second.t.tolist() == [0.0, 60.0]


def test_time_can_come_from_the_row_position_at_a_rate(tmp_path: Path) -> None:
    write(tmp_path / "S2" / "s.csv", "x,condition\n1,1\n2,1\n3,2\n4,2\n5,2\n")
    steps = [
        {"time": {"rate": 2}},
        {"label": {"task": "t", "column": "condition", "map": {1: 0, 2: 1}}},
        {"features": {}},
    ]

    (s,) = ingest(spec(tmp_path, {"reader": "csv", "path": "{subject}/s.csv"}, steps))

    assert s.t.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert s.feature_names == ["x"]


def test_a_window_is_timed_at_its_last_row(tmp_path: Path) -> None:
    write(tmp_path / "S2" / "s.csv", "x,condition\n1,1\n2,1\n3,2\n4,2\n5,2\n6,2\n")
    steps = [
        {"time": {"rate": 1}},
        {"label": {"task": "t", "column": "condition", "map": {1: 0, 2: 1}}},
        {"window": {"size": 2, "stats": ["mean"]}},
        {"features": {}},
    ]

    (s,) = ingest(spec(tmp_path, {"reader": "csv", "path": "{subject}/s.csv"}, steps))

    assert s.feature_names == ["x_mean"]  # time is not a channel
    assert s.t.tolist() == [0.0, 2.0, 4.0]  # ends at rows 1, 3, 5, from the first


def test_a_time_step_needs_one_source(tmp_path: Path) -> None:
    write(tmp_path / "table.csv", STAMPS)

    with pytest.raises(Exception, match="column or rate"):
        ingest(
            spec(
                tmp_path,
                {"reader": "csv", "path": "table.csv"},
                [BASIC[0], {"time": {}}, BASIC[1], {"features": {}}],
            )
        )


def test_rows_without_a_time_are_an_error(tmp_path: Path) -> None:
    write(tmp_path / "table.csv", STAMPS.replace("20120918T131700000", "nope"))
    time = {"time": {"column": "stamp", "format": "%Y%m%dT%H%M%S%f"}}

    with pytest.raises(DataError, match="time"):
        ingest(
            spec(
                tmp_path,
                {"reader": "csv", "path": "table.csv"},
                [BASIC[0], time, BASIC[1], {"features": {}}],
            )
        )


def test_a_time_can_be_padded_where_a_sheet_dropped_trailing_zeros(
    tmp_path: Path,
) -> None:
    write(tmp_path / "table.csv", STAMPS.replace("20120918T131700000", "20120918T1317"))
    time = {"time": {"column": "stamp", "format": "%Y%m%dT%H%M%S%f", "pad_to": 18}}

    first, _ = ingest(
        spec(
            tmp_path,
            {"reader": "csv", "path": "table.csv"},
            [BASIC[0], time, BASIC[1], {"features": {}}],
        )
    )

    assert first.t.tolist() == [0.0, 60.0, 180.0]
