"""Tests for the prepared-data cache and the dataset card (issue #86).

Format fixtures only: they check files and statistics, nothing is trained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from onion_fl.data.cache import cache_key, inspect, load_prepared, prepare
from onion_fl.data.contract import SubjectData
from onion_fl.data.ingest import DatasetSpec

STEPS = [
    {"subject": {"column": "pp"}},
    {"label": {"task": "stress", "column": "cond", "map": {"N": 0, "T": 1}}},
    {"features": {"exclude": ["blok"]}},
]


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "table.csv").write_text(
        "pp,blok,cond,keys\n1,1,N,10\n1,2,T,20\n2,1,N,30\n2,2,T,40\n", encoding="utf-8"
    )
    return root


def make_spec(root: Path, **extra) -> DatasetSpec:
    return DatasetSpec(
        name="demo",
        root=str(root),
        source={"reader": "csv", "path": "table.csv"},
        steps=STEPS,
        **extra,
    )


def test_prepare_writes_one_npz_per_subject_and_a_meta(
    raw: Path, tmp_path: Path
) -> None:
    out = prepare(make_spec(raw), cache_dir=tmp_path / "cache")

    assert out.parent == tmp_path / "cache" / "demo"
    assert sorted(p.name for p in out.iterdir()) == [
        "meta.json",
        "subject_1.npz",
        "subject_2.npz",
    ]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["dataset"] == "demo" and meta["key"] == out.name
    assert meta["feature_names"] == ["keys"]
    assert set(meta["subjects"]) == {"1", "2"}
    assert len(meta["subjects"]["1"]["sha256"]) == 64


def test_prepared_data_loads_back_identical(raw: Path, tmp_path: Path) -> None:
    from onion_fl.data.ingest import ingest

    out = prepare(make_spec(raw), cache_dir=tmp_path / "cache")

    loaded, direct = load_prepared(out), ingest(make_spec(raw))
    assert [s.subject for s in loaded] == [s.subject for s in direct]
    for a, b in zip(loaded, direct, strict=True):
        assert a.meta == b.meta
        np.testing.assert_array_equal(a.X, b.X)
        np.testing.assert_array_equal(a.y, b.y)


def test_the_key_depends_on_spec_and_options_not_on_the_root(
    raw: Path, tmp_path: Path
) -> None:
    spec = make_spec(raw, options={"strict": False})

    assert cache_key(spec) == cache_key(make_spec(tmp_path, options={"strict": False}))
    assert cache_key(spec) == cache_key(spec, {"strict": False})  # defaults resolved
    assert cache_key(spec) != cache_key(spec, {"strict": True})


def test_a_different_step_changes_the_key(raw: Path) -> None:
    other = DatasetSpec(
        name="demo",
        root=str(raw),
        source={"reader": "csv", "path": "table.csv"},
        steps=[*STEPS[:2], {"features": {"include": ["keys"]}}],
    )

    assert cache_key(make_spec(raw)) != cache_key(other)


def test_prepare_reuses_the_cache_unless_forced(raw: Path, tmp_path: Path) -> None:
    first = prepare(make_spec(raw), cache_dir=tmp_path / "cache")
    (raw / "table.csv").unlink()

    assert prepare(make_spec(raw), cache_dir=tmp_path / "cache") == first
    with pytest.raises(ValueError, match="table.csv"):
        prepare(make_spec(raw), cache_dir=tmp_path / "cache", force=True)
    assert (first / "meta.json").exists()  # a failed rebuild leaves the old cache


# --- dataset card ----------------------------------------------------------------------


def subj(name: str, X, y) -> SubjectData:
    return SubjectData(
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y),
        dataset="demo",
        subject=name,
        task="stress",
        n_classes=2,
        feature_names=["leak", "noise", "flat"],
    )


def test_inspect_summarises_and_warns_about_leaky_features() -> None:
    subjects = [
        subj("1", [[0, 1, 5], [1, 0, 5], [0, np.nan, 5]], [0, 1, 0]),
        subj("2", [[1, 1, 5], [0, 0, 5]], [1, 0]),
    ]

    card = inspect(subjects)

    assert card["n_subjects"] == 2 and card["n_samples"] == 5
    assert card["n_features"] == 3
    assert card["class_counts"] == [3, 2]
    assert card["samples_per_subject"] == {"1": 3, "2": 2}
    assert card["missing"] == {"noise": pytest.approx(0.2)}
    assert len(card["warnings"]) == 1
    assert "leak" in card["warnings"][0]


def test_inspect_ignores_constant_features() -> None:
    card = inspect([subj("1", [[0, 1, 5], [0, 0, 5]], [0, 1])])

    assert all("flat" not in w for w in card["warnings"])


def test_a_tampered_cache_is_detected(raw: Path, tmp_path: Path) -> None:
    out = prepare(make_spec(raw), cache_dir=tmp_path / "cache")
    np.savez(
        out / "subject_1.npz", X=np.zeros((2, 1), np.float32), y=np.zeros(2, np.int64)
    )

    with pytest.raises(ValueError, match="digest"):
        load_prepared(out)


def test_losing_a_race_to_another_process_keeps_its_cache(
    raw: Path, tmp_path: Path, monkeypatch
) -> None:
    import onion_fl.data.cache as cache

    real_ingest = cache.ingest

    def ingest_while_another_process_finishes(*args, **kwargs):
        monkeypatch.setattr(cache, "ingest", real_ingest)
        won = prepare(
            make_spec(raw), cache_dir=tmp_path / "cache"
        )  # the other process wins
        (won / "reader.lock").write_text(
            "someone is reading this cache", encoding="utf-8"
        )
        return real_ingest(*args, **kwargs)

    monkeypatch.setattr(cache, "ingest", ingest_while_another_process_finishes)

    out = prepare(make_spec(raw), cache_dir=tmp_path / "cache")

    assert (
        out / "reader.lock"
    ).exists()  # the winner's cache was not replaced under its readers
    assert len(load_prepared(out)) == 2
    assert sorted(p.name for p in out.parent.iterdir()) == [
        out.name
    ]  # no stray temp folder
