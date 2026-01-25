from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flower_basic.datasets.swell_federated import (
    _auto_assign_nodes,
    _read_config,
    _split_subjects,
    _split_subjects_with_test,
    plan_and_materialize_swell_federated,
)


def _mock_load_swell_all_samples(*_args, **_kwargs):
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    subjects = np.array(["1", "1", "2", "2", "3", "3"], dtype=object)
    meta = {
        "n_samples": 6,
        "n_features": 4,
        "n_subjects": 3,
        "modalities": ["mock"],
    }
    return X, y, subjects, meta


def test_read_config_validates_split_sum(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.json"
    cfg = {
        "split": {"train": 0.6, "val": 0.3, "test": 0.2},
        "dataset": {"data_dir": "data/SWELL"},
        "federation": {"mode": "manual", "manual_assignments": {"fog_0": ["1"]}},
    }
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="sum to 1.0"):
        _read_config(cfg_path)


def test_read_config_default_strategy(tmp_path: Path) -> None:
    cfg_path = tmp_path / "ok.json"
    cfg = {
        "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        "dataset": {"data_dir": "data/SWELL"},
        "federation": {"mode": "manual", "manual_assignments": {"fog_0": ["1"]}},
    }
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    cfg_obj = _read_config(cfg_path)
    assert cfg_obj.split_strategy == "per_subject"


def test_split_subjects_is_deterministic() -> None:
    subjects = ["1", "2", "3", "4", "5"]
    train_1, val_1, test_1 = _split_subjects(subjects, 0.6, 0.2, 0.2, seed=123)
    train_2, val_2, test_2 = _split_subjects(subjects, 0.6, 0.2, 0.2, seed=123)

    assert train_1 == train_2
    assert val_1 == val_2
    assert test_1 == test_2
    assert set(train_1).isdisjoint(val_1)
    assert set(val_1).isdisjoint(test_1)
    assert set(train_1 + val_1 + test_1) == set(subjects)


def test_split_subjects_with_test_holdout() -> None:
    subjects = ["1", "2", "3", "4"]
    test_subjects = ["4"]
    train_ids, val_ids, test_ids = _split_subjects_with_test(
        subjects, train=0.7, val=0.3, seed=42, test_subjects=test_subjects
    )

    assert sorted(test_ids) == ["4"]
    assert "4" not in train_ids
    assert "4" not in val_ids
    assert set(train_ids + val_ids + test_ids) == set(subjects)


def test_auto_assign_nodes_percentages() -> None:
    subjects = ["1", "2", "3", "4"]
    mapping = _auto_assign_nodes(subjects, num_nodes=2, percentages=[0.75, 0.25], seed=1)

    assert len(mapping["fog_0"]) == 3
    assert len(mapping["fog_1"]) == 1
    assert set(mapping["fog_0"] + mapping["fog_1"]) == set(subjects)


def test_plan_and_materialize_per_subject(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.datasets.swell_federated.load_swell_all_samples",
        _mock_load_swell_all_samples,
    )

    cfg = {
        "dataset": {"data_dir": "dummy", "modalities": ["mock"]},
        "split": {
            "train": 0.5,
            "val": 0.25,
            "test": 0.25,
            "seed": 7,
            "scaler": "none",
            "strategy": "per_subject",
        },
        "federation": {
            "mode": "manual",
            "manual_assignments": {"fog_0": ["1", "2"], "fog_1": ["3"]},
            "output_dir": str(tmp_path),
            "run_name": "per_subject",
        },
    }
    cfg_path = tmp_path / "per_subject.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = plan_and_materialize_swell_federated(str(cfg_path))
    out_dir = Path(result["output_dir"])
    manifest_path = out_dir / "manifest.json"

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["split"]["strategy"] == "per_subject"
    assert len(manifest["global_subjects"]["all"]) == 3
    assert len(manifest["global_subjects"]["train"]) == 3
    assert (out_dir / "fog_0" / "train.npz").exists()
    assert (out_dir / "fog_0" / "subject_1" / "train.npz").exists()


def test_plan_and_materialize_global_strategy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.datasets.swell_federated.load_swell_all_samples",
        _mock_load_swell_all_samples,
    )

    cfg = {
        "dataset": {"data_dir": "dummy", "modalities": ["mock"]},
        "split": {
            "train": 0.5,
            "val": 0.25,
            "test": 0.25,
            "seed": 3,
            "scaler": "none",
            "strategy": "global",
        },
        "federation": {
            "mode": "manual",
            "manual_assignments": {"fog_0": ["1", "2"], "fog_1": ["3"]},
            "output_dir": str(tmp_path),
            "run_name": "global",
        },
    }
    cfg_path = tmp_path / "global.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = plan_and_materialize_swell_federated(str(cfg_path))
    out_dir = Path(result["output_dir"])
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    train_set = set(manifest["global_subjects"]["train"])
    val_set = set(manifest["global_subjects"]["val"])
    test_set = set(manifest["global_subjects"]["test"])

    assert manifest["config"]["split"]["strategy"] == "global"
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set | test_set == set(manifest["global_subjects"]["all"])
