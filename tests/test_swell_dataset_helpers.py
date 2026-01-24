from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flower_basic.datasets.swell import (
    _coerce_numeric_dataframe,
    _normalize_subject_series,
    _try_read_csv,
    load_swell_all_samples,
)


def test_normalize_subject_series() -> None:
    series = pd.Series(
        ["P01", "participant_2", "subject003", "id004", "05", "abc"]
    )
    normalized = _normalize_subject_series(series)
    assert normalized.tolist() == ["1", "2", "3", "4", "5", "abc"]


def test_coerce_numeric_dataframe() -> None:
    df = pd.DataFrame(
        {
            "a": ["1,234", "2,345"],
            "b": ["1.234,5", "2.345,6"],
            "c": ["10", "20"],
        }
    )
    coerced = _coerce_numeric_dataframe(df)
    assert coerced.isnull().sum().sum() == 0
    assert np.issubdtype(coerced["a"].dtype, np.number)
    assert np.issubdtype(coerced["b"].dtype, np.number)
    assert np.issubdtype(coerced["c"].dtype, np.number)


def test_try_read_csv_semicolon(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("col1;col2\n1;2\n3;4\n", encoding="utf-8")
    df = _try_read_csv(csv_path)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]


def test_load_swell_all_samples_minimal(tmp_path: Path) -> None:
    feature_dir = tmp_path / "per sensor"
    feature_dir.mkdir(parents=True)
    csv_path = feature_dir / "computer_features.csv"
    csv_path.write_text(
        "participant,condition,feat1,feat2\n"
        "1,No stress,0.1,1.2\n"
        "1,Time pressure,0.2,1.3\n",
        encoding="utf-8",
    )

    X, y, subjects, info = load_swell_all_samples(
        data_dir=tmp_path, modalities=["computer"], normalize_features=False
    )

    assert X.shape == (2, 2)
    assert y.shape == (2,)
    assert subjects.shape == (2,)
    assert info["n_features"] == 2
