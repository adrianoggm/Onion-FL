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
from onion_fl.data.ingest import readers

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
