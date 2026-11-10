"""Write Fed-Heart-Disease (FLamby) as one CSV per center and split.

The source is the four ``processed.<center>.data`` files of the UCI Heart
Disease dataset (Janosi et al., 1988, CC BY 4.0, doi:10.24432/C52P4X), the
files FLamby downloads (``fed_heart_disease/dataset_creation_scripts/
download.py``, which checks the MD5 sums below). This script follows FLamby's
``HeartDiseaseRaw`` (``fed_heart_disease/dataset.py``, main branch, read on
2026-11-10) step by step:

1. ``?`` is missing; the columns slope, ca and thal (10, 11, 12) are dropped,
   then every row with a missing value.
2. Each center is split on its own with scikit-learn's
   ``train_test_split(np.arange(n), train_size=0.66, test_size=0.34,
   random_state=43, shuffle=True, stratify=...)``, stratified on the binary
   label when both classes have more than two rows (always the case here).
3. cp and restecg are one-hot encoded, dropping the first level (pandas
   ``get_dummies(drop_first=True)`` over the four centers: cp 2-4, restecg 1-2),
   and the dummies go after the other features, in FLamby's column order.

The label stays as ``num`` (0-4): the descriptor turns it into disease (>= 1).
The scaling is left to Onion-FL, fitted on the training rows only. The counts
per center are checked against FLamby's README (train/test 199/104, 172/89,
30/16 and 85/45). The rows keep their order in the source file.

Each center gives ``<center>.csv`` (its train split) and ``<center>_test.csv``,
so the descriptor reads one subject per file and never sees the center as a
column. Run it with the repository's environment, isolated (the inputs are
downloaded files)::

    .venv/Scripts/python -I scripts/heart_disease_flamby.py \\
        data/HeartDisease/raw data/HeartDisease/flamby
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# FLamby's centers, their MD5 sums (download.py) and the train/test counts (README).
CENTERS = {
    "cleveland": ("2d91a8ff69cfd9616aa47b59d6f843db", 199, 104),
    "hungarian": ("22e96bee155b5973568101c93b3705f6", 172, 89),
    "switzerland": ("9a87f7577310b3917730d06ba9349e20", 30, 16),
    "va": ("4249d03ca7711e84f4444768c9426170", 85, 45),
}
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]  # fmt: skip
DROPPED = ["slope", "ca", "thal"]  # FLamby's drop([10, 11, 12])
LEVELS = {"cp": [1, 2, 3, 4], "restecg": [0, 1, 2]}  # first level dropped
TRAIN_FRACTION = 0.66
RANDOM_STATE = 43


def read_center(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    return df.drop(columns=DROPPED).dropna(axis=0).reset_index(drop=True)


def split(df: pd.DataFrame) -> np.ndarray:
    """True for the test rows, as FLamby's split of one center."""
    labels = (df["num"] != 0).astype(int)
    stratify = (
        labels if labels.nunique() > 1 and (labels.value_counts() > 2).all() else None
    )
    _, test = train_test_split(
        np.arange(len(df)),
        test_size=1.0 - TRAIN_FRACTION,
        train_size=TRAIN_FRACTION,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify,
    )
    mask = np.zeros(len(df), dtype=bool)
    mask[test] = True
    return mask


def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column, levels in LEVELS.items():
        unknown = sorted(set(out[column]) - set(levels))
        if unknown:
            raise SystemExit(f"{column}: unexpected levels {unknown}")
        for level in levels[1:]:
            out[f"{column}_{level}"] = (out[column] == level).astype(int)
    # The dummies go last, after the other features, as get_dummies puts them.
    label = out.pop("num")
    return out.drop(columns=list(LEVELS)).assign(num=label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path, help="folder with processed.<center>.data")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for center, (md5, n_train, n_test) in CENTERS.items():
        path = args.raw / f"processed.{center}.data"
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != md5:
            raise SystemExit(f"{path}: MD5 {digest}, FLamby expects {md5}")
        df = read_center(path)
        test = split(df)
        counts = (int((~test).sum()), int(test.sum()))
        if counts != (n_train, n_test):
            raise SystemExit(f"{center}: train/test {counts}, FLamby has "
                             f"{(n_train, n_test)}")  # fmt: skip
        table = one_hot(df)
        table[~test].to_csv(args.out / f"{center}.csv", index=False)
        table[test].to_csv(args.out / f"{center}_test.csv", index=False)
        rows = ",".join(str(i) for i in np.flatnonzero(test))
        print(
            f"{center}: train {counts[0]}, test {counts[1]}, test rows sha256 "
            f"{hashlib.sha256(rows.encode()).hexdigest()[:16]}"
        )


if __name__ == "__main__":
    main()
