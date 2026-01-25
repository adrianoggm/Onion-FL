#!/usr/bin/env python3
"""Inspect SWELL dataset integrity using real data only."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

DATA_PATH = Path("data/SWELL")


def _find_candidate_files() -> List[Path]:
    """Return CSV files that contain SWELL features."""
    patterns = [
        "*Computer*.csv",
        "*Facial*.csv",
        "*Posture*.csv",
        "*Physiology*.csv",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(DATA_PATH.rglob(pattern))
    return sorted(files)


def analyze_file(path: Path) -> None:
    """Print basic statistics for a SWELL feature file."""
    df = pd.read_csv(path)
    print(f"\nAnalysing {path.relative_to(DATA_PATH)}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        zero_var = (numeric_df.var() == 0).sum()
        print(f"Zero-variance features: {zero_var}/{numeric_df.shape[1]}")

    if "Condition" in df.columns:
        print("Condition distribution:")
        print(df["Condition"].value_counts())

    duplicates = df.duplicated().sum()
    if duplicates:
        ratio = duplicates / len(df)
        print(f"Duplicate rows: {duplicates} ({ratio:.2%})")


def main() -> None:
    """Run the SWELL data inspection without synthetic fallbacks."""
    if not DATA_PATH.exists():
        raise SystemExit(
            "SWELL directory not found. Please provide real data at data/SWELL."
        )

    candidates = _find_candidate_files()
    if not candidates:
        raise SystemExit(
            "No SWELL CSV files were found. Ensure the dataset is extracted."
        )

    print(f"Found {len(candidates)} SWELL feature files.")
    for path in candidates[:5]:
        analyze_file(path)


if __name__ == "__main__":
    main()
