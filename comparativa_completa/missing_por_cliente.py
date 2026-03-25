#!/usr/bin/env python3
"""Missing values per client (subject) for SWELL physiology and WESAD raw signals.

Outputs:
- swell_missing_by_subject.csv
- wesad_missing_by_subject.csv
- missing_by_subject_tables.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SWELL_PHYSIO_FILE = "D - Physiology features (HR_HRV_SCL - final).csv"


def _coerce_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        if pd.api.types.is_numeric_dtype(result[col]):
            continue
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _read_csv_best_effort(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "latin-1", "cp1252"]
    last_exc: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python", **kwargs)
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Could not read CSV {path} ({last_exc})")


def _iter_csv_chunks(path: Path, chunksize: int):
    encodings = ["utf-8", "latin-1", "cp1252"]
    last_exc: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path, encoding=enc, sep=None, engine="python", chunksize=chunksize
            )
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Could not read CSV in chunks {path} ({last_exc})")


def _swell_missing_by_subject(
    swell_dir: Path, max_rows: int | None, chunk_size: int
) -> pd.DataFrame:
    file_path = swell_dir / SWELL_PHYSIO_FILE
    if not file_path.exists():
        raise FileNotFoundError(f"SWELL physiology file not found: {file_path}")

    subject_counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}

    processed = 0
    for chunk in _iter_csv_chunks(file_path, chunk_size):
        chunk.columns = chunk.columns.str.strip().str.replace(" ", "_").str.lower()
        subject_col = "participant" if "participant" in chunk.columns else "subject"
        if subject_col not in chunk.columns:
            raise ValueError("SWELL physiology file missing subject column")

        if max_rows is not None and processed >= max_rows:
            break

        if max_rows is not None and processed + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - processed]

        processed += len(chunk)

        meta_cols = {subject_col, "condition", "minute"}
        feature_cols = [c for c in chunk.columns if c not in meta_cols]
        features = _coerce_numeric_dataframe(chunk[feature_cols])

        missing_per_row = features.isna().sum(axis=1).to_numpy()
        total_per_row = np.full(len(features), len(feature_cols), dtype=np.int64)
        subjects = chunk[subject_col].astype(str).to_numpy()

        for subj, miss, total in zip(subjects, missing_per_row, total_per_row):
            subject_counts[subj] = subject_counts.get(subj, 0) + 1
            missing_counts[subj] = missing_counts.get(subj, 0) + int(miss)
            total_counts[subj] = total_counts.get(subj, 0) + int(total)

    rows = []
    for subj in sorted(subject_counts):
        total = total_counts.get(subj, 0)
        miss = missing_counts.get(subj, 0)
        ratio = (miss / total) if total else 0.0
        rows.append(
            {
                "subject": subj,
                "rows": subject_counts[subj],
                "missing_values": miss,
                "total_values": total,
                "missing_ratio": ratio,
            }
        )
    return pd.DataFrame(rows)


def _iter_empatica_csv(path: Path, chunksize: int, max_rows: int | None):
    processed = 0
    for chunk in pd.read_csv(path, header=None, skiprows=2, chunksize=chunksize):
        if max_rows is not None and processed >= max_rows:
            break
        if max_rows is not None and processed + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - processed]
        processed += len(chunk)
        yield chunk


def _wesad_missing_by_subject(
    wesad_dir: Path,
    signals: list[str],
    max_rows: int | None,
    chunk_size: int,
) -> pd.DataFrame:
    rows = []
    for subject_dir in sorted(wesad_dir.glob("S*")):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name
        e4_dir = subject_dir / f"{subject}_E4_Data"
        if not e4_dir.exists():
            continue

        subject_total = 0
        subject_missing = 0
        per_signal: dict[str, float] = {}

        for signal in signals:
            file_path = e4_dir / f"{signal}.csv"
            if not file_path.exists():
                continue
            total = 0
            missing = 0
            for chunk in _iter_empatica_csv(file_path, chunk_size, max_rows):
                data = _coerce_numeric_dataframe(chunk)
                missing += int(data.isna().sum().sum())
                total += int(data.shape[0] * data.shape[1])
            ratio = (missing / total) if total else 0.0
            per_signal[f"missing_ratio_{signal.lower()}"] = ratio
            subject_total += total
            subject_missing += missing

        subject_ratio = (subject_missing / subject_total) if subject_total else 0.0
        row = {
            "subject": subject,
            "missing_values": subject_missing,
            "total_values": subject_total,
            "missing_ratio": subject_ratio,
        }
        row.update(per_signal)
        rows.append(row)

    return pd.DataFrame(rows)


def _plot_tables(
    wesad_df: pd.DataFrame, swell_df: pd.DataFrame, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    def _make_table(ax, df: pd.DataFrame, title: str):
        ax.axis("off")
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            return
        show = df.copy()
        show["missing_ratio"] = show["missing_ratio"].map(lambda x: f"{x:.4f}")
        cols = (
            ["subject", "rows", "missing_ratio"]
            if "rows" in show.columns
            else [
                "subject",
                "missing_ratio",
            ]
        )
        cell_text = show[cols].values.tolist()
        table = ax.table(
            cellText=cell_text,
            colLabels=cols,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
        ax.set_title(title)

    _make_table(axes[0], wesad_df, "WESAD missing by subject")
    _make_table(axes[1], swell_df, "SWELL physiology missing by subject")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Missing values per client for WESAD and SWELL physiology"
    )
    parser.add_argument("--wesad-dir", default="data/WESAD")
    parser.add_argument("--swell-dir", default="data/SWELL")
    parser.add_argument("--output-dir", default="comparativa_completa")
    parser.add_argument(
        "--signals",
        default="ACC,BVP,EDA,TEMP,HR",
        help="Comma-separated WESAD signals in E4 folder",
    )
    parser.add_argument(
        "--max-rows-wesad",
        type=int,
        default=200000,
        help="Max rows per WESAD signal file (sample)",
    )
    parser.add_argument(
        "--max-rows-swell",
        type=int,
        default=200000,
        help="Max rows for SWELL physiology file (sample)",
    )
    parser.add_argument("--chunk-size", type=int, default=200000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals = [s.strip().upper() for s in args.signals.split(",") if s.strip()]

    swell_df = _swell_missing_by_subject(
        Path(args.swell_dir),
        max_rows=args.max_rows_swell,
        chunk_size=args.chunk_size,
    )
    wesad_df = _wesad_missing_by_subject(
        Path(args.wesad_dir),
        signals=signals,
        max_rows=args.max_rows_wesad,
        chunk_size=args.chunk_size,
    )

    swell_df.to_csv(out_dir / "swell_missing_by_subject.csv", index=False)
    wesad_df.to_csv(out_dir / "wesad_missing_by_subject.csv", index=False)
    _plot_tables(wesad_df, swell_df, out_dir / "missing_by_subject_tables.png")

    print("Wrote:")
    print(out_dir / "swell_missing_by_subject.csv")
    print(out_dir / "wesad_missing_by_subject.csv")
    print(out_dir / "missing_by_subject_tables.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
