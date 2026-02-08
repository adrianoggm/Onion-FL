#!/usr/bin/env python3
"""Comparative analysis and visualization for WESAD vs SWELL datasets.

Generates technical metrics and comparative plots:
- Sample/feature/subject counts
- Class balance
- Missing values (raw SWELL + processed WESAD)
- Feature overlap
- Correlation heatmaps and distributions
- Feature-label correlations
- Samples per subject

Outputs PNG plots plus JSON/MD summaries into the output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from flower_basic.datasets import wesad as wesad_module  # noqa: E402
from flower_basic.datasets.swell import load_swell_dataset  # noqa: E402


CANONICAL_SCL_MAP = {
    "scl_mean": "eda_mean",
    "scl_std": "eda_std",
    "scl_min": "eda_min",
    "scl_max": "eda_max",
    "scl_median": "eda_median",
}

WESAD_LABEL_MAPPING = {
    0: "baseline/no_stress",
    1: "stress",
}

SWELL_LABEL_MAPPING = {
    "no stress": 0,
    "control": 0,
    "neutral": 0,
    "baseline": 0,
    "0": 0,
    "n": 0,
    "time pressure": 1,
    "interruption": 1,
    "interruptions": 1,
    "combined": 1,
    "stress": 1,
    "1": 1,
    "2": 1,
    "3": 1,
    "t": 1,
    "i": 1,
    "r": 1,
}

WESAD_LABEL_TABLE = [
    ("baseline", 0),
    ("stress", 1),
]

SWELL_LABEL_TABLE = [
    ("no stress", 0),
    ("time pressure", 1),
    ("interruption", 1),
]

MODALITY_PATTERNS = {
    "computer": [
        "A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv",
        "computer_features.csv",
        "*computer*.csv",
    ],
    "facial": [
        "B - Facial expressions features (FaceReaderAllData_final (NaN is 999))-sheet_1.csv",
        "facial_features.csv",
        "*facial*.csv",
        "*face*.csv",
    ],
    "posture": [
        "C - Body posture features (Kinect C (position - per minute))- Sheet_1.csv",
        "C - Body posture features (Kinect - final (annotated and selected))-sheet_1.csv",
        "posture_features.csv",
        "*posture*.csv",
        "*kinect*.csv",
    ],
    "physiology": [
        "D - Physiology features (HR_HRV_SCL - final).csv",
        "physiology_features.csv",
        "*physiology*.csv",
        "*hr*.csv",
    ],
}


def _read_csv_best_effort(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "latin-1", "cp1252"]
    last_exc: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            if df.shape[1] == 1:
                header = path.read_text(encoding=enc, errors="ignore").splitlines()[:1]
                header_line = header[0] if header else ""
                if ";" in header_line:
                    df = pd.read_csv(path, encoding=enc, sep=";")
                else:
                    df = pd.read_csv(path, encoding=enc, sep=",")
            return df
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Could not read CSV {path} ({last_exc})")


def _detect_csv_params(path: Path) -> tuple[str, str | None]:
    encodings = ["utf-8", "latin-1", "cp1252"]
    last_exc: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python", nrows=50)
            if df.shape[1] == 1:
                header = path.read_text(encoding=enc, errors="ignore").splitlines()[:1]
                header_line = header[0] if header else ""
                sep = ";" if ";" in header_line else ","
                _ = pd.read_csv(path, encoding=enc, sep=sep, nrows=50)
                return enc, sep
            return enc, None
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Could not detect CSV params for {path} ({last_exc})")


def _normalize_subject_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    sl = s.str.lower()
    for prefix in [
        "p",
        "participant",
        "subject",
        "id",
        "pp",
        "participantno",
        "participant_id",
    ]:
        mask = sl.str.startswith(prefix)
        s.loc[mask] = s.loc[mask].str.replace(prefix, "", case=False, regex=False)
        sl = s.str.lower()
    s = s.str.replace(r"[^0-9]", "", regex=True).str.lstrip("0")
    s = s.where(s != "", other=series.astype(str).str.strip())
    return s


def _map_swell_conditions(series: pd.Series) -> np.ndarray:
    conditions = (
        series.astype(str).str.strip().str.lower().str.replace("_", " ")
    )
    return np.array([SWELL_LABEL_MAPPING.get(cond, 1) for cond in conditions], dtype=np.int64)


def _aggregate_sample_by_keys(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if not keys:
        return df
    key_cols = [k for k in keys if k in df.columns]
    if not key_cols:
        return df
    df_keys = df[key_cols].astype(str)
    df_num = _coerce_numeric_dataframe(df.drop(columns=key_cols, errors="ignore"))
    agg = pd.concat([df_keys, df_num], axis=1).groupby(key_cols, dropna=False)
    return agg.mean(numeric_only=True).reset_index()


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate column names, keeping the first occurrence."""
    if df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Merge two frames, dropping overlapping non-key columns from the right."""
    overlap = set(left.columns) & set(right.columns)
    overlap = overlap - set(keys)
    if overlap:
        right = right.drop(columns=sorted(overlap), errors="ignore")
    return pd.merge(left, right, on=keys, how="inner")


def _coerce_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        if pd.api.types.is_numeric_dtype(result[col]):
            continue
        series = result[col]
        if series.dtype == object:
            a = pd.to_numeric(series, errors="coerce")
            a_nonnull = a.notna().sum()

            b_series = (
                series.astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            b = pd.to_numeric(b_series, errors="coerce")
            b_nonnull = b.notna().sum()

            c_series = series.astype(str).str.replace(",", "", regex=False)
            c = pd.to_numeric(c_series, errors="coerce")
            c_nonnull = c.notna().sum()

            best = max(
                [(a_nonnull, a), (b_nonnull, b), (c_nonnull, c)], key=lambda x: x[0]
            )
            result[col] = best[1]
        else:
            result[col] = pd.to_numeric(series, errors="coerce")
    return result


def _detect_swell_feature_dir(data_dir: Path) -> Path:
    candidates = [
        data_dir / "3 - Feature dataset" / "per sensor",
        data_dir / "per sensor",
        data_dir,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"SWELL feature dir not found in: {candidates}")


def _load_swell_raw_missingness(
    data_dir: Path, modalities: List[str]
) -> Tuple[Dict[str, dict], dict]:
    feature_dir = _detect_swell_feature_dir(data_dir)
    per_modality: Dict[str, dict] = {}
    merged_df: pd.DataFrame | None = None

    for modality in modalities:
        file_path = None
        for pattern in MODALITY_PATTERNS[modality]:
            if "*" in pattern:
                matches = list(feature_dir.glob(pattern))
                if matches:
                    file_path = matches[0]
                    break
            else:
                candidate = feature_dir / pattern
                if candidate.exists():
                    file_path = candidate
                    break
        if file_path is None:
            raise FileNotFoundError(
                f"SWELL feature file not found for modality '{modality}' in {feature_dir}"
            )

        df = _read_csv_best_effort(Path(file_path))
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        if "participant" not in df.columns:
            for subj_col in ["pp", "subject", "participantno", "participant_id", "id"]:
                if subj_col in df.columns:
                    df = df.rename(columns={subj_col: "participant"})
                    break

        if modality == "facial":
            df = df.replace(999, np.nan)
        if modality == "physiology":
            df = df.rename(
                columns={col: CANONICAL_SCL_MAP.get(col, col) for col in df.columns}
            )

        missing = int(df.isnull().sum().sum())
        total = int(df.shape[0] * df.shape[1]) if df.size else 0
        per_modality[modality] = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "missing": missing,
            "missing_ratio": float(missing / total) if total else 0.0,
        }

        if merged_df is None:
            merged_df = df
        else:
            merge_cols = [
                key
                for key in ["participant", "condition", "blok", "block", "trial", "c"]
                if key in df.columns and key in merged_df.columns
            ]
            if not merge_cols:
                continue
            for key in merge_cols:
                merged_df[key] = merged_df[key].astype(str)
                df[key] = df[key].astype(str)
            merged_df = pd.merge(merged_df, df, on=merge_cols, how="inner")

    if merged_df is None:
        raise RuntimeError("No SWELL modalities could be loaded for missingness check")

    subject_col = "participant" if "participant" in merged_df.columns else "subject"
    meta_candidate_cols = {
        subject_col,
        "condition",
        "timestamp",
        "time",
        "minute",
        "blok",
        "block",
        "trial",
        "c",
    }
    feature_columns = [
        col for col in merged_df.columns if col not in meta_candidate_cols
    ]
    features_df = _coerce_numeric_dataframe(merged_df[feature_columns])
    missing = int(features_df.isnull().sum().sum())
    total = int(features_df.shape[0] * features_df.shape[1]) if features_df.size else 0
    merged_missing = {
        "rows": int(features_df.shape[0]),
        "cols": int(features_df.shape[1]),
        "missing": missing,
        "missing_ratio": float(missing / total) if total else 0.0,
    }
    return per_modality, merged_missing


def _load_wesad_light(
    data_dir: Path,
    signals: List[str] | None,
    sensor_location: str,
    conditions: List[str] | None,
    window_size: int,
    overlap: float,
    max_rows: int,
    seed: int,
) -> dict:
    subjects = wesad_module.WESAD_SUBJECTS.copy()
    if signals is None:
        signals = ["BVP", "EDA", "ACC", "TEMP"]
    if conditions is None:
        conditions = ["baseline", "stress"]

    rng = np.random.default_rng(seed)
    per_subject_cap = max(1, max_rows // max(1, len(subjects)))

    sample_X: List[np.ndarray] = []
    sample_y: List[np.ndarray] = []
    subject_counts: Dict[str, int] = {}
    class_counts = {0: 0, 1: 0}
    feature_names: List[str] | None = None
    total_rows = 0
    missing = 0
    total_elements = 0

    for subject_id in subjects:
        try:
            X_sub, y_sub, subject_feature_names = wesad_module._load_subject_data(
                data_dir=data_dir,
                subject_id=subject_id,
                signals=signals,
                sensor_location=sensor_location,
                conditions=conditions,
                window_size=window_size,
                overlap=overlap,
            )
        except Exception as exc:
            print(f"[WESAD] Skip {subject_id}: {exc}")
            continue

        if feature_names is None:
            feature_names = subject_feature_names

        total_rows += int(X_sub.shape[0])
        subject_counts[subject_id] = int(X_sub.shape[0])
        class_counts[0] += int(np.sum(y_sub == 0))
        class_counts[1] += int(np.sum(y_sub == 1))
        missing += int(np.isnan(X_sub).sum())
        total_elements += int(X_sub.size)

        take = min(per_subject_cap, X_sub.shape[0])
        if take > 0:
            idx = rng.choice(X_sub.shape[0], size=take, replace=False)
            sample_X.append(X_sub[idx])
            sample_y.append(y_sub[idx])

    if not sample_X or feature_names is None:
        raise RuntimeError("WESAD light load produced no samples")

    X_sample = np.vstack(sample_X)
    y_sample = np.concatenate(sample_y)
    if X_sample.shape[0] > max_rows:
        idx = rng.choice(X_sample.shape[0], size=max_rows, replace=False)
        X_sample = X_sample[idx]
        y_sample = y_sample[idx]

    return {
        "X_sample": X_sample,
        "y_sample": y_sample,
        "feature_names": feature_names,
        "subject_counts": subject_counts,
        "class_counts": class_counts,
        "n_samples": total_rows,
        "missing": missing,
        "total_elements": total_elements,
        "signals": signals,
        "sensor_location": sensor_location,
        "conditions": conditions,
        "window_size": window_size,
        "overlap": overlap,
    }


def _load_swell_light(
    data_dir: Path,
    modalities: List[str],
    max_rows: int,
    seed: int,
    chunk_size: int,
    scan_full: bool,
) -> dict:
    feature_dir = _detect_swell_feature_dir(data_dir)
    rng = np.random.default_rng(seed)

    per_modality_missing: Dict[str, dict] = {}
    feature_info: Dict[str, dict] = {}
    rows_by_modality: Dict[str, int] = {}
    subjects_union: set[str] = set()
    subject_counts: Dict[str, int] = {}
    class_counts_by_modality: Dict[str, dict] = {}

    sample_frames: Dict[str, pd.DataFrame] = {}

    for modality in modalities:
        file_path = None
        for pattern in MODALITY_PATTERNS[modality]:
            if "*" in pattern:
                matches = list(feature_dir.glob(pattern))
                if matches:
                    file_path = matches[0]
                    break
            else:
                candidate = feature_dir / pattern
                if candidate.exists():
                    file_path = candidate
                    break
        if file_path is None:
            raise FileNotFoundError(
                f"SWELL feature file not found for modality '{modality}' in {feature_dir}"
            )

        enc, sep = _detect_csv_params(Path(file_path))
        sample_df = pd.read_csv(
            file_path,
            encoding=enc,
            sep=sep,
            engine="python",
            nrows=max_rows,
        )
        sample_df.columns = (
            sample_df.columns.str.strip().str.replace(" ", "_").str.lower()
        )
        if "participant" not in sample_df.columns:
            for subj_col in [
                "pp",
                "subject",
                "participantno",
                "participant_id",
                "id",
            ]:
                if subj_col in sample_df.columns:
                    sample_df = sample_df.rename(columns={subj_col: "participant"})
                    break
        if modality == "facial":
            sample_df = sample_df.replace(999, np.nan)
        if modality == "physiology":
            sample_df = sample_df.rename(
                columns={col: CANONICAL_SCL_MAP.get(col, col) for col in sample_df.columns}
            )
        sample_df = _dedupe_columns(sample_df)
        sample_keys = ["participant", "condition", "blok", "block", "trial", "c"]
        sample_df = _aggregate_sample_by_keys(sample_df, sample_keys)
        sample_frames[modality] = _dedupe_columns(sample_df)

        if scan_full:
            chunks = pd.read_csv(
                file_path,
                encoding=enc,
                sep=sep,
                engine="python",
                chunksize=chunk_size,
            )

            total_rows = 0
            total_cols = None
            missing_total = 0
            total_cells = 0
            class_counts = {0: 0, 1: 0}
            class_rows = 0

            for chunk in chunks:
                chunk.columns = chunk.columns.str.strip().str.replace(" ", "_").str.lower()
                if "participant" not in chunk.columns:
                    for subj_col in [
                        "pp",
                        "subject",
                        "participantno",
                        "participant_id",
                        "id",
                    ]:
                        if subj_col in chunk.columns:
                            chunk = chunk.rename(columns={subj_col: "participant"})
                            break

                if modality == "facial":
                    chunk = chunk.replace(999, np.nan)
                if modality == "physiology":
                    chunk = chunk.rename(
                        columns={col: CANONICAL_SCL_MAP.get(col, col) for col in chunk.columns}
                    )

                if total_cols is None:
                    total_cols = chunk.shape[1]

                total_rows += chunk.shape[0]
                missing_total += int(chunk.isnull().sum().sum())
                total_cells += int(chunk.shape[0] * chunk.shape[1])

                if "participant" in chunk.columns:
                    normalized = _normalize_subject_series(chunk["participant"]).astype(str)
                    subjects_union.update(normalized.tolist())
                    counts = normalized.value_counts()
                    for subj, cnt in counts.items():
                        subject_counts[subj] = subject_counts.get(subj, 0) + int(cnt)

                if "condition" in chunk.columns:
                    labels = _map_swell_conditions(chunk["condition"])
                    class_counts[0] += int(np.sum(labels == 0))
                    class_counts[1] += int(np.sum(labels == 1))
                    class_rows += int(labels.size)

            rows_by_modality[modality] = total_rows
            per_modality_missing[modality] = {
                "rows": int(total_rows),
                "cols": int(total_cols or 0),
                "missing": int(missing_total),
                "missing_ratio": float(missing_total / total_cells) if total_cells else 0.0,
            }

            if class_rows > 0:
                class_counts_by_modality[modality] = {
                    "rows": class_rows,
                    "counts": class_counts,
                }
        else:
            # Sample-based stats only
            total_rows = int(sample_df.shape[0])
            total_cols = int(sample_df.shape[1])
            missing_total = int(sample_df.isnull().sum().sum())
            total_cells = int(sample_df.shape[0] * sample_df.shape[1])

            rows_by_modality[modality] = total_rows
            per_modality_missing[modality] = {
                "rows": total_rows,
                "cols": total_cols,
                "missing": missing_total,
                "missing_ratio": float(missing_total / total_cells) if total_cells else 0.0,
            }

            if "participant" in sample_df.columns:
                normalized = _normalize_subject_series(sample_df["participant"]).astype(str)
                subjects_union.update(normalized.tolist())
                counts = normalized.value_counts()
                for subj, cnt in counts.items():
                    subject_counts[subj] = subject_counts.get(subj, 0) + int(cnt)

            if "condition" in sample_df.columns:
                labels = _map_swell_conditions(sample_df["condition"])
                class_counts_by_modality[modality] = {
                    "rows": int(labels.size),
                    "counts": {
                        0: int(np.sum(labels == 0)),
                        1: int(np.sum(labels == 1)),
                    },
                }

        subject_col = "participant" if "participant" in sample_df.columns else "subject"
        meta_cols = {subject_col, "condition"}
        feature_cols = [c for c in sample_df.columns if c not in meta_cols]
        feature_info[modality] = {
            "n_features": len(feature_cols),
            "missing_ratio": per_modality_missing[modality]["missing_ratio"],
            "feature_names": feature_cols,
        }

    class_counts = {0: 0, 1: 0}
    if class_counts_by_modality:
        best_modality = max(
            class_counts_by_modality.items(), key=lambda x: x[1]["rows"]
        )[0]
        class_counts = class_counts_by_modality[best_modality]["counts"]

    # Merge sample frames
    merged_sample: pd.DataFrame | None = None
    for modality in modalities:
        df = sample_frames.get(modality)
        if df is None:
            continue
        if merged_sample is None:
            merged_sample = df
            continue
        merge_cols = [
            key
            for key in ["participant", "condition", "blok", "block", "trial", "c"]
            if key in df.columns and key in merged_sample.columns
        ]
        if not merge_cols:
            continue
        for key in merge_cols:
            merged_sample[key] = merged_sample[key].astype(str)
            df[key] = df[key].astype(str)
        merged_sample = _safe_merge(merged_sample, df, merge_cols)
        if merged_sample.shape[0] > max_rows:
            merged_sample = merged_sample.sample(n=max_rows, random_state=seed)

    if merged_sample is None:
        raise RuntimeError("Failed to build SWELL sample for correlations")

    subject_col = "participant" if "participant" in merged_sample.columns else "subject"
    meta_candidate_cols = {
        subject_col,
        "condition",
        "timestamp",
        "time",
        "minute",
        "blok",
        "block",
        "trial",
        "c",
    }
    feature_columns = [
        col for col in merged_sample.columns if col not in meta_candidate_cols
    ]
    features_df = _coerce_numeric_dataframe(merged_sample[feature_columns])

    missing = int(features_df.isnull().sum().sum())
    total = int(features_df.shape[0] * features_df.shape[1]) if features_df.size else 0
    merged_missing = {
        "rows": int(features_df.shape[0]),
        "cols": int(features_df.shape[1]),
        "missing": missing,
        "missing_ratio": float(missing / total) if total else 0.0,
    }

    if features_df.isnull().to_numpy().any():
        features_df = features_df.fillna(features_df.mean())
    features_df = features_df.fillna(0.0)

    X_sample = features_df.to_numpy(dtype=np.float32, copy=False)
    y_sample = (
        _map_swell_conditions(merged_sample["condition"])
        if "condition" in merged_sample.columns
        else np.zeros((X_sample.shape[0],), dtype=np.int64)
    )

    feature_variances = np.var(X_sample, axis=0) if X_sample.size else np.array([])
    if feature_variances.size:
        valid_features = feature_variances > 1e-8
        if not np.all(valid_features):
            X_sample = X_sample[:, valid_features]
            feature_columns = [
                col for idx, col in enumerate(feature_columns) if valid_features[idx]
            ]

    n_samples_est = min(rows_by_modality.values()) if rows_by_modality else X_sample.shape[0]

    total_missing = sum(v["missing"] for v in per_modality_missing.values())
    total_cells = sum(v["rows"] * v["cols"] for v in per_modality_missing.values())
    missing_ratio_raw = float(total_missing / total_cells) if total_cells else 0.0

    return {
        "X_sample": X_sample,
        "y_sample": y_sample,
        "feature_names": feature_columns,
        "n_samples": int(n_samples_est),
        "rows_by_modality": rows_by_modality,
        "subjects": sorted(subjects_union),
        "subject_counts": subject_counts,
        "class_counts": class_counts,
        "missing_raw": total_missing,
        "missing_ratio_raw": missing_ratio_raw,
        "missing_by_modality": per_modality_missing,
        "merged_missing_sample": merged_missing,
        "feature_info": feature_info,
        "scan_full": scan_full,
    }


def _count_swell_subjects(data_dir: Path) -> List[str]:
    rri_dir = data_dir / "data" / "raw" / "rri"
    if rri_dir.exists():
        subjects = []
        for p in sorted(rri_dir.glob("p*.txt")):
            name = p.stem.lower().lstrip("p")
            if name:
                subjects.append(name)
        if subjects:
            return subjects
    return []


def _load_swell_physiology_final(
    data_dir: Path, max_rows: int, seed: int
) -> dict:
    final_dir = data_dir / "data" / "final"
    train_path = final_dir / "train.csv"
    test_path = final_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "SWELL final physiology files not found at data/SWELL/data/final"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    condition_col = "condition" if "condition" in full_df.columns else "Condition"
    if condition_col not in full_df.columns:
        raise ValueError("SWELL final dataset missing condition column")

    meta_cols = {condition_col, "datasetId"}
    feature_cols = [c for c in full_df.columns if c not in meta_cols]
    features_df = _coerce_numeric_dataframe(full_df[feature_cols])

    missing = int(features_df.isnull().sum().sum())
    total = int(features_df.shape[0] * features_df.shape[1]) if features_df.size else 0
    missing_ratio = float(missing / total) if total else 0.0

    y_full = _map_swell_conditions(full_df[condition_col])
    class_counts = {0: int(np.sum(y_full == 0)), 1: int(np.sum(y_full == 1))}

    # Sample for correlations
    if full_df.shape[0] > max_rows:
        sampled = full_df.sample(n=max_rows, random_state=seed)
    else:
        sampled = full_df

    X_sample = _coerce_numeric_dataframe(sampled[feature_cols]).fillna(
        features_df.mean()
    )
    X_sample = X_sample.fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    y_sample = _map_swell_conditions(sampled[condition_col])

    subjects = _count_swell_subjects(data_dir)
    subject_counts = {s: 0 for s in subjects}

    return {
        "X_sample": X_sample,
        "y_sample": y_sample,
        "feature_names": feature_cols,
        "n_samples": int(full_df.shape[0]),
        "class_counts": class_counts,
        "missing": missing,
        "missing_ratio": missing_ratio,
        "subjects": subjects,
        "subject_counts": subject_counts,
        "mode": "physio_final",
    }


def _maybe_sample(
    X: np.ndarray,
    y: np.ndarray | None = None,
    subject_ids: np.ndarray | None = None,
    max_rows: int = 10000,
    seed: int = 42,
):
    if X.shape[0] <= max_rows:
        return X, y, subject_ids
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_rows, replace=False)
    return X[idx], (y[idx] if y is not None else None), (
        subject_ids[idx] if subject_ids is not None else None
    )


def _select_features_by_variance(
    X: np.ndarray, feature_names: List[str], max_features: int
) -> Tuple[np.ndarray, List[str]]:
    if X.size == 0 or X.shape[0] == 0 or X.shape[1] == 0:
        return np.array([], dtype=int), []
    if X.shape[1] <= max_features:
        return np.arange(X.shape[1]), feature_names
    variances = np.var(X, axis=0)
    idx = np.argsort(variances)[::-1][:max_features]
    selected = [feature_names[i] for i in idx]
    return idx, selected


def _safe_corrcoef(X: np.ndarray) -> np.ndarray:
    if X.size == 0 or X.shape[0] < 2 or X.shape[1] < 2:
        return np.zeros((0, 0), dtype=np.float32)
    return np.corrcoef(X, rowvar=False)


def _corr_summary(corr: np.ndarray) -> Dict[str, float]:
    if corr.size == 0 or corr.shape[0] < 2:
        return {"mean_abs": 0.0, "median_abs": 0.0}
    tri = corr[np.triu_indices_from(corr, k=1)]
    if tri.size == 0:
        return {"mean_abs": 0.0, "median_abs": 0.0}
    abs_vals = np.abs(tri)
    return {
        "mean_abs": float(np.mean(abs_vals)),
        "median_abs": float(np.median(abs_vals)),
    }


def _feature_label_corrs(
    X: np.ndarray, y: np.ndarray, feature_names: List[str]
) -> pd.DataFrame:
    if X.size == 0 or X.shape[1] == 0:
        return pd.DataFrame({"feature": [], "corr": []})
    y = y.astype(float)
    y_std = float(np.std(y))
    if y_std == 0.0:
        return pd.DataFrame({"feature": feature_names, "corr": 0.0})
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()
    denom = X_centered.std(axis=0) * y_std
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = (X_centered * y_centered[:, None]).mean(axis=0) / denom
    corrs = np.nan_to_num(corrs, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame({"feature": feature_names, "corr": corrs})


def _group_wesad_features(feature_names: List[str]) -> Dict[str, int]:
    groups: Dict[str, int] = {}
    for name in feature_names:
        if name.startswith("acc_"):
            key = "acc"
        else:
            key = name.split("_", 1)[0]
        groups[key] = groups.get(key, 0) + 1
    return groups


def _wesad_metadata_only(
    signals: List[str],
    sensor_location: str,
    stats: List[str] | None = None,
) -> dict:
    if stats is None:
        stats = ["mean", "std", "min", "max", "median"]
    channels: List[str] = []
    for signal in signals:
        key = signal.upper()
        if key == "ACC":
            channels.extend([f"{signal.lower()}_{i}" for i in range(3)])
        else:
            channels.append(signal.lower())
    feature_names = [f"{ch}_{stat}" for ch in channels for stat in stats]
    return {
        "n_subjects": len(wesad_module.WESAD_SUBJECTS),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "signals": signals,
        "sensor_location": sensor_location,
        "class_counts": None,
        "subject_counts": {},
        "n_samples": None,
        "missing": None,
        "missing_ratio": None,
    }


def _plot_overview(metrics: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    datasets = ["WESAD", "SWELL"]

    values = [metrics["wesad"]["n_subjects"], metrics["swell"]["n_subjects"]]
    labels = []
    for v in values:
        if v is None:
            labels.append("N/A")
        else:
            labels.append(str(v))
    values = [0 if v is None else v for v in values]
    axes[0, 0].bar(datasets, values, color=["#2c7fb8", "#7fcdbb"])
    axes[0, 0].set_title("Numero de sujetos")
    for i, v in enumerate(values):
        axes[0, 0].text(i, v, labels[i], ha="center", va="bottom")

    values = [metrics["wesad"]["n_samples"], metrics["swell"]["n_samples"]]
    labels = []
    for v in values:
        if v is None:
            labels.append("N/A")
        else:
            labels.append(str(v))
    values = [0 if v is None else v for v in values]
    axes[0, 1].bar(datasets, values, color=["#f03b20", "#feb24c"])
    axes[0, 1].set_title("Numero de muestras")
    for i, v in enumerate(values):
        axes[0, 1].text(i, v, labels[i], ha="center", va="bottom")

    values = [metrics["wesad"]["n_features"], metrics["swell"]["n_features"]]
    labels = []
    for v in values:
        if v is None:
            labels.append("N/A")
        else:
            labels.append(str(v))
    values = [0 if v is None else v for v in values]
    axes[1, 0].bar(datasets, values, color=["#756bb1", "#9e9ac8"])
    axes[1, 0].set_title("Numero de features")
    for i, v in enumerate(values):
        axes[1, 0].text(i, v, labels[i], ha="center", va="bottom")

    values = [metrics["wesad"]["missing_ratio"], metrics["swell"]["missing_ratio"]]
    labels = []
    for v in values:
        if v is None:
            labels.append("N/A")
        else:
            labels.append(f"{v:.4f}")
    values = [0 if v is None else v for v in values]
    axes[1, 1].bar(datasets, values, color=["#31a354", "#a1d99b"])
    axes[1, 1].set_title("Ratio de nulos")
    for i, v in enumerate(values):
        axes[1, 1].text(i, v, labels[i], ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_dir / "wesad_swell_overview.png", dpi=200)
    plt.close(fig)


def _plot_class_balance(metrics: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = ["WESAD", "SWELL"]
    stress = [
        metrics["wesad"]["class_distribution"].get("1", 0),
        metrics["swell"]["class_distribution"].get("1", 0),
    ]
    nostress = [
        metrics["wesad"]["class_distribution"].get("0", 0),
        metrics["swell"]["class_distribution"].get("0", 0),
    ]
    totals = [stress[i] + nostress[i] for i in range(2)]
    stress_pct = [
        (stress[i] / totals[i] * 100) if totals[i] else 0.0 for i in range(2)
    ]
    nostress_pct = [
        (nostress[i] / totals[i] * 100) if totals[i] else 0.0 for i in range(2)
    ]

    ax.bar(datasets, nostress_pct, label="No estres", color="#9ecae1")
    ax.bar(datasets, stress_pct, bottom=nostress_pct, label="Estres", color="#fb6a4a")
    ax.set_ylabel("Porcentaje (%)")
    ax.set_title("Balance de clases")
    ax.legend()

    for i in range(2):
        if totals[i] == 0:
            ax.text(i, 5, "N/A", ha="center")
            continue
        ax.text(i, nostress_pct[i] / 2, f"{nostress_pct[i]:.1f}%", ha="center")
        ax.text(
            i,
            nostress_pct[i] + stress_pct[i] / 2,
            f"{stress_pct[i]:.1f}%",
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(out_dir / "wesad_swell_class_balance.png", dpi=200)
    plt.close(fig)


def _plot_samples_per_subject(
    wesad_subject_counts: pd.Series,
    swell_subject_counts: pd.Series,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    w_vals = wesad_subject_counts.values if len(wesad_subject_counts) else np.array([])
    s_vals = swell_subject_counts.values if len(swell_subject_counts) else np.array([])
    data = []
    labels = []
    if w_vals.size:
        data.append(w_vals)
        labels.append("WESAD")
    if s_vals.size:
        data.append(s_vals)
        labels.append("SWELL")

    if not data:
        ax.text(0.5, 0.5, "Insuficiente", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title("Muestras por sujeto (boxplot)")
        ax.set_ylabel("Numero de muestras")
    fig.tight_layout()
    fig.savefig(out_dir / "wesad_swell_samples_per_subject.png", dpi=200)
    plt.close(fig)


def _plot_missingness_modalities(
    per_modality: Dict[str, dict], out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not per_modality:
        ax.text(0.5, 0.5, "Insuficiente", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_dir / "swell_missing_ratio_by_modality.png", dpi=200)
        plt.close(fig)
        return
    modalities = list(per_modality.keys())
    ratios = [per_modality[m].get("missing_ratio", 0.0) for m in modalities]
    ax.bar(modalities, ratios, color="#fdd0a2")
    ax.set_title("SWELL: ratio de nulos por modalidad (raw)")
    ax.set_ylabel("Ratio de nulos")
    for i, v in enumerate(ratios):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "swell_missing_ratio_by_modality.png", dpi=200)
    plt.close(fig)


def _plot_feature_overlap(
    wesad_features: List[str], swell_features: List[str], out_dir: Path
) -> None:
    if not wesad_features or not swell_features:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "Insuficiente", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_dir / "wesad_swell_feature_overlap.png", dpi=200)
        plt.close(fig)
        return
    set_w = set(wesad_features)
    set_s = set(swell_features)
    shared = set_w & set_s
    wesad_only = len(set_w - shared)
    swell_only = len(set_s - shared)
    shared_count = len(shared)

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["WESAD solo", "Compartidas", "SWELL solo"]
    values = [wesad_only, shared_count, swell_only]
    ax.bar(labels, values, color=["#3182bd", "#9ecae1", "#31a354"])
    ax.set_title("Overlap de features")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "wesad_swell_feature_overlap.png", dpi=200)
    plt.close(fig)


def _plot_feature_groups(
    wesad_groups: Dict[str, int],
    swell_modalities: Dict[str, dict],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    w_labels = list(wesad_groups.keys())
    w_values = [wesad_groups[k] for k in w_labels]
    if w_labels:
        axes[0].bar(w_labels, w_values, color="#74c476")
        axes[0].set_title("WESAD: features por sensor")
        axes[0].set_ylabel("Numero de features")
    else:
        axes[0].text(0.5, 0.5, "Insuficiente", ha="center", va="center")
        axes[0].set_axis_off()

    s_labels = list(swell_modalities.keys())
    s_values = [swell_modalities[k]["n_features"] for k in s_labels] if s_labels else []
    if s_labels:
        axes[1].bar(s_labels, s_values, color="#6baed6")
        axes[1].set_title("SWELL: features por modalidad")
        axes[1].set_ylabel("Numero de features")
    else:
        axes[1].text(0.5, 0.5, "Insuficiente", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(out_dir / "wesad_swell_feature_groups.png", dpi=200)
    plt.close(fig)


def _plot_corr_heatmaps(
    wesad_corr: np.ndarray,
    wesad_names: List[str],
    swell_corr: np.ndarray,
    swell_names: List[str],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    if wesad_corr.size == 0 or wesad_corr.shape[0] < 2:
        axes[0].text(0.5, 0.5, "Insuficiente", ha="center", va="center")
        axes[0].set_axis_off()
    else:
        sns.heatmap(
            wesad_corr,
            ax=axes[0],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
    axes[0].set_title("WESAD: correlacion features")

    if swell_corr.size == 0 or swell_corr.shape[0] < 2:
        axes[1].text(0.5, 0.5, "Insuficiente", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        sns.heatmap(
            swell_corr,
            ax=axes[1],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
    axes[1].set_title("SWELL: correlacion (top varianza)")

    fig.tight_layout()
    fig.savefig(out_dir / "wesad_swell_corr_heatmaps.png", dpi=200)
    plt.close(fig)


def _plot_corr_distributions(
    wesad_corr: np.ndarray, swell_corr: np.ndarray, out_dir: Path
) -> None:
    if wesad_corr.size == 0 or wesad_corr.shape[0] < 2:
        w_vals = np.array([])
    else:
        w_vals = wesad_corr[np.triu_indices_from(wesad_corr, k=1)]
    if swell_corr.size == 0 or swell_corr.shape[0] < 2:
        s_vals = np.array([])
    else:
        s_vals = swell_corr[np.triu_indices_from(swell_corr, k=1)]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(w_vals, bins=40, kde=True, color="#3182bd", label="WESAD", ax=ax)
    sns.histplot(s_vals, bins=40, kde=True, color="#31a354", label="SWELL", ax=ax)
    ax.set_title("Distribucion de correlaciones (pares de features)")
    ax.set_xlabel("Correlacion Pearson")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "wesad_swell_corr_distribution.png", dpi=200)
    plt.close(fig)


def _plot_top_label_corrs(
    wesad_top: pd.DataFrame,
    swell_top: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if wesad_top.empty:
        axes[0].text(0.5, 0.5, "Insuficiente", ha="center", va="center")
        axes[0].set_axis_off()
    else:
        wesad_top = wesad_top.sort_values("abs_corr", ascending=True)
        axes[0].barh(wesad_top["feature"], wesad_top["abs_corr"], color="#9ecae1")
        axes[0].set_title("WESAD: top correlacion feature-label")
        axes[0].set_xlabel("|corr|")

    if swell_top.empty:
        axes[1].text(0.5, 0.5, "Insuficiente", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        swell_top = swell_top.sort_values("abs_corr", ascending=True)
        axes[1].barh(swell_top["feature"], swell_top["abs_corr"], color="#a1d99b")
        axes[1].set_title("SWELL: top correlacion feature-label")
        axes[1].set_xlabel("|corr|")

    fig.tight_layout()
    fig.savefig(out_dir / "wesad_swell_top_label_corrs.png", dpi=200)
    plt.close(fig)


def _plot_label_tables(out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    wesad_rows = [[name, cls] for name, cls in WESAD_LABEL_TABLE]
    swell_rows = [[name, cls] for name, cls in SWELL_LABEL_TABLE]

    for ax, title, rows in [
        (axes[0], "WESAD labels", wesad_rows),
        (axes[1], "SWELL labels (physiology)", swell_rows),
    ]:
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=["label", "class_id"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 1.3)
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_dir / "wesad_swell_label_tables.png", dpi=200)
    plt.close(fig)


def _write_feature_table(
    wesad_features: List[str], swell_features: List[str], out_dir: Path
) -> None:
    max_len = max(len(wesad_features), len(swell_features))
    wesad_padded = wesad_features + [""] * (max_len - len(wesad_features))
    swell_padded = swell_features + [""] * (max_len - len(swell_features))

    df = pd.DataFrame(
        {"WESAD_features": wesad_padded, "SWELL_features": swell_padded}
    )
    df.to_csv(out_dir / "wesad_swell_feature_table.csv", index=False)
    df.to_markdown(out_dir / "wesad_swell_feature_table.md", index=False)

    # Minimal LaTeX table (for memoria)
    lines = [
        "\\begin{table}[H]",
        "  \\centering",
        "  \\caption{Listado de \\emph{features} WESAD vs SWELL (fisiol\\'ogico).}",
        "  \\label{tab:features_wesad_swell}",
        "  \\begin{tabular}{ll}",
        "    \\hline",
        "    \\textbf{WESAD} & \\textbf{SWELL (fisiol\\'ogico)} \\\\",
        "    \\hline",
    ]
    for w, s in zip(wesad_padded, swell_padded):
        lines.append(f"    {w} & {s} \\\\")
    lines.extend(["    \\hline", "  \\end{tabular}", "\\end{table}"])
    (out_dir / "wesad_swell_feature_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # PNG table for memory
    fig, ax = plt.subplots(figsize=(8, max(4, max_len * 0.25)))
    ax.axis("off")
    table = ax.table(
        cellText=list(zip(wesad_padded, swell_padded)),
        colLabels=["WESAD", "SWELL (fisiologico)"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    fig.tight_layout()
    fig.savefig(out_dir / "wesad_swell_feature_table.png", dpi=200)
    plt.close(fig)


def _write_summary(summary: dict, out_dir: Path) -> None:
    with open(out_dir / "wesad_swell_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    wesad_samples = (
        str(summary["wesad"]["n_samples"])
        if summary["wesad"]["n_samples"] is not None
        else "N/A"
    )
    swell_samples = (
        str(summary["swell"]["n_samples"])
        if summary["swell"]["n_samples"] is not None
        else "N/A"
    )
    wesad_missing_ratio = (
        f"{summary['wesad']['missing_ratio']:.6f}"
        if summary["wesad"]["missing_ratio"] is not None
        else "N/A"
    )
    swell_missing_ratio = (
        f"{summary['swell']['missing_ratio']:.6f}"
        if summary["swell"]["missing_ratio"] is not None
        else "N/A"
    )

    md_lines = [
        "# WESAD vs SWELL - Resumen",
        "",
        "| Metrica | WESAD | SWELL |",
        "| --- | --- | --- |",
        f"| Sujetos | {summary['wesad']['n_subjects']} | {summary['swell']['n_subjects']} |",
        f"| Muestras | {wesad_samples} | {swell_samples} |",
        f"| Features | {summary['wesad']['n_features']} | {summary['swell']['n_features']} |",
        f"| Ratio nulos | {wesad_missing_ratio} | {swell_missing_ratio} |",
        "",
        "## Distribucion de clases",
        f"- WESAD: {summary['wesad']['class_distribution']}",
        f"- SWELL: {summary['swell']['class_distribution']}",
        "",
    ]

    md_lines.extend(
        [
            "## Etiquetas",
            f"- WESAD: {summary['wesad'].get('label_mapping', {})}",
            f"- SWELL: {summary['swell'].get('label_mapping', {})}",
            "- Figura: `wesad_swell_label_tables.png`",
            "",
            "## Features y correlacion",
            "- WESAD: ver `wesad_feature_list.txt` y `wesad_feature_correlations.csv`",
            "- SWELL (fisiologico): ver `swell_physiology_feature_list.txt` y `swell_physiology_feature_correlations.csv`",
            "- Tabla compartida: `wesad_swell_feature_table.(csv|md|tex|png)`",
            "",
        ]
    )

    if summary["wesad"].get("mode") == "metadata":
        md_lines.extend(
            [
                "## WESAD: modo metadata",
                "- Se omitio la carga completa por memoria; algunas metricas pueden estar incompletas.",
                "",
            ]
        )

    if summary["swell"].get("mode") == "sample":
        md_lines.extend(
            [
                "## SWELL: modo sample",
                "- Estadisticas de nulos/clase basadas en muestra; use --scan-swell para exactitud.",
                "",
            ]
        )

    if summary["swell"].get("rows_by_modality"):
        md_lines.extend(
            [
                "## SWELL: filas por modalidad (raw)",
                *[
                    f"- {mod}: {rows}"
                    for mod, rows in summary["swell"]["rows_by_modality"].items()
                ],
                "- Nota: n_samples SWELL se estima como min(filas por modalidad).",
                "",
            ]
        )

    if summary["swell"].get("subjects"):
        md_lines.extend(
            [
                "## SWELL: sujetos detectados (raw rri)",
                "- " + ", ".join(summary["swell"]["subjects"]),
                "",
            ]
        )

    md_lines.extend(
        [
        "## Features compartidas",
        f"- Conteo: {summary['shared_features']['count']}",
        ]
    )

    shared_names = summary["shared_features"]["names"]
    if shared_names:
        md_lines.append("- Ejemplos: " + ", ".join(shared_names[:10]))
    else:
        md_lines.append("- Ejemplos: (ninguna)")

    md_lines.append("")
    md_lines.append("## Correlacion media absoluta (pares de features)")
    md_lines.append(
        f"- WESAD: {summary['wesad']['corr_summary']['mean_abs']:.4f} "
        f"(mediana {summary['wesad']['corr_summary']['median_abs']:.4f})"
    )
    md_lines.append(
        f"- SWELL: {summary['swell']['corr_summary']['mean_abs']:.4f} "
        f"(mediana {summary['swell']['corr_summary']['median_abs']:.4f})"
    )
    md_lines.append("")
    md_lines.append("## Top correlaciones feature-label")
    wesad_top = summary["wesad"]["top_label_corrs"][:5]
    swell_top = summary["swell"]["top_label_corrs"][:5]
    md_lines.append(
        "- WESAD: "
        + (
            ", ".join(f"{item['feature']} ({item['corr']:.3f})" for item in wesad_top)
            if wesad_top
            else "(sin datos)"
        )
    )
    md_lines.append(
        "- SWELL: "
        + (
            ", ".join(f"{item['feature']} ({item['corr']:.3f})" for item in swell_top)
            if swell_top
            else "(sin datos)"
        )
    )

    with open(out_dir / "wesad_swell_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comparative analysis of WESAD vs SWELL datasets"
    )
    parser.add_argument(
        "--wesad-dir", default="data/WESAD", help="Path to WESAD data dir"
    )
    parser.add_argument(
        "--swell-dir", default="data/SWELL", help="Path to SWELL data dir"
    )
    parser.add_argument(
        "--output-dir",
        default="comparativa_completa",
        help="Output directory for plots and summary",
    )
    parser.add_argument(
        "--max-corr-features",
        type=int,
        default=40,
        help="Max features for correlation heatmap",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000,
        help="Max rows sampled for correlation and label-corr",
    )
    parser.add_argument(
        "--swell-chunk-size",
        type=int,
        default=5000,
        help="Chunk size for SWELL streaming stats",
    )
    parser.add_argument(
        "--swell-all",
        action="store_true",
        help="Use full SWELL multimodal loader (not only physiology)",
    )
    parser.add_argument(
        "--scan-swell",
        action="store_true",
        help="Scan full SWELL CSVs for exact missingness/class counts (slower/heavier)",
    )
    parser.add_argument(
        "--full-swell",
        action="store_true",
        help="Use full SWELL loader (may require lots of RAM)",
    )
    parser.add_argument(
        "--wesad-metadata",
        action="store_true",
        help="Skip WESAD window extraction (metadata only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Load WESAD
    if not args.wesad_metadata:
        w_light = _load_wesad_light(
            data_dir=Path(args.wesad_dir),
            signals=None,
            sensor_location="wrist",
            conditions=None,
            window_size=60,
            overlap=0.5,
            max_rows=args.max_rows,
            seed=args.seed,
        )
        Xw_s = w_light["X_sample"]
        yw_s = w_light["y_sample"]
        w_feature_names = w_light["feature_names"]
        w_subject_counts = pd.Series(w_light["subject_counts"])
        w_class_counts = w_light["class_counts"]
        w_n_samples = w_light["n_samples"]
        w_missing = int(w_light["missing"])
        w_total = int(w_light["total_elements"])
        w_missing_ratio = float(w_missing / w_total) if w_total else 0.0
        w_signals = w_light.get("signals", [])
        w_sensor = w_light.get("sensor_location", "")
        w_mode = "full"
    else:
        w_meta = _wesad_metadata_only(
            signals=["BVP", "EDA", "ACC", "TEMP"],
            sensor_location="wrist",
        )
        Xw_s = np.zeros((0, w_meta["n_features"]), dtype=np.float32)
        yw_s = np.zeros((0,), dtype=np.int64)
        w_feature_names = w_meta["feature_names"]
        w_subject_counts = pd.Series(dtype=int)
        w_class_counts = {0: 0, 1: 0}
        w_n_samples = None
        w_missing = None
        w_missing_ratio = None
        w_signals = w_meta.get("signals", [])
        w_sensor = w_meta.get("sensor_location", "")
        w_mode = "metadata"

    # Load SWELL (physiology by default)
    if args.swell_all:
        if args.full_swell:
            Xs_train, Xs_test, ys_train, ys_test, s_info = load_swell_dataset(
                data_dir=args.swell_dir,
                normalize_features=False,
                return_subject_info=True,
            )
            Xs = np.vstack([Xs_train, Xs_test])
            ys = np.concatenate([ys_train, ys_test])
            s_subjects = np.array(
                s_info["train_subject_ids"] + s_info["test_subject_ids"]
            )
            s_feature_names = s_info["feature_names"]
            per_modality_missing = s_info.get("feature_info", {})
            merged_missing = {
                "missing": int(np.isnan(Xs).sum()),
                "missing_ratio": float(np.isnan(Xs).sum() / Xs.size)
                if Xs.size
                else 0.0,
                "rows": int(Xs.shape[0]),
                "cols": int(Xs.shape[1]),
            }
            swell_modalities = s_info.get(
                "modalities", ["computer", "facial", "posture", "physiology"]
            )
            s_mode = "full"
            s_feature_info = s_info.get("feature_info", {})
            s_missing_values = int(merged_missing["missing"])
            s_missing_ratio = float(merged_missing["missing_ratio"])
            s_class_counts = {
                0: int(np.sum(ys == 0)),
                1: int(np.sum(ys == 1)),
            }
            s_n_samples = int(Xs.shape[0])
            s_rows_by_modality = {}
            s_subject_counts = pd.Series(s_subjects).value_counts()
        else:
            swell_modalities = ["computer", "facial", "posture", "physiology"]
            s_light = _load_swell_light(
                data_dir=Path(args.swell_dir),
                modalities=swell_modalities,
                max_rows=args.max_rows,
                seed=args.seed,
                chunk_size=args.swell_chunk_size,
                scan_full=args.scan_swell,
            )
            Xs = s_light["X_sample"]
            ys = s_light["y_sample"]
            s_subjects = np.array(s_light["subjects"])
            s_feature_names = s_light["feature_names"]
            per_modality_missing = s_light["missing_by_modality"]
            merged_missing = s_light["merged_missing_sample"]
            s_mode = "scan_full" if s_light["scan_full"] else "sample"
            s_feature_info = s_light["feature_info"]
            s_missing_values = int(s_light["missing_raw"])
            s_missing_ratio = float(s_light["missing_ratio_raw"])
            s_class_counts = s_light["class_counts"]
            s_n_samples = int(s_light["n_samples"])
            s_rows_by_modality = s_light["rows_by_modality"]
            s_subject_counts = pd.Series(s_light["subject_counts"])
    else:
        s_phys = _load_swell_physiology_final(
            data_dir=Path(args.swell_dir),
            max_rows=args.max_rows,
            seed=args.seed,
        )
        Xs = s_phys["X_sample"]
        ys = s_phys["y_sample"]
        s_feature_names = s_phys["feature_names"]
        s_subjects = np.array(s_phys["subjects"])
        s_subject_counts = pd.Series(s_phys["subject_counts"])
        s_class_counts = s_phys["class_counts"]
        s_n_samples = s_phys["n_samples"]
        s_missing_values = int(s_phys["missing"])
        s_missing_ratio = float(s_phys["missing_ratio"])
        swell_modalities = ["physiology"]
        per_modality_missing = {
            "physiology": {
                "missing_ratio": s_missing_ratio,
                "missing": s_missing_values,
                "rows": s_n_samples,
                "cols": len(s_feature_names),
            }
        }
        merged_missing = {
            "missing": s_missing_values,
            "missing_ratio": s_missing_ratio,
            "rows": s_n_samples,
            "cols": len(s_feature_names),
        }
        s_feature_info = {
            "physiology": {
                "n_features": len(s_feature_names),
                "missing_ratio": s_missing_ratio,
                "feature_names": s_feature_names,
            }
        }
        s_mode = s_phys["mode"]
        s_rows_by_modality = {}

    # Samples per subject
    if s_subject_counts is None:
        s_subject_counts = pd.Series(dtype=int)

    # Correlation matrices (subset of rows and features)
    Xs_s, ys_s, _ = _maybe_sample(Xs, ys, None, args.max_rows, args.seed)

    w_idx, w_corr_names = _select_features_by_variance(
        Xw_s, w_feature_names, args.max_corr_features
    )
    s_idx, s_corr_names = _select_features_by_variance(
        Xs_s, s_feature_names, args.max_corr_features
    )

    w_corr = _safe_corrcoef(Xw_s[:, w_idx])
    s_corr = _safe_corrcoef(Xs_s[:, s_idx])

    # Feature-label correlations
    w_fl = _feature_label_corrs(Xw_s, yw_s, w_feature_names)
    w_fl["abs_corr"] = w_fl["corr"].abs()
    w_top = w_fl.sort_values("abs_corr", ascending=False).head(10)

    s_fl = _feature_label_corrs(Xs_s, ys_s, s_feature_names)
    s_fl["abs_corr"] = s_fl["corr"].abs()
    s_top = s_fl.sort_values("abs_corr", ascending=False).head(10)

    if not w_fl.empty:
        w_fl.sort_values("abs_corr", ascending=False).to_csv(
            out_dir / "wesad_feature_correlations.csv", index=False
        )
        with open(out_dir / "wesad_feature_list.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(w_feature_names))

    if not s_fl.empty:
        s_fl.sort_values("abs_corr", ascending=False).to_csv(
            out_dir / "swell_physiology_feature_correlations.csv", index=False
        )
        with open(out_dir / "swell_physiology_feature_list.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(s_feature_names))

    # Summary metrics
    summary = {
        "wesad": {
            "n_subjects": int(len(wesad_module.WESAD_SUBJECTS)),
            "n_samples": int(w_n_samples) if w_n_samples is not None else None,
            "n_features": int(len(w_feature_names)),
            "class_distribution": {
                "0": int(w_class_counts[0]) if w_class_counts else 0,
                "1": int(w_class_counts[1]) if w_class_counts else 0,
            },
            "missing_values": w_missing,
            "missing_ratio": w_missing_ratio,
            "signals": w_signals,
            "sensor_location": w_sensor,
            "mode": w_mode,
            "label_mapping": WESAD_LABEL_MAPPING,
            "corr_summary": _corr_summary(w_corr),
            "top_label_corrs": [
                {"feature": row["feature"], "corr": float(row["corr"])}
                for _, row in w_top.iterrows()
            ],
        },
        "swell": {
            "n_subjects": int(len(set(s_subjects))) if len(s_subjects) else int(len(s_subject_counts)),
            "n_samples": int(s_n_samples),
            "n_features": int(len(s_feature_names)),
            "class_distribution": {
                "0": int(s_class_counts[0]) if s_class_counts else 0,
                "1": int(s_class_counts[1]) if s_class_counts else 0,
            },
            "missing_values": int(s_missing_values),
            "missing_ratio": float(s_missing_ratio),
            "modalities": swell_modalities,
            "feature_info": s_feature_info,
            "missing_by_modality": per_modality_missing,
            "merged_missing_sample": merged_missing,
            "rows_by_modality": s_rows_by_modality,
            "mode": s_mode,
            "label_mapping": SWELL_LABEL_MAPPING,
            "subjects": sorted(set(s_subjects.tolist())) if len(s_subjects) else [],
            "corr_summary": _corr_summary(s_corr),
            "top_label_corrs": [
                {"feature": row["feature"], "corr": float(row["corr"])}
                for _, row in s_top.iterrows()
            ],
        },
    }

    shared = sorted(set(w_feature_names) & set(s_feature_names))
    summary["shared_features"] = {
        "count": len(shared),
        "names": shared[:50],
        "truncated": len(shared) > 50,
    }

    # Plots
    _plot_overview(summary, out_dir)
    _plot_class_balance(summary, out_dir)
    _plot_samples_per_subject(w_subject_counts, s_subject_counts, out_dir)
    _plot_missingness_modalities(per_modality_missing, out_dir)
    _plot_feature_overlap(w_feature_names, s_feature_names, out_dir)
    _plot_feature_groups(
        _group_wesad_features(w_feature_names),
        s_feature_info,
        out_dir,
    )
    _plot_corr_heatmaps(w_corr, w_corr_names, s_corr, s_corr_names, out_dir)
    _plot_corr_distributions(w_corr, s_corr, out_dir)
    _plot_top_label_corrs(w_top, s_top, out_dir)
    _plot_label_tables(out_dir)
    _write_feature_table(w_feature_names, s_feature_names, out_dir)

    _write_summary(summary, out_dir)

    print("Output written to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
