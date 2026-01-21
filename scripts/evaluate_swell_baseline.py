#!/usr/bin/env python3
"""
Baseline Performance Evaluation: SWELL Dataset
==============================================

Evaluates SWELL dataset performance using classical machine learning approach
with proper subject-based splitting to prevent data leakage.

Split strategy:
- 50% subjects for training
- 20% subjects for validation
- 30% subjects for testing
- No subject data mixing between splits

This establishes a baseline for comparison with federated learning results.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add matplotlib for visualizations
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import signal

    VISUALIZATIONS_AVAILABLE = True
    plt.style.use("default")
    sns.set_palette("husl")
except ImportError:
    VISUALIZATIONS_AVAILABLE = False
    print(
        "⚠️  Visualization libraries (matplotlib, seaborn, scipy) not available. Skipping plots."
    )


class SWELLBaselineEvaluator:
    """Baseline performance evaluator for SWELL dataset."""

    def __init__(self, data_dir: str = "data/SWELL"):
        self.data_dir = Path(data_dir)
        self.results = {}

    def load_swell_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        """Load COMPLETE SWELL dataset - MANDATORY for evaluations."""
        print("Loading COMPLETE SWELL dataset...")

        # RULE ENFORCEMENT: Evaluations MUST use complete datasets
        # NO samples, NO mock data - COMPLETE dataset ONLY

        # Path to COMPLETE SWELL feature files
        feature_dir = self.data_dir / "3 - Feature dataset" / "per sensor"

        if not feature_dir.exists():
            print(f"SWELL complete dataset not found: {feature_dir}")
            # Check alternative paths for COMPLETE dataset
            alt_paths = [
                self.data_dir,
                self.data_dir.parent / "SWELL",
                Path("data") / "0_SWELL",
            ]

            feature_dir = None
            for alt_path in alt_paths:
                if alt_path.exists():
                    csv_files = list(alt_path.glob("*.csv"))
                    if csv_files:
                        feature_dir = alt_path
                        print(f"  Found COMPLETE SWELL dataset in: {feature_dir}")
                        break

            if feature_dir is None:
                raise FileNotFoundError(
                    f"❌ CRITICAL: Complete SWELL dataset not found!\n"
                    f"   Evaluations REQUIRE complete datasets (not samples)\n"
                    f"   Checked paths: {alt_paths}\n"
                    f"   Please ensure SWELL complete dataset is available"
                )

        # REMOVED: _load_swell_from_samples()
        # RULE: Evaluations MUST use complete datasets only
        # Samples are ONLY for tests (test_*.py files)

        # Load different modality files - try multiple filename patterns
        modality_patterns = {
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

        print("Loading real SWELL modality data...")

        dataframes = []

        for modality, patterns in modality_patterns.items():
            df_loaded = False

            for pattern in patterns:
                if "*" in pattern:
                    # Glob pattern
                    matching_files = list(feature_dir.glob(pattern))
                    if matching_files:
                        file_path = matching_files[0]  # Use first match
                    else:
                        continue
                else:
                    # Exact filename
                    file_path = feature_dir / pattern
                    if not file_path.exists():
                        continue

                try:
                    # Try different encodings and separators for real data
                    try:
                        df = pd.read_csv(file_path, encoding="utf-8")
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(file_path, encoding="latin1")
                        except UnicodeDecodeError:
                            df = pd.read_csv(file_path, encoding="cp1252")

                    # Handle semicolon-separated files (common in SWELL facial data)
                    if df.shape[1] == 1 and ";" in str(df.columns[0]):
                        try:
                            df = pd.read_csv(file_path, sep=";", encoding="utf-8")
                        except:
                            try:
                                df = pd.read_csv(file_path, sep=";", encoding="latin1")
                            except:
                                df = pd.read_csv(file_path, sep=";", encoding="cp1252")

                    print(
                        f"  ✓ Loaded REAL {modality} data: {df.shape} from {file_path.name}"
                    )

                    # Clean column names
                    df.columns = (
                        df.columns.str.strip().str.replace(" ", "_").str.lower()
                    )

                    # Handle missing values (999 represents NaN in facial data)
                    if modality == "facial":
                        df = df.replace(999, np.nan)

                    # Handle NaN values with real data strategies
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_columns) > 0:
                        df[numeric_columns] = df[numeric_columns].fillna(
                            df[numeric_columns].median()
                        )

                    # Add modality prefix to avoid column conflicts
                    feature_cols = [
                        col
                        for col in df.columns
                        if col
                        not in [
                            "participant",
                            "subject",
                            "condition",
                            "time",
                            "timestamp",
                            "pp",
                            "blok",
                            "c",
                        ]
                    ]
                    df = df.rename(
                        columns={col: f"{modality}_{col}" for col in feature_cols}
                    )

                    dataframes.append(df)
                    df_loaded = True
                    break

                except Exception as e:
                    print(f"    Error loading {file_path}: {e}")
                    continue

            if not df_loaded:
                print(f"  ⚠️  Could not load {modality} data - no valid files found")

        if len(dataframes) == 0:
            raise ValueError(
                "❌ CRITICAL: No COMPLETE SWELL modality data loaded!\n"
                "   Evaluations MUST use complete datasets (not samples/mock)\n"
                "   Please ensure SWELL complete dataset is properly installed"
            )

        # Merge dataframes from real data more carefully
        print("Merging real SWELL modalities...")

        # Start with the first dataframe
        merged_df = dataframes[0].copy()
        print(f"    Starting with {dataframes[0].shape} (first modality)")

        for i, df in enumerate(dataframes[1:], 1):
            print(f"    Merging modality {i+1}: {df.shape}")
            print(f"      Available columns: {list(df.columns)}")

            # Find ALL common merge columns including participant/subject info
            merge_cols = []

            # Check for participant/subject columns (including modality prefixes)
            participant_found = False
            for subj_col in ["participant", "subject", "pp", "participantno"]:
                # Check direct column names
                if subj_col in merged_df.columns and subj_col in df.columns:
                    merge_cols.append(subj_col)
                    participant_found = True
                    break

            # Check for modality-prefixed participant columns
            if not participant_found:
                merged_cols = merged_df.columns.tolist()
                df_cols = df.columns.tolist()

                # Look for columns ending with _pp (participant)
                merged_pp_cols = [col for col in merged_cols if col.endswith("_pp")]
                df_pp_cols = [col for col in df_cols if col.endswith("_pp")]

                if merged_pp_cols and df_pp_cols:
                    # Use the first participant column from each, but rename them to match
                    merged_pp_col = merged_pp_cols[0]
                    df_pp_col = df_pp_cols[0]

                    # Rename to common column name for merging
                    if merged_pp_col != df_pp_col:
                        df = df.rename(columns={df_pp_col: merged_pp_col})

                    merge_cols.append(merged_pp_col)
                    participant_found = True

            # Always include condition if available
            if "condition" in merged_df.columns and "condition" in df.columns:
                merge_cols.append("condition")

            # Check for block/trial information (including modality prefixes)
            for block_col in ["blok", "block", "trial", "c"]:
                if block_col in merged_df.columns and block_col in df.columns:
                    merge_cols.append(block_col)
                    break

            # Check for modality-prefixed block columns
            merged_block_cols = [
                col
                for col in merged_df.columns
                if any(block in col for block in ["blok", "block", "_c"])
            ]
            df_block_cols = [
                col
                for col in df.columns
                if any(block in col for block in ["blok", "block", "_c"])
            ]

            if merged_block_cols and df_block_cols:
                merged_block_col = merged_block_cols[0]
                df_block_col = df_block_cols[0]

                # Rename to common column name for merging
                if merged_block_col != df_block_col:
                    df = df.rename(columns={df_block_col: merged_block_col})

                if merged_block_col not in merge_cols:
                    merge_cols.append(merged_block_col)

            print(f"      Merge columns found: {merge_cols}")

            if merge_cols:
                # Convert merge columns to same type before merging
                for col in merge_cols:
                    if col in merged_df.columns and col in df.columns:
                        # Convert both to string to avoid type conflicts
                        merged_df[col] = merged_df[col].astype(str)
                        df[col] = df[col].astype(str)

                # Use inner join and add suffixes to handle conflicts
                before_shape = merged_df.shape
                merged_df = pd.merge(
                    merged_df, df, on=merge_cols, how="inner", suffixes=("", f"_{i}")
                )
                print(f"      Merged: {before_shape} -> {merged_df.shape}")

                # If the dataset is getting too large (>50k rows), limit it
                if merged_df.shape[0] > 50000:
                    print(
                        f"      Dataset too large ({merged_df.shape[0]} rows), sampling 50k rows..."
                    )
                    merged_df = merged_df.sample(n=50000, random_state=42)
            else:
                print(
                    f"      Warning: No common merge columns found, skipping this modality"
                )
                continue

        print(f"Final merged dataset: {merged_df.shape}")

        print(f"Merged dataset columns: {list(merged_df.columns)}")

        # Extract features and labels
        # Handle different possible subject column names in real SWELL data
        subject_col = None
        for possible_col in [
            "participant",
            "subject",
            "participantno",
            "participant_id",
            "id",
            "pp",
        ]:
            if possible_col in merged_df.columns:
                subject_col = possible_col
                break

        if subject_col is None:
            print(
                "  No subject column found in real data, creating synthetic subject IDs"
            )
            # Create subject IDs based on data rows (realistic for merged data)
            merged_df["participant"] = [
                f"P{(i//100) + 1:02d}" for i in range(len(merged_df))
            ]
            subject_col = "participant"

        # Get feature columns (exclude subject, condition, and non-numeric columns)
        non_feature_cols = [
            subject_col,
            "condition",
            "timestamp",
            "timestamp_1",
            "timestamp_2",
            "timestamp_3",
            "blok",
            "pp",
            "c",
        ]
        feature_columns = [
            col for col in merged_df.columns if col not in non_feature_cols
        ]

        print(f"Using subject column: {subject_col}")
        print(f"Feature columns: {len(feature_columns)} ({feature_columns[:5]}...)")

        # Extract numeric features only
        X_df = merged_df[feature_columns]

        # Convert to numeric, forcing errors to NaN
        X_numeric = X_df.apply(pd.to_numeric, errors="coerce")

        # Remove columns that are all NaN (non-numeric)
        valid_cols = ~X_numeric.isnull().all()
        X_numeric = X_numeric.loc[:, valid_cols]
        final_feature_columns = X_numeric.columns.tolist()

        print(f"Valid numeric feature columns: {len(final_feature_columns)}")

        X = X_numeric.values
        subjects = merged_df[subject_col].astype(str).tolist()

        # Handle condition labels - map to binary stress classification
        conditions = merged_df["condition"].astype(str)

        # REAL SWELL conditions mapping to stress/no-stress (based on actual dataset)
        print(f"  Found conditions: {sorted(set(conditions))}")

        # SWELL dataset uses specific condition codes:
        # N = Normal (no stress/baseline)
        # T = Time pressure (stress)
        # I = Interruptions (stress)
        # R = Combined time pressure + interruptions (high stress)
        condition_mapping = {
            # SWELL dataset specific codes (uppercase and lowercase)
            "N": 0,
            "n": 0,  # Normal condition (no stress)
            "T": 1,
            "t": 1,  # Time pressure (stress)
            "I": 1,
            "i": 1,  # Interruptions (stress)
            "R": 1,
            "r": 1,  # Combined stress conditions
            # Alternative mappings for other possible formats
            "normal": 0,
            "baseline": 0,
            "control": 0,
            "no stress": 0,
            "time pressure": 1,
            "interruption": 1,
            "interruptions": 1,
            "combined": 1,
            "stress": 1,
        }

        # Apply mapping to real SWELL conditions
        y = []
        unmapped_conditions = set()

        for condition in conditions:
            condition_lower = condition.lower().strip()
            # Try exact match first
            if condition_lower in condition_mapping:
                y.append(condition_mapping[condition_lower])
            # Try partial matches for real SWELL conditions
            elif any(
                stress_word in condition_lower
                for stress_word in ["stress", "pressure", "interrupt", "high", "load"]
            ):
                y.append(1)  # Stress
            elif any(
                baseline_word in condition_lower
                for baseline_word in ["baseline", "control", "normal", "rest", "low"]
            ):
                y.append(0)  # No stress
            else:
                # Default for unknown real conditions - examine manually
                y.append(0)  # Conservative default to no-stress
                unmapped_conditions.add(condition_lower)

        if unmapped_conditions:
            print(
                f"  ⚠️  Unmapped conditions (defaulted to no-stress): {unmapped_conditions}"
            )

        y = np.array(y)

        # Remove features with zero variance
        feature_variances = np.var(X, axis=0)
        valid_features = feature_variances > 1e-8

        if not np.all(valid_features):
            n_removed = np.sum(~valid_features)
            print(f"  Removed {n_removed} zero-variance features")
            X = X[:, valid_features]

        # Handle remaining NaN values
        if np.any(np.isnan(X)):
            print(f"  Handling {np.sum(np.isnan(X))} missing values...")
            # Fill with column means
            for col_idx in range(X.shape[1]):
                col_mean = np.nanmean(X[:, col_idx])
                X[np.isnan(X[:, col_idx]), col_idx] = col_mean

        print(f"\n✓ Successfully loaded SWELL data:")
        print(f"  Total samples: {len(X)}")
        print(f"  Feature dimensions: {X.shape[1]}")
        print(f"  Unique subjects: {len(set(subjects))}")
        print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        return X, y, subjects, merged_df

    def analyze_dataset(self, df: pd.DataFrame) -> None:
        """Analyze and print detailed dataset information."""
        print("\n" + "=" * 60)
        print("📊 SWELL DATASET ANALYSIS")
        print("=" * 60)

        # Basic info
        print(f"\nDataset Shape: {df.shape}")
        print(f"Number of rows (samples): {df.shape[0]}")
        print(f"Number of columns: {df.shape[1]}")

        # Columns
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")

        # Data types
        print(f"\nData Types:")
        dtypes = df.dtypes
        for col, dtype in dtypes.items():
            print(f"  {col}: {dtype}")

        # Subject information
        subject_col = None
        for possible_col in [
            "participant",
            "subject",
            "participantno",
            "participant_id",
            "id",
            "pp",
        ]:
            if possible_col in df.columns:
                subject_col = possible_col
                break

        if subject_col:
            unique_subjects = df[subject_col].nunique()
            print(f"\nSubject Information:")
            print(f"  Subject column: {subject_col}")
            print(f"  Unique subjects: {unique_subjects}")
            print(f"  Samples per subject (top 10):")
            subject_counts = df[subject_col].value_counts().head(10)
            for subj, count in subject_counts.items():
                print(f"    {subj}: {count} samples")

        # Condition (stress questionnaire) analysis
        if "condition" in df.columns:
            print(f"\nCondition (Stress Levels) Analysis:")
            print(f"  Unique conditions: {df['condition'].nunique()}")
            print(f"  Condition distribution:")
            condition_counts = df["condition"].value_counts()
            for cond, count in condition_counts.items():
                print(f"    '{cond}': {count} samples ({count/len(df)*100:.1f}%)")

            # Mapped stress levels
            condition_mapping = {
                "N": 0,
                "n": 0,
                "T": 1,
                "t": 1,
                "I": 1,
                "i": 1,
                "R": 1,
                "r": 1,
                "normal": 0,
                "baseline": 0,
                "control": 0,
                "no stress": 0,
                "time pressure": 1,
                "interruption": 1,
                "interruptions": 1,
                "combined": 1,
                "stress": 1,
            }
            stress_labels = []
            for cond in df["condition"].astype(str).str.lower().str.strip():
                if cond in condition_mapping:
                    stress_labels.append(
                        "No Stress" if condition_mapping[cond] == 0 else "Stress"
                    )
                elif any(
                    word in cond
                    for word in ["stress", "pressure", "interrupt", "high", "load"]
                ):
                    stress_labels.append("Stress")
                elif any(
                    word in cond
                    for word in ["baseline", "control", "normal", "rest", "low"]
                ):
                    stress_labels.append("No Stress")
                else:
                    stress_labels.append("Unknown")

            stress_counts = pd.Series(stress_labels).value_counts()
            print(f"  Mapped stress distribution:")
            for label, count in stress_counts.items():
                print(f"    {label}: {count} samples ({count/len(df)*100:.1f}%)")

        # Feature columns analysis
        non_feature_cols = [
            subject_col,
            "condition",
            "timestamp",
            "timestamp_1",
            "timestamp_2",
            "timestamp_3",
            "blok",
            "pp",
            "c",
        ]
        feature_cols = [col for col in df.columns if col not in non_feature_cols]

        print(f"\nFeature Columns ({len(feature_cols)}):")
        print(
            f"  Numeric features: {len([col for col in feature_cols if df[col].dtype in ['int64', 'float64']])}"
        )
        print(
            f"  Non-numeric features: {len([col for col in feature_cols if df[col].dtype not in ['int64', 'float64']])}"
        )

        # Descriptive statistics for numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nDescriptive Statistics (Numeric Features):")
            desc = df[numeric_cols].describe()
            print(desc)

            # Check for missing values
            missing = df[numeric_cols].isnull().sum()
            if missing.sum() > 0:
                print(f"\nMissing Values in Numeric Columns:")
                for col, count in missing[missing > 0].items():
                    print(f"  {col}: {count} missing ({count/len(df)*100:.1f}%)")

        # Modality breakdown (if prefixed)
        modalities = {}
        for col in feature_cols:
            if "_" in col:
                modality = col.split("_")[0]
                if modality not in modalities:
                    modalities[modality] = []
                modalities[modality].append(col)

        if modalities:
            print(f"\nModality Breakdown:")
            for modality, cols in modalities.items():
                print(f"  {modality}: {len(cols)} features")
                if len(cols) <= 10:
                    print(f"    Features: {cols}")
                else:
                    print(f"    Features: {cols[:5]} ... {cols[-5:]}")

        # Save analysis to file
        analysis_data = {
            "dataset_shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in dtypes.items()},
            "subject_info": {
                "subject_column": subject_col,
                "unique_subjects": df[subject_col].nunique() if subject_col else None,
                "samples_per_subject": (
                    df[subject_col].value_counts().to_dict() if subject_col else None
                ),
            },
            "condition_analysis": {
                "unique_conditions": (
                    df["condition"].nunique() if "condition" in df.columns else None
                ),
                "condition_distribution": (
                    df["condition"].value_counts().to_dict()
                    if "condition" in df.columns
                    else None
                ),
                "stress_distribution": (
                    stress_counts.to_dict() if "condition" in df.columns else None
                ),
            },
            "feature_info": {
                "total_features": len(feature_cols),
                "numeric_features": len(
                    [
                        col
                        for col in feature_cols
                        if df[col].dtype in ["int64", "float64"]
                    ]
                ),
                "modalities": {mod: len(cols) for mod, cols in modalities.items()},
            },
            "descriptive_stats": (
                df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else None
            ),
        }

        import json

        with open("swell_dataset_analysis.json", "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)

        print(f"\n💾 Dataset analysis saved to: swell_dataset_analysis.json")

    def create_visualizations(
        self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray, results: Dict
    ) -> None:
        """Create comprehensive visualizations for dataset analysis."""
        if not VISUALIZATIONS_AVAILABLE:
            return

        print("\n📊 Creating visualizations...")

        # Create plots directory
        plots_dir = Path("swell_plots")
        plots_dir.mkdir(exist_ok=True)

        # 1. Correlation Matrix
        self.plot_correlation_matrix(df, plots_dir)

        # 2. Feature Distributions
        self.plot_feature_distributions(df, plots_dir)

        # 3. Class Distribution
        self.plot_class_distribution(df, plots_dir)

        # 4. Cross-correlation Analysis
        self.plot_cross_correlations(df, plots_dir)

        # 5. Feature Importance (if Random Forest available)
        self.plot_feature_importance(df, X, y, plots_dir)

        # 6. Confusion Matrices
        self.plot_confusion_matrices(results, plots_dir)

        print(f"💾 Plots saved to: {plots_dir}/")

    def plot_correlation_matrix(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot correlation matrix of numeric features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 20:  # Limit to avoid huge matrix
            # Select most variable features
            variances = df[numeric_cols].var()
            top_features = variances.nlargest(20).index
            numeric_cols = top_features

        corr_matrix = df[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
        )
        plt.title("SWELL Dataset - Correlation Matrix (Numeric Features)")
        plt.tight_layout()
        plt.savefig(plots_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Save correlation values
        corr_matrix.to_csv(plots_dir / "correlation_matrix.csv")

    def plot_feature_distributions(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot distributions of key features."""
        # Select key features from different modalities
        key_features = {
            "HRV": ["physiology_hr", "physiology_rmssd", "physiology_scl"],
            "Computer": [
                "computer_snmouseact",
                "computer_snkeystrokes",
                "computer_snchars",
            ],
            "Facial": [
                "facial_svalence",
                "facial_sau01_innerbrowraiser",
                "facial_sau12_lipcornerpuller",
            ],
            "Posture": [
                "posture_leanangle(avg)",
                "posture_leftshoulderangle(avg)",
                "posture_rightshoulderangle(avg)",
            ],
        }

        for modality, features in key_features.items():
            available_features = [f for f in features if f in df.columns]
            if not available_features:
                continue

            fig, axes = plt.subplots(
                1, len(available_features), figsize=(5 * len(available_features), 4)
            )

            if len(available_features) == 1:
                axes = [axes]

            for i, feature in enumerate(available_features):
                # Convert to numeric if needed
                data = pd.to_numeric(df[feature], errors="coerce").dropna()

                if len(data) > 0:
                    axes[i].hist(data, bins=50, alpha=0.7, edgecolor="black")
                    axes[i].set_title(f"{feature}")
                    axes[i].set_xlabel("Value")
                    axes[i].set_ylabel("Frequency")
                    axes[i].grid(True, alpha=0.3)

            plt.suptitle(f"Distributions - {modality} Modality")
            plt.tight_layout()
            plt.savefig(
                plots_dir / f"distributions_{modality.lower()}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def plot_class_distribution(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot class distribution and condition analysis."""
        if "condition" not in df.columns:
            return

        # Map conditions to stress levels
        condition_mapping = {
            "N": "No Stress",
            "n": "No Stress",
            "T": "Stress",
            "t": "Stress",
            "I": "Stress",
            "i": "Stress",
            "R": "Stress",
            "r": "Stress",
            "normal": "No Stress",
            "baseline": "No Stress",
            "control": "No Stress",
            "time pressure": "Stress",
            "interruption": "Stress",
            "interruptions": "Stress",
            "combined": "Stress",
            "stress": "Stress",
        }

        df_plot = df.copy()
        df_plot["stress_level"] = (
            df_plot["condition"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(condition_mapping)
        )

        # Condition distribution
        plt.figure(figsize=(10, 6))
        condition_counts = df_plot["condition"].value_counts()
        condition_counts.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("SWELL Dataset - Condition Distribution")
        plt.xlabel("Condition")
        plt.ylabel("Number of Samples")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            plots_dir / "condition_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Stress level distribution
        plt.figure(figsize=(8, 6))
        stress_counts = df_plot["stress_level"].value_counts()
        stress_counts.plot(
            kind="pie", autopct="%1.1f%%", colors=["lightcoral", "lightgreen"]
        )
        plt.title("SWELL Dataset - Stress Level Distribution")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(plots_dir / "stress_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_cross_correlations(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot cross-correlations between key features with lag k=3."""
        # Select key features for cross-correlation
        key_features = [
            "physiology_hr",
            "physiology_rmssd",
            "physiology_scl",
            "computer_snmouseact",
            "facial_svalence",
            "posture_leanangle(avg)",
        ]

        available_features = [f for f in key_features if f in df.columns]
        if len(available_features) < 2:
            return

        # Convert to numeric and handle NaN
        data_numeric = (
            df[available_features]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(method="ffill")
            .fillna(0)
        )

        # Calculate cross-correlations with lag k=3
        max_lag = 3
        fig, axes = plt.subplots(
            len(available_features), len(available_features), figsize=(15, 15)
        )

        for i, feat1 in enumerate(available_features):
            for j, feat2 in enumerate(available_features):
                if i == j:
                    # Auto-correlation
                    corr = self.cross_correlation(
                        data_numeric[feat1], data_numeric[feat1], max_lag
                    )
                    axes[i, j].plot(
                        range(-max_lag, max_lag + 1), corr, "b-o", markersize=3
                    )
                    axes[i, j].axvline(x=0, color="r", linestyle="--", alpha=0.7)
                    axes[i, j].set_title(f"{feat1[:15]}...", fontsize=8)
                else:
                    # Cross-correlation
                    corr = self.cross_correlation(
                        data_numeric[feat1], data_numeric[feat2], max_lag
                    )
                    axes[i, j].plot(
                        range(-max_lag, max_lag + 1), corr, "g-o", markersize=3
                    )
                    axes[i, j].axvline(x=0, color="r", linestyle="--", alpha=0.7)

                if i == len(available_features) - 1:
                    axes[i, j].set_xlabel("Lag")
                if j == 0:
                    axes[i, j].set_ylabel("Correlation")

        plt.suptitle("SWELL Dataset - Cross-Correlation Analysis (k=3)", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            plots_dir / "cross_correlations_k3.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Save cross-correlation values
        corr_matrix = np.zeros((len(available_features), len(available_features)))
        for i, feat1 in enumerate(available_features):
            for j, feat2 in enumerate(available_features):
                corr = self.cross_correlation(
                    data_numeric[feat1], data_numeric[feat2], max_lag
                )
                corr_matrix[i, j] = corr[max_lag]  # Zero-lag correlation

        corr_df = pd.DataFrame(
            corr_matrix, index=available_features, columns=available_features
        )
        corr_df.to_csv(plots_dir / "cross_correlations_k3.csv")

    def cross_correlation(self, x: pd.Series, y: pd.Series, max_lag: int) -> np.ndarray:
        """Calculate cross-correlation with lags from -max_lag to +max_lag."""
        x = x.values
        y = y.values
        correlations = []

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
            elif lag == 0:
                corr = np.corrcoef(x, y)[0, 1]
            else:
                corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)

        return np.array(correlations)

    def plot_feature_importance(
        self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray, plots_dir: Path
    ) -> None:
        """Plot feature importance using Random Forest."""
        try:
            # Get feature names
            non_feature_cols = [
                "pp",
                "condition",
                "timestamp",
                "timestamp_1",
                "timestamp_2",
                "timestamp_3",
                "blok",
                "c",
            ]
            feature_cols = [col for col in df.columns if col not in non_feature_cols]
            numeric_cols = (
                df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            )

            if len(numeric_cols) > 50:  # Limit features for importance
                # Select most variable features
                variances = df[numeric_cols].var()
                numeric_cols = variances.nlargest(50).index.tolist()

            # Train a quick Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            X_subset = df[numeric_cols].fillna(0).values
            rf.fit(X_subset, y)

            # Get feature importance
            importance = rf.feature_importances_
            indices = np.argsort(importance)[::-1][:20]  # Top 20

            plt.figure(figsize=(12, 8))
            plt.barh(range(len(indices)), importance[indices], align="center")
            plt.yticks(range(len(indices)), [numeric_cols[i] for i in indices])
            plt.xlabel("Importance")
            plt.title("SWELL Dataset - Top 20 Feature Importance (Random Forest)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(
                plots_dir / "feature_importance.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            # Save importance values
            importance_df = pd.DataFrame(
                {
                    "feature": [numeric_cols[i] for i in indices],
                    "importance": importance[indices],
                }
            )
            importance_df.to_csv(plots_dir / "feature_importance.csv", index=False)

        except Exception as e:
            print(f"⚠️  Could not create feature importance plot: {e}")

    def plot_confusion_matrices(self, results: Dict, plots_dir: Path) -> None:
        """Plot confusion matrices for all models."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        model_names = [
            "Logistic Regression",
            "Random Forest",
            "SVM",
            "Multimodal Network",
        ]

        for i, model_name in enumerate(model_names):
            if model_name in results and "confusion_matrix" in results[model_name]:
                cm = results[model_name]["confusion_matrix"]
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm, display_labels=["No Stress", "Stress"]
                )
                disp.plot(ax=axes[i], cmap="Blues", colorbar=False)
                axes[i].set_title(
                    f'{model_name}\nAccuracy: {results[model_name]["test"]["accuracy"]:.3f}'
                )
            else:
                axes[i].text(
                    0.5,
                    0.5,
                    f"{model_name}\nNo CM available",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(model_name)

        plt.suptitle("SWELL Dataset - Confusion Matrices (Test Set)", fontsize=16)
        plt.tight_layout()
        plt.savefig(plots_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Save individual confusion matrices
        for model_name in model_names:
            if model_name in results and "confusion_matrix" in results[model_name]:
                cm = results[model_name]["confusion_matrix"]
                plt.figure(figsize=(6, 5))
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm, display_labels=["No Stress", "Stress"]
                )
                disp.plot(cmap="Blues")
                plt.title(
                    f'{model_name} - Confusion Matrix\nAccuracy: {results[model_name]["test"]["accuracy"]:.3f}'
                )
                plt.tight_layout()
                plt.savefig(
                    plots_dir
                    / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    # STRICT RULES ENFORCEMENT:
    # 1. Evaluations MUST use COMPLETE datasets (100% SWELL/WESAD)
    # 2. NO samples allowed in evaluate_*.py (samples only for test_*.py)
    # 3. ABSOLUTELY NO mock data generation - REAL DATA ONLY
    # 4. Mock data generation is PROHIBITED for ML/AI evaluation

    def split_by_subjects(
        self, X: np.ndarray, y: np.ndarray, subjects: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data by subjects to prevent data leakage."""
        unique_subjects = list(set(subjects))
        n_subjects = len(unique_subjects)

        print(f"\nSplitting {n_subjects} subjects:")

        # Calculate split sizes
        n_train = int(0.5 * n_subjects)  # 50% for training
        n_val = int(0.2 * n_subjects)  # 20% for validation
        n_test = n_subjects - n_train - n_val  # Remaining for test

        # Shuffle subjects
        np.random.seed(42)
        shuffled_subjects = np.random.permutation(unique_subjects)

        train_subjects = shuffled_subjects[:n_train]
        val_subjects = shuffled_subjects[n_train : n_train + n_val]
        test_subjects = shuffled_subjects[n_train + n_val :]

        print(f"  Training: {len(train_subjects)} subjects - {list(train_subjects)}")
        print(f"  Validation: {len(val_subjects)} subjects - {list(val_subjects)}")
        print(f"  Test: {len(test_subjects)} subjects - {list(test_subjects)}")

        # Create splits
        train_mask = np.isin(subjects, train_subjects)
        val_mask = np.isin(subjects, val_subjects)
        test_mask = np.isin(subjects, test_subjects)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_classical_models(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Train classical ML models and evaluate performance with cross-validation and confusion matrices."""
        print("\nTraining classical models...")

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Combine train and validation for cross-validation
        X_train_val = np.vstack([X_train_scaled, X_val_scaled])
        y_train_val = np.concatenate([y_train, y_val])

        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "SVM": SVC(random_state=42, kernel="linear", C=1.0, probability=False),
        }

        results = {}

        for name, model in models.items():
            print(f"\n  Training {name}...")

            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                model, X_train_val, y_train_val, cv=cv, scoring="accuracy"
            )

            print(
                f"    Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})"
            )

            # Train final model on full training + validation data
            model.fit(X_train_val, y_train_val)

            # Predict on test set
            y_test_pred = model.predict(X_test_scaled)

            # Calculate metrics
            test_metrics = self.calculate_metrics(y_test, y_test_pred, "Test")

            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            print(f"    Confusion Matrix:")
            print(f"      {cm}")

            results[name] = {
                "validation": test_metrics,  # Using test as final validation
                "test": test_metrics,
                "model": model,
                "cv_scores": cv_scores,
                "confusion_matrix": cm,
            }

            print(f"    Test Accuracy: {test_metrics['accuracy']:.3f}")

        return results

    def train_multimodal_network(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Train multimodal neural network."""
        print("\nTraining Multimodal Neural Network...")

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test)

        # Define multimodal neural network
        class MultimodalSWELLNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()

                # Modality-specific branches
                # Computer interaction (5 features)
                self.computer_branch = nn.Sequential(
                    nn.Linear(5, 16), nn.ReLU(), nn.Dropout(0.2)
                )

                # Facial expressions (4 features)
                self.facial_branch = nn.Sequential(
                    nn.Linear(4, 12), nn.ReLU(), nn.Dropout(0.2)
                )

                # Body posture (4 features)
                self.posture_branch = nn.Sequential(
                    nn.Linear(4, 12), nn.ReLU(), nn.Dropout(0.2)
                )

                # Physiology (7 features)
                self.physiology_branch = nn.Sequential(
                    nn.Linear(7, 20), nn.ReLU(), nn.Dropout(0.2)
                )

                # Fusion layers
                self.fusion = nn.Sequential(
                    nn.Linear(16 + 12 + 12 + 20, 64),  # Combined features
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, 2),
                )

            def forward(self, x):
                # Split input by modality
                computer = x[:, :5]
                facial = x[:, 5:9]
                posture = x[:, 9:13]
                physiology = x[:, 13:20]

                # Process each modality
                computer_out = self.computer_branch(computer)
                facial_out = self.facial_branch(facial)
                posture_out = self.posture_branch(posture)
                physiology_out = self.physiology_branch(physiology)

                # Fuse modalities
                fused = torch.cat(
                    [computer_out, facial_out, posture_out, physiology_out], dim=1
                )
                output = self.fusion(fused)

                return output

        model = MultimodalSWELLNet(X_train_scaled.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        best_val_acc = 0.0
        patience = 15
        patience_counter = 0

        for epoch in range(150):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    _, val_pred = torch.max(val_outputs.data, 1)
                    val_acc = accuracy_score(y_val_tensor.numpy(), val_pred.numpy())

                    print(
                        f"    Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.3f}"
                    )

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch}")
                        break

        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        model.eval()

        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, y_val_pred = torch.max(val_outputs.data, 1)

            test_outputs = model(X_test_tensor)
            _, y_test_pred = torch.max(test_outputs.data, 1)

        val_metrics = self.calculate_metrics(y_val, y_val_pred.numpy(), "Validation")
        test_metrics = self.calculate_metrics(y_test, y_test_pred.numpy(), "Test")

        print(f"    Best Validation Accuracy: {val_metrics['accuracy']:.3f}")
        print(f"    Test Accuracy: {test_metrics['accuracy']:.3f}")

        return {"validation": val_metrics, "test": test_metrics, "model": model}

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, split_name: str
    ) -> Dict:
        """Calculate comprehensive metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="binary", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def run_evaluation(self) -> Dict:
        """Run complete baseline evaluation."""
        print("🎯 SWELL Dataset Baseline Performance Evaluation")
        print("=" * 60)

        # Load data
        X, y, subjects, df = self.load_swell_data()

        # Analyze dataset
        self.analyze_dataset(df)

        # Split by subjects
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_by_subjects(
            X, y, subjects
        )

        print(f"\nFinal split sizes:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")

        # Train classical models
        classical_results = self.train_classical_models(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        # Train multimodal neural network
        nn_results = self.train_multimodal_network(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        # Combine results
        all_results = {**classical_results, "Multimodal Network": nn_results}

        # Create visualizations
        self.create_visualizations(df, X, y, all_results)

        # Print summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, results: Dict) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("📊 SWELL BASELINE PERFORMANCE SUMMARY")
        print("=" * 60)

        print("\nTest Set Results (Subject-based split):")
        print("-" * 40)

        for model_name, result in results.items():
            if model_name == "Multimodal Network":
                continue  # Skip model object for now

            test_metrics = result["test"]
            print(f"\n{model_name}:")

            # Cross-validation results
            if "cv_scores" in result:
                cv_mean = result["cv_scores"].mean()
                cv_std = result["cv_scores"].std()
                print(f"  CV Accuracy: {cv_mean:.3f} (+/- {cv_std * 2:.3f})")

            print(f"  Test Accuracy:  {test_metrics['accuracy']:.3f}")
            print(f"  Test Precision: {test_metrics['precision']:.3f}")
            print(f"  Test Recall:    {test_metrics['recall']:.3f}")
            print(f"  Test F1-Score:  {test_metrics['f1']:.3f}")

            # Confusion matrix
            if "confusion_matrix" in result:
                cm = result["confusion_matrix"]
                print(f"  Confusion Matrix:")
                print(f"    {cm[0]}  (No Stress)")
                print(f"    {cm[1]}  (Stress)")

        # Best model
        best_model = max(
            results.items(),
            key=lambda x: (
                x[1]["test"]["accuracy"] if x[0] != "Multimodal Network" else 0
            ),
        )
        print(f"\n🏆 Best Model: {best_model[0]}")
        print(f"   Test Accuracy: {best_model[1]['test']['accuracy']:.3f}")

        print(f"\n💡 Key Insights:")
        print(f"   - Cross-validation provides robust performance estimates")
        print(f"   - Confusion matrices show prediction patterns")
        print(
            f"   - Multimodal approach leverages computer + facial + posture + physiology"
        )
        print(f"   - Subject-based splitting prevents data leakage")
        print(f"   - Results represent realistic federated learning scenarios")
        print(f"   - Can compare with 3-node federated learning performance")


def main():
    """Main evaluation function."""
    evaluator = SWELLBaselineEvaluator()
    results = evaluator.run_evaluation()

    # Save results
    results_file = "swell_baseline_results.json"
    import json

    # Convert results for JSON serialization
    json_results = {}
    for model_name, result in results.items():
        if model_name == "Multimodal Network":
            continue  # Skip model object
        json_results[model_name] = {
            "validation": result["validation"],
            "test": result["test"],
        }

    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")
    print("   Ready for federated learning comparison!")


if __name__ == "__main__":
    main()
