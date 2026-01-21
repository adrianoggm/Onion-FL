#!/usr/bin/env python3
"""
Train baseline XGBoost model on SWEET selection1 dataset.
This pre-trained model will be used as the starting point for federated fine-tuning on selection2.

Usage:
    python scripts/train_sweet_baseline_selection1.py

Output:
    - baseline_models/sweet/xgboost_tuned_model.json: Pre-trained XGBoost model
    - baseline_models/sweet/scaler.json: Fitted StandardScaler
    - baseline_models/sweet/training_report.json: Training metrics and configuration
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flower_basic.datasets.sweet_samples import load_sweet_sample_full


def main():
    print("=" * 80)
    print("SWEET Selection1 Baseline Training (Pre-training for Transfer Learning)")
    print("=" * 80)

    # Configuration
    DATA_DIR = "data/SWEET/selection1/users"
    LABEL_STRATEGY = "ordinal_3class"
    OUTPUT_DIR = Path("baseline_models/sweet")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # XGBoost optimal hyperparameters from hypertuning_results/XGBoost_Tuned_tuning.json
    XGBOOST_PARAMS = {
        "max_depth": 4,
        "n_estimators": 300,
        "learning_rate": 0.01,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 2,
        "gamma": 0.3,
        "min_child_weight": 1,
        "objective": "multi:softmax",
        "num_class": 3,
        "random_state": 42,
        "n_jobs": -1,
    }

    # Load selection1 dataset
    print(f"\n[1/5] Loading SWEET selection1 from {DATA_DIR}...")
    X, y, subject_ids, feature_names = load_sweet_sample_full(
        data_dir=DATA_DIR,
        label_strategy=LABEL_STRATEGY,
        elevated_threshold=2.0,
        min_samples_per_subject=5,
    )

    unique_subjects = np.unique(subject_ids)
    print(f"  ✓ Loaded: {len(X)} samples from {len(unique_subjects)} subjects")
    print(f"  ✓ Features: {len(feature_names)}")
    print(f"  ✓ Class distribution: {np.bincount(y)}")

    # Fit global scaler on all training data
    print("\n[2/5] Fitting global StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    scaler_path = OUTPUT_DIR / "scaler.json"
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "feature_names": feature_names,
    }
    scaler_path.write_text(json.dumps(scaler_data, indent=2))
    print(f"  ✓ Scaler saved to {scaler_path}")

    # Subject-level 5-fold cross-validation
    print("\n[3/5] Subject-level 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use subject-level stratification
    subject_labels = []
    for subj in unique_subjects:
        subj_mask = subject_ids == subj
        # Use majority class for subject
        subject_labels.append(np.bincount(y[subj_mask]).argmax())

    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(unique_subjects, subject_labels), 1
    ):
        train_subjects = unique_subjects[train_idx]
        val_subjects = unique_subjects[val_idx]

        train_mask = np.isin(subject_ids, train_subjects)
        val_mask = np.isin(subject_ids, val_subjects)

        X_train_fold = X_scaled[train_mask]
        y_train_fold = y[train_mask]
        X_val_fold = X_scaled[val_mask]
        y_val_fold = y[val_mask]

        # Train XGBoost
        model = xgb.XGBClassifier(**XGBOOST_PARAMS, verbosity=0)
        model.fit(X_train_fold, y_train_fold)

        # Evaluate
        y_pred = model.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_pred)
        cv_scores.append(acc)

        print(
            f"  Fold {fold}/5: {acc:.4f} ({len(train_subjects)} train subjects, {len(val_subjects)} val subjects)"
        )

    mean_cv_acc = np.mean(cv_scores)
    std_cv_acc = np.std(cv_scores)
    print(f"\n  ✓ Mean CV Accuracy: {mean_cv_acc:.4f} ± {std_cv_acc:.4f}")

    # Train final model on all data
    print("\n[4/5] Training final model on all selection1 data...")
    final_model = xgb.XGBClassifier(**XGBOOST_PARAMS, verbosity=0)
    final_model.fit(X_scaled, y)

    # Final predictions
    y_pred_final = final_model.predict(X_scaled)
    final_acc = accuracy_score(y, y_pred_final)
    print(f"  ✓ Training accuracy: {final_acc:.4f}")

    # Save model
    model_path = OUTPUT_DIR / "xgboost_tuned_model.json"
    final_model.save_model(model_path)
    print(f"  ✓ Model saved to {model_path}")

    # Save training report
    print("\n[5/5] Saving training report...")
    report = {
        "dataset": {
            "name": "SWEET selection1",
            "data_dir": DATA_DIR,
            "label_strategy": LABEL_STRATEGY,
            "num_subjects": int(len(unique_subjects)),
            "num_samples": int(len(X)),
            "num_features": int(len(feature_names)),
            "class_distribution": {
                "0_low": int(np.sum(y == 0)),
                "1_medium": int(np.sum(y == 1)),
                "2_high": int(np.sum(y == 2)),
            },
        },
        "model": {
            "type": "XGBoost",
            "hyperparameters": XGBOOST_PARAMS,
        },
        "training": {
            "cv_method": "5-fold subject-level stratified",
            "cv_scores": [float(s) for s in cv_scores],
            "mean_cv_accuracy": float(mean_cv_acc),
            "std_cv_accuracy": float(std_cv_acc),
            "final_train_accuracy": float(final_acc),
        },
        "outputs": {
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
        },
    }

    report_path = OUTPUT_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  ✓ Report saved to {report_path}")

    print("\n" + "=" * 80)
    print("✅ SWEET Selection1 Pre-training Complete!")
    print(f"   Mean CV Accuracy: {mean_cv_acc:.4f} ± {std_cv_acc:.4f}")
    print(f"   Final Model: {model_path}")
    print(f"   Ready for federated fine-tuning on selection2")
    print("=" * 80)


if __name__ == "__main__":
    main()
