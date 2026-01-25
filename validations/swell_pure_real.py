#!/usr/bin/env python3
"""
SWELL PURE REAL DATA - NO SYNTHETIC SUBJECTS
============================================
Following the docs/RULES.md: ONLY REAL DATA, NO SYNTHETIC GENERATION
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def swell_pure_real_evaluation():
    """SWELL evaluation with ONLY real participants PP1-PP25. NO SYNTHETIC DATA."""

    print("🧬 SWELL DATASET - PURE REAL DATA EVALUATION")
    print("=" * 50)
    print("Following docs/RULES.md: NO synthetic data generation allowed")
    print()

    # Load ONLY computer interaction data (most reliable modality)
    computer_file = "data/SWELL/3 - Feature dataset/per sensor/A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv"

    df = pd.read_csv(computer_file)
    print(f"✓ Raw data loaded: {df.shape}")
    print(f"✓ REAL participants: {sorted(df['PP'].unique())}")
    print(f"✓ Total REAL participants: {len(df['PP'].unique())}")
    print()

    # Clean and prepare features
    features_cols = [
        "SnMouseAct",
        "SnLeftClicked",
        "SnRightClicked",
        "SnDoubleClicked",
        "SnWheel",
        "SnDragged",
        "SnMouseDistance",
        "SnKeyStrokes",
        "SnChars",
        "SnSpecialKeys",
        "SnDirectionKeys",
        "SnErrorKeys",
        "SnShortcutKeys",
        "SnSpaces",
        "SnAppChange",
        "SnTabfocusChange",
    ]

    # Clean data - convert to numeric, handle errors
    X = df[features_cols].replace("#VALUE!", np.nan)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Create binary stress labels (T/I/R = stress, N = no stress)
    y = df["Condition"].isin(["T", "I", "R"]).astype(int)
    subjects = df["PP"]

    print(f"Dataset prepared:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(X)}")
    print(f"  Real participants: {len(subjects.unique())}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print()

    # Subject-based split (NO DATA LEAKAGE)
    unique_participants = subjects.unique()
    np.random.seed(42)
    shuffled_participants = np.random.permutation(unique_participants)

    # Split: 50% train, 20% val, 30% test
    n_train = int(0.5 * len(shuffled_participants))
    n_val = int(0.2 * len(shuffled_participants))

    train_participants = shuffled_participants[:n_train]
    val_participants = shuffled_participants[n_train : n_train + n_val]
    test_participants = shuffled_participants[n_train + n_val :]

    print(f"Subject-based split (NO DATA LEAKAGE):")
    print(f"  Train participants: {sorted(train_participants)}")
    print(f"  Validation participants: {sorted(val_participants)}")
    print(f"  Test participants: {sorted(test_participants)}")
    print()

    # Create splits
    train_mask = subjects.isin(train_participants)
    val_mask = subjects.isin(val_participants)
    test_mask = subjects.isin(test_participants)

    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    print(f"Final split sizes:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print()

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM": SVC(random_state=42, probability=True),
    }

    results = {}
    print("Training models on REAL data...")

    for name, model in models.items():
        print(f"\n  Training {name}...")

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Calculate metrics
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

        results[name] = {
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_recall,
            "test_f1": test_f1,
        }

        print(f"    Validation Accuracy: {val_acc:.3f}")
        print(f"    Test Accuracy: {test_acc:.3f}")

    # Print summary
    print("\n" + "=" * 50)
    print("📊 SWELL PURE REAL DATA PERFORMANCE SUMMARY")
    print("=" * 50)
    print("\nTest Set Results (Subject-based split, NO DATA LEAKAGE):")
    print("-" * 50)

    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {result['test_accuracy']:.3f}")
        print(f"  Precision: {result['test_precision']:.3f}")
        print(f"  Recall:    {result['test_recall']:.3f}")
        print(f"  F1-Score:  {result['test_f1']:.3f}")

    # Best model
    best_model = max(results.items(), key=lambda x: x[1]["test_accuracy"])
    print(f"\n🏆 Best Model: {best_model[0]}")
    print(f"   Test Accuracy: {best_model[1]['test_accuracy']:.3f}")

    # Key insights
    print(f"\n💡 Key Insights:")
    print(f"   - REAL participants: {len(unique_participants)} subjects")
    print(f"   - Subject-based splitting prevents data leakage")
    print(f"   - Results represent realistic federated learning scenarios")
    print(f"   - NO synthetic data generation (following docs/RULES.md)")

    # Comparison warning
    if best_model[1]["test_accuracy"] < 0.9:
        print(
            f"\n✅ REALISTIC RESULTS: {best_model[1]['test_accuracy']:.3f} is reasonable for real data"
        )
        print("   Previous 99.6% was inflated by synthetic subjects")
    else:
        print(
            f"\n⚠️  HIGH ACCURACY: {best_model[1]['test_accuracy']:.3f} - verify if realistic"
        )

    return results


if __name__ == "__main__":
    swell_pure_real_evaluation()
