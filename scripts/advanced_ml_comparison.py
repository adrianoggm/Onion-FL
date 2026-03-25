"""Advanced ML models with 5-fold cross-validation for SWEET dataset.

This script tests multiple ML algorithms with proper subject-level CV to ensure
truly independent evaluation and identify whether performance is limited by
the dataset quality or model complexity.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not available, skipping XGBoost models")

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flower_basic.datasets.sweet_samples import load_sweet_sample_dataset


def create_subject_level_cv_splits(
    subject_ids: np.ndarray,
    y_per_subject: dict,
    n_splits: int = 5,
    random_state: int = 42,
):
    """Create CV splits at subject level (not sample level) to avoid data leakage.

    Args:
        subject_ids: Array of all subject IDs (repeated per sample)
        y_per_subject: Dict mapping subject_id -> most common class
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        List of (train_idx, test_idx) tuples
    """
    unique_subjects = np.unique(subject_ids)

    # Get label for each subject (use most common class)
    subject_labels = np.array([y_per_subject[subj] for subj in unique_subjects])

    # Create stratified K-fold at subject level
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_splits = []
    for train_subject_idx, test_subject_idx in skf.split(
        unique_subjects, subject_labels
    ):
        train_subjects = unique_subjects[train_subject_idx]
        test_subjects = unique_subjects[test_subject_idx]

        # Convert subject indices to sample indices
        train_sample_idx = np.isin(subject_ids, train_subjects)
        test_sample_idx = np.isin(subject_ids, test_subjects)

        cv_splits.append((np.where(train_sample_idx)[0], np.where(test_sample_idx)[0]))

    return cv_splits


def get_advanced_models():
    """Return dictionary of advanced ML models."""
    models = {
        # Simple baseline
        "Logistic_Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        # Tree-based models
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=10, class_weight="balanced", random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            random_state=42,
        ),
        # Neural Networks
        "MLP_Small": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=32,
            learning_rate="adaptive",
            max_iter=500,
            random_state=42,
        ),
        "MLP_Deep": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=32,
            learning_rate="adaptive",
            max_iter=500,
            random_state=42,
        ),
        "MLP_Wide": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=64,
            learning_rate="adaptive",
            max_iter=500,
            random_state=42,
        ),
        # Distance-based
        "KNN": KNeighborsClassifier(
            n_neighbors=15, weights="distance", metric="minkowski", n_jobs=-1
        ),
        "SVM_RBF": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            random_state=42,
            probability=True,
        ),
        # Probabilistic
        "Naive_Bayes": GaussianNB(),
    }

    # Add XGBoost if available
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=3,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

    return models


def train_and_evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    model_name: str,
    model: Any,
    cv_splits: list,
    verbose: bool = True,
) -> dict:
    """Train and evaluate a model using cross-validation.

    Args:
        X: Feature matrix
        y: Labels
        subject_ids: Subject IDs for each sample
        model_name: Name of the model
        model: Sklearn-compatible model
        cv_splits: List of (train_idx, test_idx) tuples
        verbose: Whether to print progress

    Returns:
        Dictionary with results
    """
    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
        if verbose:
            print(f"  Fold {fold}/{len(cv_splits)}...", end=" ")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        fold_results.append(
            {
                "fold": fold,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
                "train_subjects": len(np.unique(subject_ids[train_idx])),
                "test_subjects": len(np.unique(subject_ids[test_idx])),
            }
        )

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        if verbose:
            print(f"Acc: {acc:.4f}, F1: {f1:.4f}")

    # Aggregate results
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)

    # Per-class metrics
    class_report = classification_report(
        all_y_true, all_y_pred, output_dict=True, zero_division=0
    )

    # Summary statistics
    results = {
        "model_name": model_name,
        "fold_results": fold_results,
        "mean_accuracy": np.mean([f["accuracy"] for f in fold_results]),
        "std_accuracy": np.std([f["accuracy"] for f in fold_results]),
        "mean_f1": np.mean([f["f1"] for f in fold_results]),
        "std_f1": np.std([f["f1"] for f in fold_results]),
        "mean_precision": np.mean([f["precision"] for f in fold_results]),
        "mean_recall": np.mean([f["recall"] for f in fold_results]),
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
    }

    return results, all_y_true, all_y_pred, cm


def plot_results(results_dict: dict, output_dir: Path):
    """Create comprehensive visualizations of results."""

    # 1. Model comparison barplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    model_names = list(results_dict.keys())
    mean_accs = [results_dict[m]["mean_accuracy"] for m in model_names]
    std_accs = [results_dict[m]["std_accuracy"] for m in model_names]
    mean_f1s = [results_dict[m]["mean_f1"] for m in model_names]
    std_f1s = [results_dict[m]["std_f1"] for m in model_names]

    # Sort by accuracy
    sorted_idx = np.argsort(mean_accs)[::-1]
    model_names_sorted = [model_names[i] for i in sorted_idx]
    mean_accs_sorted = [mean_accs[i] for i in sorted_idx]
    std_accs_sorted = [std_accs[i] for i in sorted_idx]
    mean_f1s_sorted = [mean_f1s[i] for i in sorted_idx]
    std_f1s_sorted = [std_f1s[i] for i in sorted_idx]

    # Accuracy comparison
    ax1 = axes[0, 0]
    y_pos = np.arange(len(model_names_sorted))
    bars = ax1.barh(
        y_pos,
        mean_accs_sorted,
        xerr=std_accs_sorted,
        color="steelblue",
        alpha=0.7,
        capsize=5,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m.replace("_", " ") for m in model_names_sorted], fontsize=9)
    ax1.set_xlabel("Accuracy (5-Fold CV)", fontweight="bold")
    ax1.set_title("Model Comparison: Accuracy", fontweight="bold", fontsize=12)
    ax1.axvline(0.333, color="red", linestyle="--", alpha=0.5, label="Random (33.3%)")
    ax1.legend()
    ax1.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, acc, std in zip(bars, mean_accs_sorted, std_accs_sorted):
        ax1.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.3f}±{std:.3f}",
            va="center",
            fontsize=8,
        )

    # F1 comparison
    ax2 = axes[0, 1]
    bars = ax2.barh(
        y_pos,
        mean_f1s_sorted,
        xerr=std_f1s_sorted,
        color="forestgreen",
        alpha=0.7,
        capsize=5,
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([m.replace("_", " ") for m in model_names_sorted], fontsize=9)
    ax2.set_xlabel("F1-Score (5-Fold CV)", fontweight="bold")
    ax2.set_title("Model Comparison: F1-Score", fontweight="bold", fontsize=12)
    ax2.grid(axis="x", alpha=0.3)

    for bar, f1, std in zip(bars, mean_f1s_sorted, std_f1s_sorted):
        ax2.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{f1:.3f}±{std:.3f}",
            va="center",
            fontsize=8,
        )

    # Best model confusion matrix
    best_model = model_names_sorted[0]
    cm = np.array(results_dict[best_model]["confusion_matrix"])

    ax3 = axes[1, 0]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax3,
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"],
    )
    ax3.set_xlabel("Predicted", fontweight="bold")
    ax3.set_ylabel("True", fontweight="bold")
    ax3.set_title(
        f'Confusion Matrix: {best_model.replace("_", " ")}',
        fontweight="bold",
        fontsize=12,
    )

    # Per-class performance for best model
    ax4 = axes[1, 1]
    class_report = results_dict[best_model]["classification_report"]
    classes = ["0", "1", "2"]
    metrics_data = {
        "Precision": [class_report[c]["precision"] for c in classes],
        "Recall": [class_report[c]["recall"] for c in classes],
        "F1-Score": [class_report[c]["f1-score"] for c in classes],
    }

    x = np.arange(len(classes))
    width = 0.25

    ax4.bar(x - width, metrics_data["Precision"], width, label="Precision", alpha=0.8)
    ax4.bar(x, metrics_data["Recall"], width, label="Recall", alpha=0.8)
    ax4.bar(x + width, metrics_data["F1-Score"], width, label="F1-Score", alpha=0.8)

    ax4.set_xlabel("Class", fontweight="bold")
    ax4.set_ylabel("Score", fontweight="bold")
    ax4.set_title(
        f'Per-Class Metrics: {best_model.replace("_", " ")}',
        fontweight="bold",
        fontsize=12,
    )
    ax4.set_xticks(x)
    ax4.set_xticklabels(["Low Stress", "Medium Stress", "High Stress"])
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)
    ax4.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
    print(f"\n✅ Model comparison saved to: {output_dir / 'model_comparison.png'}")

    # 2. Individual confusion matrices for all models
    n_models = len(results_dict)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (model_name, results) in enumerate(results_dict.items()):
        cm = np.array(results["confusion_matrix"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            ax=axes[idx],
            xticklabels=["Low", "Med", "High"],
            yticklabels=["Low", "Med", "High"],
            cbar_kws={"shrink": 0.8},
        )

        acc = results["mean_accuracy"]
        axes[idx].set_title(
            f'{model_name.replace("_", " ")}\nAcc: {acc:.3f}',
            fontweight="bold",
            fontsize=11,
        )
        axes[idx].set_xlabel("Predicted", fontsize=9)
        axes[idx].set_ylabel("True", fontsize=9)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "all_confusion_matrices.png", dpi=300, bbox_inches="tight")
    print(
        f"✅ All confusion matrices saved to: {output_dir / 'all_confusion_matrices.png'}"
    )

    plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced ML models with 5-fold cross-validation for SWEET"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/SWEET/selection1/users",
        help="Path to SWEET users directory",
    )
    parser.add_argument(
        "--label-strategy",
        type=str,
        default="ordinal_3class",
        choices=["binary", "ordinal", "ordinal_3class"],
        help="Label strategy",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="advanced_ml_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("SWEET DATASET - ADVANCED ML COMPARISON WITH 5-FOLD CV")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Label strategy: {args.label_strategy}")
    print(f"CV folds: {args.n_folds}")
    print(f"Output directory: {output_dir}")
    print("\nℹ️  SUBJECT-LEVEL CV: Ensuring no subject appears in both train and test")

    # Load full dataset
    print("\nLoading SWEET dataset...")
    dataset = load_sweet_sample_dataset(
        data_dir=args.data_dir,
        label_strategy=args.label_strategy,
        train_fraction=0.6,
        val_fraction=0.2,
        random_state=args.seed,
    )

    # Combine all data
    X = np.vstack([dataset.train.X, dataset.val.X, dataset.test.X])
    y = np.concatenate([dataset.train.y, dataset.val.y, dataset.test.y])
    subject_ids = np.concatenate(
        [dataset.train.subject_ids, dataset.val.subject_ids, dataset.test.subject_ids]
    )

    print("✓ Dataset loaded")
    print(f"  Total samples: {len(y)}")
    print(f"  Total subjects: {len(np.unique(subject_ids))}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Create subject-level CV splits
    print(f"\nCreating {args.n_folds}-fold subject-level CV splits...")

    # Get most common class per subject
    y_per_subject = {}
    for subj in np.unique(subject_ids):
        subj_labels = y[subject_ids == subj]
        y_per_subject[subj] = np.bincount(subj_labels.astype(int)).argmax()

    cv_splits = create_subject_level_cv_splits(
        subject_ids, y_per_subject, args.n_folds, args.seed
    )

    print("✓ CV splits created")
    for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
        train_subjs = len(np.unique(subject_ids[train_idx]))
        test_subjs = len(np.unique(subject_ids[test_idx]))
        print(
            f"  Fold {fold}: {train_subjs} train subjects, {test_subjs} test subjects"
        )

    # Get models
    models = get_advanced_models()

    print(f"\n{'=' * 80}")
    print(f"TRAINING {len(models)} MODELS WITH {args.n_folds}-FOLD CV")
    print(f"{'=' * 80}\n")

    all_results = {}

    for model_name, model in models.items():
        print(f"📊 {model_name.replace('_', ' ')}:")

        try:
            results, y_true, y_pred, cm = train_and_evaluate_cv(
                X, y, subject_ids, model_name, model, cv_splits, verbose=True
            )

            all_results[model_name] = results

            print(
                f"  ✓ Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}"
            )
            print(
                f"  ✓ Mean F1-Score: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}"
            )

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    # Save results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")

    # Save JSON results
    results_file = output_dir / "cv_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Results saved to: {results_file}")

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_results(all_results, output_dir)

    # Print summary table
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 80}\n")

    # Sort by accuracy
    sorted_models = sorted(
        all_results.items(), key=lambda x: x[1]["mean_accuracy"], reverse=True
    )

    print(f"{'Rank':<5} {'Model':<25} {'Accuracy':<20} {'F1-Score':<20}")
    print("-" * 70)

    for rank, (model_name, results) in enumerate(sorted_models, 1):
        acc_str = f"{results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}"
        f1_str = f"{results['mean_f1']:.4f} ± {results['std_f1']:.4f}"
        print(
            f"{rank:<5} {model_name.replace('_', ' '):<25} {acc_str:<20} {f1_str:<20}"
        )

    # Best model details
    best_model_name, best_results = sorted_models[0]
    print(f"\n{'=' * 80}")
    print(f"🏆 BEST MODEL: {best_model_name.replace('_', ' ')}")
    print(f"{'=' * 80}")
    print(
        f"Mean Accuracy: {best_results['mean_accuracy']:.4f} ± {best_results['std_accuracy']:.4f}"
    )
    print(
        f"Mean F1-Score: {best_results['mean_f1']:.4f} ± {best_results['std_f1']:.4f}"
    )
    print(f"Mean Precision: {best_results['mean_precision']:.4f}")
    print(f"Mean Recall: {best_results['mean_recall']:.4f}")

    print("\nPer-class metrics:")
    class_report = best_results["classification_report"]
    print(
        f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}"
    )
    print("-" * 65)
    for cls in ["0", "1", "2"]:
        if cls in class_report:
            print(
                f"{'Class ' + cls + ' ':<15} "
                f"{class_report[cls]['precision']:<12.4f} "
                f"{class_report[cls]['recall']:<12.4f} "
                f"{class_report[cls]['f1-score']:<12.4f} "
                f"{class_report[cls]['support']:<10.0f}"
            )

    print(f"\n{'=' * 80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print("\nℹ️  INTERPRETATION:")
    print("  - Subject-level CV ensures truly independent evaluation")
    print(f"  - Best accuracy: {best_results['mean_accuracy']:.1%} (vs 33.3% random)")
    print(
        f"  - Improvement: {(best_results['mean_accuracy'] - 0.333) / 0.333 * 100:.1f}% over random"
    )

    if best_results["mean_accuracy"] < 0.65:
        print("\n⚠️  CONCLUSION: Limited performance suggests:")
        print("  1. Dataset features have weak discriminative power")
        print("  2. Stress is highly individual/contextual")
        print("  3. Need for feature engineering or temporal modeling")
        print("  4. Federated learning with personalization may help")
    else:
        print("\n✅ Good performance! Model has learned meaningful patterns.")


if __name__ == "__main__":
    main()
