"""Comprehensive hyperparameter tuning for top SWEET models."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not available")

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flower_basic.datasets.sweet_samples import load_sweet_sample_full


def create_subject_level_cv_splits(data_dict, n_folds=5, seed=42):
    """Create subject-level CV splits."""
    rng = np.random.RandomState(seed)

    # Get unique subjects
    subjects = sorted(set(data_dict["subjects"]))
    rng.shuffle(subjects)

    # Get subject -> label mapping for stratification
    subject_labels = {}
    for subj, label in zip(data_dict["subjects"], data_dict["labels"]):
        if subj not in subject_labels:
            subject_labels[subj] = label

    # Stratified split by majority class per subject
    from collections import Counter

    subject_class_counts = {}
    for subj in subjects:
        subj_labels = [
            label
            for s, label in zip(data_dict["subjects"], data_dict["labels"])
            if s == subj
        ]
        subject_class_counts[subj] = Counter(subj_labels).most_common(1)[0][0]

    # Group subjects by class
    class_subjects = {}
    for subj, cls in subject_class_counts.items():
        if cls not in class_subjects:
            class_subjects[cls] = []
        class_subjects[cls].append(subj)

    # Create folds
    folds = [[] for _ in range(n_folds)]
    for cls_subjects in class_subjects.values():
        rng.shuffle(cls_subjects)
        for i, subj in enumerate(cls_subjects):
            folds[i % n_folds].append(subj)

    # Convert to train/test indices
    cv_splits = []
    for fold_idx in range(n_folds):
        test_subjects = set(folds[fold_idx])
        train_indices = [
            i for i, s in enumerate(data_dict["subjects"]) if s not in test_subjects
        ]
        test_indices = [
            i for i, s in enumerate(data_dict["subjects"]) if s in test_subjects
        ]
        cv_splits.append((train_indices, test_indices))

    return cv_splits


def tune_gradient_boosting(X, y, cv_splits):
    """Tune Gradient Boosting with extensive grid search."""
    print("\n" + "=" * 80)
    print("🔧 TUNING: Gradient Boosting")
    print("=" * 80)

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6, 7],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "subsample": [0.8, 0.9, 1.0],
        "max_features": ["sqrt", "log2", None],
    }

    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()]):,}")
    print("Using RandomizedSearchCV (200 iterations)...")

    base_model = GradientBoostingClassifier(random_state=42)

    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=200,
        cv=cv_splits,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    print(f"\n✓ Tuning completed in {elapsed/60:.1f} minutes")
    print(f"Best accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return {
        "model_name": "Gradient_Boosting_Tuned",
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_estimator": search.best_estimator_,
        "cv_results": search.cv_results_,
        "tuning_time": elapsed,
    }


def tune_random_forest(X, y, cv_splits):
    """Tune Random Forest."""
    print("\n" + "=" * 80)
    print("🔧 TUNING: Random Forest")
    print("=" * 80)

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 20, 30, 40, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()]):,}")
    print("Using RandomizedSearchCV (150 iterations)...")

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=150,
        cv=cv_splits,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    print(f"\n✓ Tuning completed in {elapsed/60:.1f} minutes")
    print(f"Best accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return {
        "model_name": "Random_Forest_Tuned",
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_estimator": search.best_estimator_,
        "cv_results": search.cv_results_,
        "tuning_time": elapsed,
    }


def tune_xgboost(X, y, cv_splits):
    """Tune XGBoost."""
    if not HAS_XGBOOST:
        return None

    print("\n" + "=" * 80)
    print("🔧 TUNING: XGBoost")
    print("=" * 80)

    param_grid = {
        "n_estimators": [100, 200, 300, 500, 700],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [1, 1.5, 2, 3],
    }

    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()]):,}")
    print("Using RandomizedSearchCV (200 iterations)...")

    base_model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=200,
        cv=cv_splits,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    print(f"\n✓ Tuning completed in {elapsed/60:.1f} minutes")
    print(f"Best accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return {
        "model_name": "XGBoost_Tuned",
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_estimator": search.best_estimator_,
        "cv_results": search.cv_results_,
        "tuning_time": elapsed,
    }


def tune_knn(X, y, cv_splits):
    """Tune KNN."""
    print("\n" + "=" * 80)
    print("🔧 TUNING: K-Nearest Neighbors")
    print("=" * 80)

    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31, 51],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2, 3],  # for minkowski
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    }

    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()]):,}")
    print("Using GridSearchCV (full search)...")

    base_model = KNeighborsClassifier(n_jobs=-1)

    search = GridSearchCV(
        base_model, param_grid, cv=cv_splits, scoring="accuracy", n_jobs=-1, verbose=2
    )

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    print(f"\n✓ Tuning completed in {elapsed/60:.1f} minutes")
    print(f"Best accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return {
        "model_name": "KNN_Tuned",
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_estimator": search.best_estimator_,
        "cv_results": search.cv_results_,
        "tuning_time": elapsed,
    }


def tune_mlp(X, y, cv_splits):
    """Tune MLP."""
    print("\n" + "=" * 80)
    print("🔧 TUNING: Multi-Layer Perceptron")
    print("=" * 80)

    param_grid = {
        "hidden_layer_sizes": [
            (64,),
            (128,),
            (256,),
            (64, 64),
            (128, 64),
            (256, 128),
            (128, 128, 64),
            (256, 128, 64),
        ],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["constant", "adaptive"],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "max_iter": [500, 1000],
        "early_stopping": [True],
        "validation_fraction": [0.1],
    }

    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()]):,}")
    print("Using RandomizedSearchCV (100 iterations)...")

    base_model = MLPClassifier(random_state=42)

    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=100,
        cv=cv_splits,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    print(f"\n✓ Tuning completed in {elapsed/60:.1f} minutes")
    print(f"Best accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return {
        "model_name": "MLP_Tuned",
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_estimator": search.best_estimator_,
        "cv_results": search.cv_results_,
        "tuning_time": elapsed,
    }


def tune_svm(X, y, cv_splits):
    """Tune SVM (limited search due to computational cost)."""
    print("\n" + "=" * 80)
    print("🔧 TUNING: Support Vector Machine")
    print("=" * 80)

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "kernel": ["rbf", "poly", "sigmoid"],
        "degree": [2, 3, 4],  # for poly
    }

    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()]):,}")
    print("Using RandomizedSearchCV (50 iterations)...")

    base_model = SVC(random_state=42)

    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=50,
        cv=cv_splits,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    print(f"\n✓ Tuning completed in {elapsed/60:.1f} minutes")
    print(f"Best accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return {
        "model_name": "SVM_Tuned",
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_estimator": search.best_estimator_,
        "cv_results": search.cv_results_,
        "tuning_time": elapsed,
    }


def evaluate_on_test(model, X_train, y_train, X_test, y_test, scaler=None):
    """Evaluate best model on test set."""
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "test_accuracy": float(acc),
        "test_f1": float(f1),
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for SWEET models"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to SWEET data"
    )
    parser.add_argument(
        "--label-strategy",
        type=str,
        default="ordinal_3class",
        choices=["ordinal", "ordinal_3class", "binary"],
        help="Label strategy",
    )
    parser.add_argument(
        "--output-dir", type=str, default="hypertuning_results", help="Output directory"
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gb", "xgb", "rf", "knn", "mlp"],
        help="Models to tune: gb, xgb, rf, knn, mlp, svm",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("COMPREHENSIVE HYPERPARAMETER TUNING")
    print("=" * 80)
    print(f"Models: {', '.join(args.models)}")
    print(f"CV Folds: {args.n_folds}")
    print(f"Label Strategy: {args.label_strategy}")

    # Load data
    print("\nLoading SWEET dataset...")
    X_train_full, y_train_full, subjects_full, feature_names = load_sweet_sample_full(
        data_dir=Path(args.data_dir), label_strategy=args.label_strategy
    )

    print(
        f"✓ Dataset loaded: {len(X_train_full)} samples, {len(set(subjects_full))} subjects"
    )

    # Create data dict for CV splits
    data_dict = {
        "features": X_train_full,
        "labels": y_train_full,
        "subjects": subjects_full,
    }

    # Create CV splits
    print("\nCreating subject-level CV splits...")
    cv_splits = create_subject_level_cv_splits(data_dict, args.n_folds, args.seed)
    print(f"✓ Created {args.n_folds} folds (subject-level stratified)")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_full)

    # Tune models
    all_results = {}
    tuning_functions = {
        "gb": tune_gradient_boosting,
        "rf": tune_random_forest,
        "xgb": tune_xgboost,
        "knn": tune_knn,
        "mlp": tune_mlp,
        "svm": tune_svm,
    }

    total_start = time.time()

    for model_key in args.models:
        if model_key not in tuning_functions:
            print(f"⚠️  Unknown model: {model_key}")
            continue

        if model_key == "xgb" and not HAS_XGBOOST:
            print("⚠️  Skipping XGBoost (not installed)")
            continue

        result = tuning_functions[model_key](X_scaled, y_train_full, cv_splits)

        if result is not None:
            all_results[result["model_name"]] = result

            # Save intermediate results
            output_file = output_dir / f"{result['model_name']}_tuning.json"
            with open(output_file, "w") as f:
                # Convert non-serializable objects
                save_result = {
                    "model_name": result["model_name"],
                    "best_params": result["best_params"],
                    "best_score": result["best_score"],
                    "tuning_time": result["tuning_time"],
                }
                json.dump(save_result, f, indent=2)
            print(f"✓ Saved results to {output_file}")

    total_elapsed = time.time() - total_start

    # Print summary
    print("\n" + "=" * 80)
    print("TUNING SUMMARY")
    print("=" * 80)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print("\nBest scores (CV):")

    sorted_results = sorted(
        all_results.items(), key=lambda x: x[1]["best_score"], reverse=True
    )

    for rank, (name, result) in enumerate(sorted_results, 1):
        print(
            f"{rank}. {name}: {result['best_score']:.4f} ({result['tuning_time']/60:.1f} min)"
        )

    # Compare with baseline
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE (untuned)")
    print("=" * 80)

    baseline_scores = {
        "Gradient_Boosting": 0.5086,
        "XGBoost": 0.4979,
        "Random_Forest": 0.4962,
        "KNN": 0.5069,
        "MLP": 0.4440,
    }

    for name, result in sorted_results:
        base_name = name.replace("_Tuned", "")
        if base_name in baseline_scores:
            baseline = baseline_scores[base_name]
            improvement = (result["best_score"] - baseline) * 100
            symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(
                f"{symbol} {base_name}: {baseline:.4f} → {result['best_score']:.4f} ({improvement:+.2f}%)"
            )

    # Save final results
    final_output = output_dir / "hypertuning_summary.json"
    summary = {
        "total_time_minutes": total_elapsed / 60,
        "models_tuned": list(all_results.keys()),
        "best_model": sorted_results[0][0] if sorted_results else None,
        "best_score": sorted_results[0][1]["best_score"] if sorted_results else None,
        "results": {
            name: {
                "best_score": res["best_score"],
                "best_params": res["best_params"],
                "tuning_time_minutes": res["tuning_time"] / 60,
            }
            for name, res in all_results.items()
        },
    }

    with open(final_output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Final summary saved to {final_output}")
    print("\n🎯 CONCLUSION:")
    if sorted_results:
        best_name, best_result = sorted_results[0]
        print(f"Best tuned model: {best_name}")
        print(f"Best CV accuracy: {best_result['best_score']:.4f}")
        print(f"Best params: {best_result['best_params']}")


if __name__ == "__main__":
    main()
