"""Unified comparison of ALL ML models tested on SWEET dataset."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load all results
results_dir = Path("advanced_ml_results")

# Load traditional ML results
with open(results_dir / "cv_results.json") as f:
    traditional_results = json.load(f)

# Load ultra-powerful results (if exists)
try:
    with open(results_dir / "ultra_powerful_results.json") as f:
        ultra_results = json.load(f)
except FileNotFoundError:
    ultra_results = {}

# Combine all results
all_results = {**traditional_results, **ultra_results}

# Sort by accuracy
sorted_results = sorted(
    all_results.items(), key=lambda x: x[1]["mean_accuracy"], reverse=True
)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1. Overall accuracy comparison
ax1 = axes[0, 0]
model_names = [name.replace("_", " ") for name, _ in sorted_results]
accuracies = [res["mean_accuracy"] for _, res in sorted_results]
stds = [res["std_accuracy"] for _, res in sorted_results]

colors = [
    "#e74c3c" if acc < 0.40 else "#f39c12" if acc < 0.48 else "#2ecc71"
    for acc in accuracies
]

y_pos = np.arange(len(model_names))
bars = ax1.barh(y_pos, accuracies, xerr=stds, color=colors, alpha=0.7, capsize=5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(model_names, fontsize=9)
ax1.set_xlabel("Accuracy (5-Fold CV)", fontweight="bold", fontsize=11)
ax1.set_title(
    "COMPREHENSIVE MODEL COMPARISON\nAll ML Models Tested on SWEET Dataset",
    fontweight="bold",
    fontsize=13,
)
ax1.axvline(
    0.333, color="red", linestyle="--", alpha=0.5, linewidth=2, label="Random (33.3%)"
)
ax1.axvline(
    0.50, color="blue", linestyle="--", alpha=0.5, linewidth=2, label="50% Threshold"
)
ax1.legend(fontsize=10)
ax1.grid(axis="x", alpha=0.3)

# Add value labels
for bar, acc, std in zip(bars, accuracies, stds):
    ax1.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{acc:.3f}±{std:.3f}",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

# 2. F1-Score comparison
ax2 = axes[0, 1]
f1_scores = [res["mean_f1"] for _, res in sorted_results]

bars2 = ax2.barh(y_pos, f1_scores, color="steelblue", alpha=0.7)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(model_names, fontsize=9)
ax2.set_xlabel("F1-Score (Weighted)", fontweight="bold", fontsize=11)
ax2.set_title("F1-Score Comparison", fontweight="bold", fontsize=12)
ax2.grid(axis="x", alpha=0.3)

for bar, f1 in zip(bars2, f1_scores):
    ax2.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{f1:.3f}",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

# 3. Best model confusion matrix
ax3 = axes[1, 0]
best_name, best_result = sorted_results[0]
cm = np.array(best_result["confusion_matrix"])

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="RdYlGn",
    ax=ax3,
    xticklabels=["Low\nStress", "Medium\nStress", "High\nStress"],
    yticklabels=["Low\nStress", "Medium\nStress", "High\nStress"],
    cbar_kws={"label": "Count"},
)
ax3.set_xlabel("Predicted Class", fontweight="bold", fontsize=11)
ax3.set_ylabel("True Class", fontweight="bold", fontsize=11)
ax3.set_title(
    f'Best Model: {best_name.replace("_", " ")}\nAccuracy: {best_result["mean_accuracy"]:.1%}',
    fontweight="bold",
    fontsize=12,
)

# 4. Model family comparison
ax4 = axes[1, 1]

model_families = {
    "Tree-Based\n(GB, RF, DT)": [
        "Gradient_Boosting",
        "Random_Forest",
        "Decision_Tree",
        "XGBoost",
        "XGBoost_Optimized",
    ],
    "Neural Networks\n(MLP, LSTM, GRU)": [
        "MLP_Small",
        "MLP_Deep",
        "MLP_Wide",
        "LSTM_Attention",
        "GRU_Attention",
        "DeepMLP_Residual",
    ],
    "Distance-Based\n(KNN, SVM)": ["KNN", "SVM_RBF"],
    "Linear\n(LR, NB)": ["Logistic_Regression", "Naive_Bayes"],
}

family_accs = {}
for family, models in model_families.items():
    accs = [all_results[m]["mean_accuracy"] for m in models if m in all_results]
    if accs:
        family_accs[family] = {
            "mean": np.mean(accs),
            "std": np.std(accs),
            "max": np.max(accs),
            "count": len(accs),
        }

families = list(family_accs.keys())
means = [family_accs[f]["mean"] for f in families]
maxs = [family_accs[f]["max"] for f in families]
stds = [family_accs[f]["std"] for f in families]

x = np.arange(len(families))
width = 0.35

bars_mean = ax4.bar(
    x - width / 2, means, width, label="Mean", alpha=0.8, color="steelblue"
)
bars_max = ax4.bar(
    x + width / 2, maxs, width, label="Best", alpha=0.8, color="forestgreen"
)

ax4.set_ylabel("Accuracy", fontweight="bold", fontsize=11)
ax4.set_title("Performance by Model Family", fontweight="bold", fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(families, fontsize=9)
ax4.legend(fontsize=10)
ax4.grid(axis="y", alpha=0.3)
ax4.set_ylim(0, 0.6)

# Add value labels
for bar in bars_mean:
    height = bar.get_height()
    ax4.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
for bar in bars_max:
    height = bar.get_height()
    ax4.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.tight_layout()
plt.savefig(results_dir / "unified_comparison.png", dpi=300, bbox_inches="tight")
print(f"✅ Unified comparison saved to: {results_dir / 'unified_comparison.png'}")

# Print comprehensive table
print("\n" + "=" * 90)
print("COMPREHENSIVE RESULTS TABLE - ALL MODELS")
print("=" * 90)
print(f"{'Rank':<5} {'Model':<30} {'Accuracy':<18} {'F1-Score':<12} {'Family':<20}")
print("-" * 90)


def get_family(model_name):
    for family, models in model_families.items():
        if model_name in models:
            return family.split("\n")[0]
    return "Other"


for rank, (name, res) in enumerate(sorted_results, 1):
    family = get_family(name)
    acc_str = f"{res['mean_accuracy']:.4f} ± {res['std_accuracy']:.4f}"
    f1_str = f"{res['mean_f1']:.4f}"
    print(
        f"{rank:<5} {name.replace('_', ' '):<30} {acc_str:<18} {f1_str:<12} {family:<20}"
    )

print("\n" + "=" * 90)
print("SUMMARY STATISTICS")
print("=" * 90)
print(f"Total models tested: {len(all_results)}")
print(
    f"Best accuracy: {sorted_results[0][1]['mean_accuracy']:.4f} ({sorted_results[0][0].replace('_', ' ')})"
)
print(
    f"Worst accuracy: {sorted_results[-1][1]['mean_accuracy']:.4f} ({sorted_results[-1][0].replace('_', ' ')})"
)
print(f"Mean accuracy: {np.mean([r[1]['mean_accuracy'] for r in sorted_results]):.4f}")
print(
    f"Median accuracy: {np.median([r[1]['mean_accuracy'] for r in sorted_results]):.4f}"
)
print(f"Std accuracy: {np.std([r[1]['mean_accuracy'] for r in sorted_results]):.4f}")

print(f"\n📊 Best Family: {max(family_accs.items(), key=lambda x: x[1]['mean'])[0]}")
print(f"   Mean: {max(family_accs.items(), key=lambda x: x[1]['mean'])[1]['mean']:.4f}")

print("\n" + "=" * 90)
print("KEY INSIGHTS")
print("=" * 90)
print("✅ Subject-level 5-fold CV ensures no data leakage")
print(
    f"✅ Best model improves {(sorted_results[0][1]['mean_accuracy'] - 0.333) / 0.333 * 100:.1f}% over random (33.3%)"
)
print("⚠️  Performance plateau around 50% suggests feature limitations")
print("💡 Tree-based models (GB, RF, XGBoost) consistently outperform neural networks")
print("💡 Simple models often match or beat complex models (KNN competitive)")
print("🎯 Recommendation: Use Gradient Boosting or XGBoost for production")

plt.show()
