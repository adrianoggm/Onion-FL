"""Visualize SWEET dataset class distribution to show imbalance."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flower_basic.datasets.sweet_samples import load_sweet_sample_dataset


def plot_class_distribution(data_dir: str, label_strategy: str = "ordinal"):
    """Plot class distribution across train/val/test splits."""

    print("=" * 80)
    print("SWEET Class Distribution Visualization")
    print("=" * 80)
    print(f"\nLoading dataset (label_strategy={label_strategy})...")

    # Load dataset
    dataset = load_sweet_sample_dataset(
        data_dir=data_dir,
        label_strategy=label_strategy,
        train_fraction=0.6,
        val_fraction=0.2,
    )

    # Get unique classes
    all_labels = np.concatenate([dataset.train.y, dataset.val.y, dataset.test.y])
    unique_classes = np.unique(all_labels.astype(int))

    # Count classes in each split
    train_unique, train_counts = np.unique(
        dataset.train.y.astype(int), return_counts=True
    )
    val_unique, val_counts = np.unique(dataset.val.y.astype(int), return_counts=True)
    test_unique, test_counts = np.unique(dataset.test.y.astype(int), return_counts=True)

    train_dict = dict(zip(train_unique, train_counts))
    val_dict = dict(zip(val_unique, val_counts))
    test_dict = dict(zip(test_unique, test_counts))

    # Prepare data for plotting
    train_data = [train_dict.get(cls, 0) for cls in unique_classes]
    val_data = [val_dict.get(cls, 0) for cls in unique_classes]
    test_data = [test_dict.get(cls, 0) for cls in unique_classes]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "SWEET Dataset - Class Distribution Analysis", fontsize=16, fontweight="bold"
    )

    # Define colors
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6"]

    # 1. Combined bar chart (all splits)
    ax1 = axes[0, 0]
    x = np.arange(len(unique_classes))
    width = 0.25

    bars1 = ax1.bar(
        x - width, train_data, width, label="Train", color="#3498db", alpha=0.8
    )
    bars2 = ax1.bar(x, val_data, width, label="Val", color="#2ecc71", alpha=0.8)
    bars3 = ax1.bar(
        x + width, test_data, width, label="Test", color="#e74c3c", alpha=0.8
    )

    ax1.set_xlabel("Stress Level", fontweight="bold")
    ax1.set_ylabel("Number of Samples", fontweight="bold")
    ax1.set_title("Distribution Across Splits")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Class {cls}" for cls in unique_classes])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # 2. Training set distribution (pie chart)
    ax2 = axes[0, 1]
    train_percentages = [count / sum(train_data) * 100 for count in train_data]
    wedges, texts, autotexts = ax2.pie(
        train_data,
        labels=[
            f"Class {cls}\n({train_data[i]} samples)"
            for i, cls in enumerate(unique_classes)
        ],
        autopct="%1.1f%%",
        colors=colors[: len(unique_classes)],
        startangle=90,
    )
    ax2.set_title("Training Set Distribution")
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    # 3. Log scale bar chart (to show small classes)
    ax3 = axes[1, 0]
    total_data = [
        train_data[i] + val_data[i] + test_data[i] for i in range(len(unique_classes))
    ]
    bars = ax3.bar(
        unique_classes, total_data, color=colors[: len(unique_classes)], alpha=0.8
    )
    ax3.set_xlabel("Stress Level", fontweight="bold")
    ax3.set_ylabel("Total Samples (log scale)", fontweight="bold")
    ax3.set_title("Total Distribution (Log Scale)")
    ax3.set_yscale("log")
    ax3.set_xticks(unique_classes)
    ax3.set_xticklabels([f"Class {cls}" for cls in unique_classes])
    ax3.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # 4. Statistics table
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Calculate statistics
    stats_data = []
    for i, cls in enumerate(unique_classes):
        total = total_data[i]
        train_pct = train_data[i] / sum(train_data) * 100 if sum(train_data) > 0 else 0
        stats_data.append(
            [
                f"Class {cls}",
                total,
                train_data[i],
                val_data[i],
                test_data[i],
                f"{train_pct:.1f}%",
            ]
        )

    # Add totals
    stats_data.append(
        [
            "TOTAL",
            sum(total_data),
            sum(train_data),
            sum(val_data),
            sum(test_data),
            "100.0%",
        ]
    )

    # Create table
    table = ax4.table(
        cellText=stats_data,
        colLabels=["Class", "Total", "Train", "Val", "Test", "Train %"],
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style total row
    for i in range(6):
        table[(len(stats_data), i)].set_facecolor("#ecf0f1")
        table[(len(stats_data), i)].set_text_props(weight="bold")

    # Alternate row colors
    for i in range(1, len(stats_data)):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f8f9fa")

    ax4.set_title("Class Distribution Statistics", fontweight="bold", pad=20)

    # Calculate and display imbalance ratio
    max_class = max(train_data)
    min_class = min([x for x in train_data if x > 0])
    imbalance_ratio = max_class / min_class

    fig.text(
        0.5,
        0.02,
        f"⚠️ Class Imbalance Ratio: {imbalance_ratio:.1f}:1 (Class {unique_classes[train_data.index(max_class)]} vs Class {unique_classes[train_data.index(min_class)]})",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#e74c3c",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save plot
    output_path = Path("swell_plots") / "sweet_class_distribution.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Plot saved to: {output_path}")

    # Show plot
    plt.show()

    print("\n" + "=" * 80)
    print("IMBALANCE SUMMARY:")
    print("=" * 80)
    print(
        f"Most common class: Class {unique_classes[train_data.index(max_class)]} ({max_class} samples)"
    )
    print(
        f"Least common class: Class {unique_classes[train_data.index(min_class)]} ({min_class} samples)"
    )
    print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
    print("\n⚠️ RECOMMENDATION: Merge classes 3, 4, 5 into single 'high stress' class")
    print("   This would create 3 balanced classes:")
    print("   - Class 0 (Low stress): original class 1")
    print("   - Class 1 (Medium stress): original class 2")
    print(
        f"   - Class 2 (High stress): original classes 3+4+5 = {train_data[2] + train_data[3] + train_data[4]} samples"
    )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SWEET dataset class distribution"
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
        default="ordinal",
        choices=["ordinal", "binary"],
        help="Label strategy to visualize",
    )

    args = parser.parse_args()

    try:
        plot_class_distribution(args.data_dir, args.label_strategy)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
