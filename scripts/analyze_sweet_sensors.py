"""Detailed analysis of SWEET dataset sensors and features."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flower_basic.datasets.sweet_samples import load_sweet_sample_dataset


def analyze_sensor_data(data_dir: str, label_strategy: str = "ordinal_3class"):
    """Comprehensive analysis of SWEET sensor data and features."""
    
    print("=" * 80)
    print("SWEET DATASET - DETAILED SENSOR & FEATURE ANALYSIS")
    print("=" * 80)
    print(f"\nLoading dataset (label_strategy={label_strategy})...")
    
    # Load dataset
    dataset = load_sweet_sample_dataset(
        data_dir=data_dir,
        label_strategy=label_strategy,
        train_fraction=0.6,
        val_fraction=0.2,
    )
    
    # Combine all data for analysis
    all_X = np.vstack([dataset.train.X, dataset.val.X, dataset.test.X])
    all_y = np.concatenate([dataset.train.y, dataset.val.y, dataset.test.y])
    
    # Get subject IDs from partition.subject_ids attribute
    all_subjects = np.concatenate([
        dataset.train.subject_ids,
        dataset.val.subject_ids,
        dataset.test.subject_ids
    ])
    
    feature_names = dataset.feature_names
    
    print(f"\n{'=' * 80}")
    print("1. DATASET OVERVIEW")
    print(f"{'=' * 80}")
    print(f"Total samples: {len(all_y)}")
    print(f"Total subjects: {len(np.unique(all_subjects))}")
    print(f"Features: {len(feature_names)}")
    print(f"\nClass distribution (post-mapping to {label_strategy}):")
    unique, counts = np.unique(all_y, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = count / len(all_y) * 100
        if label_strategy == "ordinal_3class":
            label = ["Low stress", "Medium stress", "High stress"][int(cls)]
            print(f"  Class {int(cls)} ({label}): {count:4d} samples ({pct:5.1f}%)")
        else:
            print(f"  Class {int(cls)}: {count:4d} samples ({pct:5.1f}%)")
    
    # Analyze features by sensor type
    print(f"\n{'=' * 80}")
    print("2. SENSOR FEATURE BREAKDOWN")
    print(f"{'=' * 80}")
    
    # Categorize features
    ecg_features = [f for f in feature_names if 'ECG' in f or 'heart' in f.lower() or 'hrv' in f.lower() 
                    or 'sdnn' in f.lower() or 'rmssd' in f.lower() or 'LF' in f or 'HF' in f]
    acc_features = [f for f in feature_names if 'ACC' in f or 'accel' in f.lower() or 'std' in f.lower()]
    eda_features = [f for f in feature_names if 'EDA' in f or 'gsr' in f.lower() or 'skin' in f.lower()]
    temp_features = [f for f in feature_names if 'temp' in f.lower() or 'TEMP' in f]
    other_features = [f for f in feature_names if f not in ecg_features + acc_features + eda_features + temp_features]
    
    print(f"\n📊 ECG/Heart Rate Features ({len(ecg_features)}):")
    for feat in ecg_features:
        idx = feature_names.index(feat)
        vals = all_X[:, idx]
        print(f"  - {feat:30s}: mean={np.mean(vals):10.2f}, std={np.std(vals):10.2f}, "
              f"min={np.min(vals):10.2f}, max={np.max(vals):10.2f}")
    
    print(f"\n📱 Accelerometer Features ({len(acc_features)}):")
    for feat in acc_features:
        idx = feature_names.index(feat)
        vals = all_X[:, idx]
        print(f"  - {feat:30s}: mean={np.mean(vals):10.2f}, std={np.std(vals):10.2f}, "
              f"min={np.min(vals):10.2f}, max={np.max(vals):10.2f}")
    
    if eda_features:
        print(f"\n💧 EDA/GSR Features ({len(eda_features)}):")
        for feat in eda_features:
            idx = feature_names.index(feat)
            vals = all_X[:, idx]
            print(f"  - {feat:30s}: mean={np.mean(vals):10.2f}, std={np.std(vals):10.2f}, "
                  f"min={np.min(vals):10.2f}, max={np.max(vals):10.2f}")
    
    if temp_features:
        print(f"\n🌡️  Temperature Features ({len(temp_features)}):")
        for feat in temp_features:
            idx = feature_names.index(feat)
            vals = all_X[:, idx]
            print(f"  - {feat:30s}: mean={np.mean(vals):10.2f}, std={np.std(vals):10.2f}, "
                  f"min={np.min(vals):10.2f}, max={np.max(vals):10.2f}")
    
    if other_features:
        print(f"\n🔧 Other Features ({len(other_features)}):")
        for feat in other_features:
            idx = feature_names.index(feat)
            vals = all_X[:, idx]
            print(f"  - {feat:30s}: mean={np.mean(vals):10.2f}, std={np.std(vals):10.2f}, "
                  f"min={np.min(vals):10.2f}, max={np.max(vals):10.2f}")
    
    # Per-subject analysis
    print(f"\n{'=' * 80}")
    print("3. PER-SUBJECT STATISTICS")
    print(f"{'=' * 80}")
    
    unique_subjects = np.unique(all_subjects)
    subject_stats = []
    
    for subject in unique_subjects[:10]:  # Show first 10 subjects
        subject_mask = all_subjects == subject
        subject_samples = np.sum(subject_mask)
        subject_classes = np.unique(all_y[subject_mask], return_counts=True)
        
        subject_stats.append({
            'subject': subject,
            'samples': subject_samples,
            'classes': dict(zip(subject_classes[0].astype(int), subject_classes[1]))
        })
    
    print(f"\nShowing first 10 subjects (out of {len(unique_subjects)}):")
    print(f"{'Subject':<12} {'Samples':>8} {'Class Distribution':>30}")
    print("-" * 52)
    for stat in subject_stats:
        classes_str = ", ".join([f"C{k}:{v}" for k, v in stat['classes'].items()])
        print(f"{stat['subject']:<12} {stat['samples']:>8} {classes_str:>30}")
    
    # Overall subject statistics
    samples_per_subject = [np.sum(all_subjects == subj) for subj in unique_subjects]
    print(f"\n📊 Samples per subject statistics:")
    print(f"  Mean: {np.mean(samples_per_subject):.1f}")
    print(f"  Median: {np.median(samples_per_subject):.1f}")
    print(f"  Std: {np.std(samples_per_subject):.1f}")
    print(f"  Min: {np.min(samples_per_subject)}")
    print(f"  Max: {np.max(samples_per_subject)}")
    
    # Feature correlation analysis
    print(f"\n{'=' * 80}")
    print("4. FEATURE CORRELATIONS WITH STRESS LEVEL")
    print(f"{'=' * 80}")
    
    correlations = []
    for i, feat in enumerate(feature_names):
        corr = np.corrcoef(all_X[:, i], all_y)[0, 1]
        correlations.append((feat, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop 10 features most correlated with stress level:")
    for i, (feat, corr) in enumerate(correlations[:10], 1):
        direction = "↑ positive" if corr > 0 else "↓ negative"
        print(f"  {i:2d}. {feat:30s}: {corr:+.4f} ({direction})")
    
    # Create visualizations
    print(f"\n{'=' * 80}")
    print("5. GENERATING VISUALIZATIONS")
    print(f"{'=' * 80}")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Samples per subject histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(samples_per_subject, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Samples per Subject')
    ax1.set_ylabel('Number of Subjects')
    ax1.set_title('Distribution of Samples per Subject')
    ax1.axvline(np.mean(samples_per_subject), color='red', linestyle='--', label=f'Mean: {np.mean(samples_per_subject):.1f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Class distribution by split
    ax2 = fig.add_subplot(gs[0, 1])
    splits = ['Train', 'Val', 'Test']
    split_data = [dataset.train.y, dataset.val.y, dataset.test.y]
    split_colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    x_pos = np.arange(len(unique))
    width = 0.25
    
    for i, (split_name, split_y, color) in enumerate(zip(splits, split_data, split_colors)):
        unique_split, counts_split = np.unique(split_y, return_counts=True)
        counts_aligned = [np.sum(split_y == cls) for cls in unique]
        ax2.bar(x_pos + i*width, counts_aligned, width, label=split_name, alpha=0.8, color=color)
    
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Class Distribution by Split')
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels([f'C{int(c)}' for c in unique])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Feature correlation heatmap (top features)
    ax3 = fig.add_subplot(gs[0, 2])
    top_features_idx = [feature_names.index(feat) for feat, _ in correlations[:8]]
    top_features_names = [correlations[i][0] for i in range(8)]
    corr_matrix = np.corrcoef(all_X[:, top_features_idx].T)
    
    im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax3.set_xticks(range(len(top_features_names)))
    ax3.set_yticks(range(len(top_features_names)))
    ax3.set_xticklabels([f[:15] for f in top_features_names], rotation=45, ha='right', fontsize=8)
    ax3.set_yticklabels([f[:15] for f in top_features_names], fontsize=8)
    ax3.set_title('Feature Correlation (Top 8)')
    plt.colorbar(im, ax=ax3)
    
    # 4-6. Feature distributions by class (for top 3 correlated features)
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        feat_name, _ = correlations[i]
        feat_idx = feature_names.index(feat_name)
        
        for cls in unique:
            class_data = all_X[all_y == cls, feat_idx]
            ax.hist(class_data, bins=30, alpha=0.5, label=f'Class {int(cls)}', density=True)
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{feat_name[:25]}...' if len(feat_name) > 25 else feat_name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # 7. ECG features boxplot by class
    if len(ecg_features) > 0:
        ax7 = fig.add_subplot(gs[2, 0])
        ecg_data = []
        ecg_labels = []
        ecg_classes = []
        
        for cls in unique:
            for ecg_feat in ecg_features[:4]:  # Top 4 ECG features
                feat_idx = feature_names.index(ecg_feat)
                ecg_data.extend(all_X[all_y == cls, feat_idx])
                ecg_labels.extend([ecg_feat[:15]] * np.sum(all_y == cls))
                ecg_classes.extend([f'C{int(cls)}'] * np.sum(all_y == cls))
        
        df_ecg = pd.DataFrame({'Feature': ecg_labels, 'Class': ecg_classes, 'Value': ecg_data})
        
        # Use seaborn for better boxplot
        sns.boxplot(data=df_ecg, x='Feature', y='Value', hue='Class', ax=ax7)
        ax7.set_title('ECG Features by Class')
        ax7.set_xlabel('')
        ax7.set_ylabel('Normalized Value')
        ax7.tick_params(axis='x', rotation=45)
        ax7.legend(fontsize=8)
    
    # 8. ACC features boxplot by class
    if len(acc_features) > 0:
        ax8 = fig.add_subplot(gs[2, 1])
        acc_data = []
        acc_labels = []
        acc_classes = []
        
        for cls in unique:
            for acc_feat in acc_features[:4]:  # Top 4 ACC features
                feat_idx = feature_names.index(acc_feat)
                acc_data.extend(all_X[all_y == cls, feat_idx])
                acc_labels.extend([acc_feat[:15]] * np.sum(all_y == cls))
                acc_classes.extend([f'C{int(cls)}'] * np.sum(all_y == cls))
        
        df_acc = pd.DataFrame({'Feature': acc_labels, 'Class': acc_classes, 'Value': acc_data})
        
        sns.boxplot(data=df_acc, x='Feature', y='Value', hue='Class', ax=ax8)
        ax8.set_title('Accelerometer Features by Class')
        ax8.set_xlabel('')
        ax8.set_ylabel('Normalized Value')
        ax8.tick_params(axis='x', rotation=45)
        ax8.legend(fontsize=8)
    
    # 9. Samples and classes per subject
    ax9 = fig.add_subplot(gs[2, 2])
    subject_class_diversity = []
    for subject in unique_subjects:
        n_classes = len(np.unique(all_y[all_subjects == subject]))
        subject_class_diversity.append(n_classes)
    
    diversity_counts = np.bincount(subject_class_diversity)
    ax9.bar(range(len(diversity_counts)), diversity_counts, color='mediumpurple', alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Number of Different Classes per Subject')
    ax9.set_ylabel('Number of Subjects')
    ax9.set_title('Class Diversity per Subject')
    ax9.set_xticks(range(len(diversity_counts)))
    ax9.grid(axis='y', alpha=0.3)
    
    plt.suptitle('SWEET Dataset - Comprehensive Sensor & Feature Analysis', fontsize=16, fontweight='bold')
    
    # Save plot
    output_path = Path('swell_plots') / 'sweet_sensor_analysis.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    plt.show()
    
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Detailed analysis of SWEET dataset sensors and features"
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
        help="Label strategy to analyze",
    )
    
    args = parser.parse_args()
    
    try:
        analyze_sensor_data(args.data_dir, args.label_strategy)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
