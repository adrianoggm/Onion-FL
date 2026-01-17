#!/usr/bin/env python3
"""
SWELL Dataset Comprehensive Visualization
==========================================
Creates professional visualizations of the SWELL stress detection dataset
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Configure matplotlib for better visuals
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def load_swell_data():
    """Load and prepare SWELL dataset."""
    print("📂 Loading SWELL dataset...")
    
    # Use the actual SWELL dataset structure
    train_file = "data/SWELL/data/final/train.csv"
    test_file = "data/SWELL/data/final/test.csv"
    
    # Load data
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Add dataset split indicator
    df_train['split'] = 'Train'
    df_test['split'] = 'Test'
    
    # Combine train and test for comprehensive visualization
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"   ✓ Combined dataset shape: {df.shape}")
    print(f"   ✓ Train samples: {len(df_train)}")
    print(f"   ✓ Test samples: {len(df_test)}")
    
    # Get feature columns (exclude 'condition', 'split', 'datasetId')
    features_cols = [col for col in df.columns if col not in ['condition', 'split', 'datasetId']]
    
    print(f"   ✓ Number of features: {len(features_cols)}")
    print(f"   ✓ Conditions: {df['condition'].unique()}")
    
    # Create binary stress label (stress = time pressure or interruption, no stress = no stress)
    df['label'] = df['condition'].apply(lambda x: 0 if x == 'no stress' else 1)
    df['Stress_Label'] = df['label'].map({0: 'No Stress', 1: 'Stress'})
    
    return df, features_cols


def create_comprehensive_visualization():
    """Create comprehensive SWELL dataset visualization"""
    
    print("🎨 Creating SWELL Dataset Comprehensive Visualization")
    print("=" * 60)
    
    # Load data
    df, features_cols = load_swell_data()
    
    print(f"✓ Data loaded: {df.shape[0]} samples, {len(features_cols)} features")
    print(f"✓ Stress samples: {df['label'].sum()}")
    print(f"✓ No stress samples: {(df['label'] == 0).sum()}")
    print()
    
    # Create output directory
    output_dir = Path("swell_plots")
    output_dir.mkdir(exist_ok=True)
    
    # ==================== VISUALIZATION 1: Overview ====================
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('SWELL Dataset - Comprehensive Overview', fontsize=16, fontweight='bold')
    
    # 1. Class Distribution
    ax1 = plt.subplot(3, 4, 1)
    stress_counts = df['Stress_Label'].value_counts()
    colors = ['#3498db', '#e74c3c']
    stress_counts.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Distribution by Stress', fontweight='bold')
    ax1.set_xlabel('Stress Level')
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=0)
    for i, v in enumerate(stress_counts):
        ax1.text(i, v + 500, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. Stress vs No Stress (Pie)
    ax2 = plt.subplot(3, 4, 2)
    colors_stress = ['#3498db', '#e74c3c']
    wedges, texts, autotexts = ax2.pie(stress_counts, labels=stress_counts.index, 
                                         autopct='%1.1f%%', colors=colors_stress,
                                         startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Binary Stress Classification', fontweight='bold')
    
    # 3. Train/Test Split Distribution
    ax3 = plt.subplot(3, 4, 3)
    split_counts = df['split'].value_counts()
    split_counts.plot(kind='bar', ax=ax3, color=['#2ecc71', '#f39c12'])
    ax3.set_title('Train/Test Split', fontweight='bold')
    ax3.set_xlabel('Dataset Split')
    ax3.set_ylabel('Number of Samples')
    ax3.tick_params(axis='x', rotation=0)
    for i, v in enumerate(split_counts):
        ax3.text(i, v + 500, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 4. Stress Distribution by Split
    ax4 = plt.subplot(3, 4, 4)
    split_stress = df.groupby(['split', 'Stress_Label']).size().unstack(fill_value=0)
    split_stress.plot(kind='bar', stacked=False, ax=ax4, color=colors_stress)
    ax4.set_title('Stress Distribution by Split', fontweight='bold')
    ax4.set_xlabel('Dataset Split')
    ax4.set_ylabel('Number of Samples')
    ax4.tick_params(axis='x', rotation=0)
    ax4.legend(title='Stress', fontsize=8)
    
    # 5-8. Feature Distributions (Box plots for top features)
    # Select top features with highest variance
    feature_variances = df[features_cols].var().sort_values(ascending=False)
    key_features = feature_variances.head(4).index.tolist()
    
    for idx, feature in enumerate(key_features, start=5):
        ax = plt.subplot(3, 4, idx)
        data_to_plot = [df[df['Stress_Label'] == 'No Stress'][feature],
                       df[df['Stress_Label'] == 'Stress'][feature]]
        bp = ax.boxplot(data_to_plot, labels=['No Stress', 'Stress'], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_stress):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(f'{feature} by Stress', fontweight='bold', fontsize=9)
        ax.set_ylabel(feature, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    
    # 9. Correlation Heatmap (selected features)
    ax9 = plt.subplot(3, 4, 9)
    # Select top 6 features by variance
    selected_features = feature_variances.head(6).index.tolist() + ['label']
    corr_matrix = df[selected_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax9, square=True, cbar_kws={'shrink': 0.8}, 
                annot_kws={'fontsize': 7})
    ax9.set_title('Feature Correlation Matrix', fontweight='bold')
    ax9.tick_params(labelsize=7)
    
    # 10. Feature 1 vs Feature 2 (Scatter)
    ax10 = plt.subplot(3, 4, 10)
    feat1, feat2 = key_features[0], key_features[1]
    for stress_label, color in zip(['No Stress', 'Stress'], colors_stress):
        mask = df['Stress_Label'] == stress_label
        ax10.scatter(df.loc[mask, feat1], df.loc[mask, feat2],
                    alpha=0.3, s=10, label=stress_label, color=color)
    ax10.set_title(f'{feat1} vs {feat2}', fontweight='bold', fontsize=9)
    ax10.set_xlabel(feat1, fontsize=8)
    ax10.set_ylabel(feat2, fontsize=8)
    ax10.legend(fontsize=7)
    ax10.grid(True, alpha=0.3)
    ax10.tick_params(labelsize=8)
    
    # 11. Feature Importance (based on variance by stress)
    ax11 = plt.subplot(3, 4, 11)
    stress_diff = []
    top_features = feature_variances.head(10).index.tolist()
    for col in top_features:
        stress_mean = df[df['label'] == 1][col].mean()
        no_stress_mean = df[df['label'] == 0][col].mean()
        stress_diff.append(abs(stress_mean - no_stress_mean))
    
    feature_importance = pd.Series(stress_diff, index=top_features).sort_values(ascending=True)
    feature_importance.plot(kind='barh', ax=ax11, color='#e67e22')
    ax11.set_title('Top 10 Features Mean Difference', fontweight='bold', fontsize=9)
    ax11.set_xlabel('Absolute Difference', fontsize=8)
    ax11.tick_params(labelsize=7)
    
    # 12. Feature Statistics Summary
    ax12 = plt.subplot(3, 4, 12)
    # Show basic statistics
    stats_text = f"Dataset Statistics\n{'='*30}\n"
    stats_text += f"Total Samples: {len(df):,}\n"
    stats_text += f"Features: {len(features_cols)}\n"
    stats_text += f"Stress: {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)\n"
    stats_text += f"No Stress: {(df['label']==0).sum():,} ({(1-df['label'].mean())*100:.1f}%)\n\n"
    stats_text += f"Train: {(df['split']=='Train').sum():,}\n"
    stats_text += f"Test: {(df['split']=='Test').sum():,}\n\n"
    stats_text += f"Class Balance:\n"
    stats_text += f"{'Balanced' if abs(df['label'].mean() - 0.5) < 0.1 else 'Imbalanced'}"
    
    ax12.text(0.1, 0.5, stats_text, transform=ax12.transAxes, 
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax12.axis('off')
    
    plt.tight_layout()
    overview_path = output_dir / 'swell_comprehensive_overview.png'
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {overview_path}")
    plt.close()
    
    # ==================== VISUALIZATION 2: Detailed Feature Analysis ====================
    fig2, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig2.suptitle('SWELL Dataset - Detailed Feature Analysis (Top 16 Features)', fontsize=16, fontweight='bold')
    
    # Select top 16 features by variance
    top_16_features = feature_variances.head(16).index.tolist()
    
    for idx, feature in enumerate(top_16_features):
        ax = axes[idx // 4, idx % 4]
        
        # Violin plot for each feature by stress
        data_no_stress = df[df['label'] == 0][feature].values
        data_stress = df[df['label'] == 1][feature].values
        
        parts = ax.violinplot([data_no_stress, data_stress],
                              positions=[0, 1], showmeans=True, showmedians=True)
        
        # Color the violin plots
        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.7)
        
        ax.set_title(feature, fontweight='bold', fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No Stress', 'Stress'], fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    features_path = output_dir / 'swell_detailed_features.png'
    plt.savefig(features_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {features_path}")
    plt.close()
    
    # ==================== VISUALIZATION 3: Statistical Summary ====================
    fig3 = plt.figure(figsize=(18, 10))
    fig3.suptitle('SWELL Dataset - Statistical Summary', fontsize=16, fontweight='bold')
    
    # 1. Distribution plots for top 6 features by variance
    top_6_features = feature_variances.head(6).index.tolist()
    
    for idx, feature in enumerate(top_6_features, start=1):
        ax = plt.subplot(2, 3, idx)
        
        # Histogram with KDE
        df[df['label'] == 0][feature].hist(bins=30, alpha=0.5, label='No Stress', 
                                             color='#3498db', density=True, ax=ax)
        df[df['label'] == 1][feature].hist(bins=30, alpha=0.5, label='Stress', 
                                             color='#e74c3c', density=True, ax=ax)
        
        ax.set_title(f'{feature} Distribution', fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    stats_path = output_dir / 'swell_statistical_summary.png'
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {stats_path}")
    plt.close()
    
    # ==================== VISUALIZATION 4: Additional Analysis ====================
    fig4 = plt.figure(figsize=(18, 12))
    fig4.suptitle('SWELL Dataset - Additional Analysis', fontsize=16, fontweight='bold')
    
    # 1. Feature variance comparison
    ax1 = plt.subplot(2, 2, 1)
    top_10_var = feature_variances.head(10)
    top_10_var.plot(kind='barh', ax=ax1, color='#9b59b6')
    ax1.set_title('Top 10 Features by Variance', fontweight='bold')
    ax1.set_xlabel('Variance')
    ax1.set_ylabel('Feature')
    
    # 2. Feature distribution comparison (stress vs no stress)
    ax2 = plt.subplot(2, 2, 2)
    stress_means = df[df['label'] == 1][top_6_features].mean()
    no_stress_means = df[df['label'] == 0][top_6_features].mean()
    
    x = np.arange(len(top_6_features))
    width = 0.35
    ax2.bar(x - width/2, no_stress_means, width, label='No Stress', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, stress_means, width, label='Stress', color='#e74c3c', alpha=0.8)
    ax2.set_title('Mean Feature Values by Stress', fontweight='bold')
    ax2.set_xlabel('Feature')
    ax2.set_ylabel('Mean Value')
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_6_features, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Train/Test split class distribution
    ax3 = plt.subplot(2, 2, 3)
    split_stress_df = df.groupby(['split', 'Stress_Label']).size().reset_index(name='count')
    pivot_data = split_stress_df.pivot(index='split', columns='Stress_Label', values='count')
    pivot_data.plot(kind='bar', ax=ax3, color=colors_stress, alpha=0.8)
    ax3.set_title('Class Distribution by Split', fontweight='bold')
    ax3.set_xlabel('Dataset Split')
    ax3.set_ylabel('Number of Samples')
    ax3.legend(title='Stress')
    ax3.tick_params(axis='x', rotation=0)
    for container in ax3.containers:
        ax3.bar_label(container, fontsize=9)
    
    # 4. Feature correlation with label
    ax4 = plt.subplot(2, 2, 4)
    correlations = df[features_cols].corrwith(df['label']).abs().sort_values(ascending=False).head(15)
    correlations.plot(kind='barh', ax=ax4, color='#e67e22')
    ax4.set_title('Top 15 Features Correlation with Stress', fontweight='bold')
    ax4.set_xlabel('Absolute Correlation')
    ax4.set_ylabel('Feature')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    additional_path = output_dir / 'swell_additional_analysis.png'
    plt.savefig(additional_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {additional_path}")
    plt.close()
    
    # ==================== Generate Summary Statistics ====================
    print("\n" + "=" * 60)
    print("📊 SWELL Dataset Summary Statistics")
    print("=" * 60)
    print(f"\n📈 Dataset Overview:")
    print(f"  Total Samples: {len(df):,}")
    print(f"  Total Features: {len(features_cols)}")
    print(f"  Train Samples: {(df['split'] == 'Train').sum():,}")
    print(f"  Test Samples: {(df['split'] == 'Test').sum():,}")
    
    print(f"\n⚖️ Binary Classification:")
    for label, count in df['Stress_Label'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count:,} samples ({percentage:.1f}%)")
    
    print(f"\n📊 Feature Statistics (Top 5 by variance):")
    for feature, var_val in feature_variances.head(5).items():
        mean_val = df[feature].mean()
        print(f"  {feature}: mean={mean_val:.2f}, var={var_val:.2f}")
    
    print(f"\n🔗 Top 5 Features Correlated with Stress:")
    correlations = df[features_cols].corrwith(df['label']).abs().sort_values(ascending=False).head(5)
    for feature, corr_val in correlations.items():
        print(f"  {feature}: {corr_val:.4f}")
    
    print("\n" + "=" * 60)
    print(f"✅ All visualizations saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    create_comprehensive_visualization()
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print("   - swell_comprehensive_overview.png")
    print("   - swell_detailed_features.png")
    print("   - swell_statistical_summary.png")
    print("   - swell_participant_analysis.png")
    print("=" * 60)


if __name__ == "__main__":
    create_comprehensive_visualization()
