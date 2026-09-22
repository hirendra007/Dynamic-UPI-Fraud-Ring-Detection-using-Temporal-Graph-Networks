"""
02_resample_smote.py - Class Balancing with SMOTE & Exploratory Data Analysis

Reads the cleaned Parquet dataset, performs an 80/20 train/test split, applies
SMOTE to balance the training split while preserving an uncorrupted test set,
updates reports/metrics.json with before/after class distributions, and exports
EDA visualizations to reports/figures/.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Setup styling for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

def get_paths():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    cleaned_path = base_dir / "data" / "processed" / "creditcard_cleaned.parquet"
    train_path = base_dir / "data" / "processed" / "train_resampled.parquet"
    test_path = base_dir / "data" / "processed" / "test.parquet"
    metrics_path = base_dir / "reports" / "metrics.json"
    figures_dir = base_dir / "reports" / "figures"
    
    # Fallback to root layout if needed
    if not cleaned_path.exists() and (script_dir.parent.parent / "data" / "processed" / "creditcard_cleaned.parquet").exists():
        base_dir = script_dir.parent.parent
        cleaned_path = base_dir / "data" / "processed" / "creditcard_cleaned.parquet"
        train_path = base_dir / "data" / "processed" / "train_resampled.parquet"
        test_path = base_dir / "data" / "processed" / "test.parquet"
        metrics_path = base_dir / "reports" / "metrics.json"
        figures_dir = base_dir / "reports" / "figures"

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    return cleaned_path, train_path, test_path, metrics_path, figures_dir

def generate_eda_figures(df_full, y_train_orig, y_train_resampled, figures_dir):
    print("[*] Generating EDA figures...")
    
    # 1. Class Balance Chart (Before vs After SMOTE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before SMOTE (Train split)
    counts_before = pd.Series(y_train_orig).value_counts().sort_index()
    colors_before = ['#2b5c8f', '#d9534f']
    bars1 = axes[0].bar(['Legitimate (0)', 'Fraud (1)'], counts_before.values, color=colors_before, width=0.5, edgecolor='black', alpha=0.85)
    axes[0].set_yscale('log')
    axes[0].set_title('Training Set Before SMOTE (Log Scale)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Number of Transactions')
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height * 1.15,
                     f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    # After SMOTE (Train split)
    counts_after = pd.Series(y_train_resampled).value_counts().sort_index()
    colors_after = ['#2b5c8f', '#e67e22']
    bars2 = axes[1].bar(['Legitimate (0)', 'Fraud (1)'], counts_after.values, color=colors_after, width=0.5, edgecolor='black', alpha=0.85)
    axes[1].set_title('Training Set After SMOTE (Balanced 1:1)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Number of Transactions')
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height * 1.02,
                     f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Class Imbalance Resolution via SMOTE (Synthetic Minority Over-sampling)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    balance_plot_path = figures_dir / "class_balance.png"
    plt.savefig(balance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {balance_plot_path}")

    # 2. Transaction Amount Distribution (Fraud vs Legitimate)
    fig, ax = plt.subplots(figsize=(10, 5))
    fraud_amounts = df_full[df_full['Class'] == 1]['Amount']
    legit_amounts = df_full[df_full['Class'] == 0]['Amount']

    sns.kdeplot(np.log1p(legit_amounts), label='Legitimate (Class 0)', color='#2b5c8f', fill=True, alpha=0.35, ax=ax)
    sns.kdeplot(np.log1p(fraud_amounts), label='Fraudulent (Class 1)', color='#d9534f', fill=True, alpha=0.45, ax=ax)
    
    ax.set_title('Transaction Amount Density Distribution: log1p(Amount)', fontsize=13, fontweight='bold')
    ax.set_xlabel('log1p(Transaction Amount in USD/EUR)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.axvline(np.log1p(fraud_amounts.median()), color='#d9534f', linestyle='--', label=f'Fraud Median: ${fraud_amounts.median():.2f}')
    ax.axvline(np.log1p(legit_amounts.median()), color='#2b5c8f', linestyle=':', label=f'Legit Median: ${legit_amounts.median():.2f}')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    amount_plot_path = figures_dir / "amount_distribution.png"
    plt.savefig(amount_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {amount_plot_path}")

    # 3. Top Feature Correlations with Fraud Label
    corr = df_full.corr()['Class'].drop('Class')
    top_corr = pd.concat([corr.nlargest(8), corr.nsmallest(8)]).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d9534f' if val < 0 else '#2ecc71' for val in top_corr.values]
    bars = ax.barh(top_corr.index, top_corr.values, color=colors, edgecolor='black', alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Top Positive & Negative Feature Correlations with Fraud Label (Class)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
    
    for bar in bars:
        width = bar.get_width()
        offset = 0.01 if width >= 0 else -0.04
        ax.text(width + offset, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    corr_plot_path = figures_dir / "correlation_heatmap.png"
    plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {corr_plot_path}")

def main():
    start_time = time.time()
    print("=" * 70)
    print("PHASE 1: SMOTE Class Balancing & EDA Pass")
    print("=" * 70)

    cleaned_path, train_path, test_path, metrics_path, figures_dir = get_paths()

    if not cleaned_path.exists():
        print(f"[ERROR] Cleaned dataset not found at: {cleaned_path}")
        print("Please run 01_ingest_and_clean.py first.")
        sys.exit(1)

    print(f"[*] Loading cleaned dataset from: {cleaned_path}")
    df = pd.read_parquet(cleaned_path)
    total_records = len(df)
    feature_cols = [c for c in df.columns if c != 'Class']
    X = df[feature_cols]
    y = df['Class']

    total_fraud = int((y == 1).sum())
    total_legit = int((y == 0).sum())
    raw_ratio = (total_fraud / total_records) * 100

    print(f"[*] Total dataset size: {total_records:,} rows")
    print(f"    Legitimate: {total_legit:,} | Fraud: {total_fraud:,} ({raw_ratio:.4f}%)")

    # Stratified 80/20 Train-Test split
    print("[*] Performing stratified 80/20 Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    train_fraud_before = int((y_train == 1).sum())
    train_legit_before = int((y_train == 0).sum())
    test_fraud = int((y_test == 1).sum())
    test_legit = int((y_test == 0).sum())

    print(f"    Train Split: {len(X_train):,} rows (Legit: {train_legit_before:,}, Fraud: {train_fraud_before:,})")
    print(f"    Test Split (Pristine): {len(X_test):,} rows (Legit: {test_legit:,}, Fraud: {test_fraud:,})")

    # Apply SMOTE to training split only
    print("[*] Applying SMOTE over-sampling to training split...")
    smote = SMOTE(random_state=42, sampling_strategy=1.0) # 1:1 balance
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    train_fraud_after = int((y_train_resampled == 1).sum())
    train_legit_after = int((y_train_resampled == 0).sum())

    print("-" * 50)
    print(f"    Train After SMOTE: {len(X_train_resampled):,} rows")
    print(f"    Legitimate (Class 0): {train_legit_after:,}")
    print(f"    Fraudulent (Class 1): {train_fraud_after:,} (50.0% / 1:1 ratio)")
    print("-" * 50)

    # Save resampled train and untouched test splits
    print(f"[*] Writing resampled train split to: {train_path}")
    df_train_resampled = pd.concat([X_train_resampled, y_train_resampled], axis=1)
    df_train_resampled.to_parquet(train_path, index=False, compression='snappy')

    print(f"[*] Writing pristine test split to: {test_path}")
    df_test = pd.concat([X_test, y_test], axis=1)
    df_test.to_parquet(test_path, index=False, compression='snappy')

    # Generate EDA Figures
    generate_eda_figures(df, y_train, y_train_resampled, figures_dir)

    # Update reports/metrics.json
    metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    metrics["dataset_summary"] = {
        "total_records_cleaned": total_records,
        "total_legitimate": total_legit,
        "total_fraud": total_fraud,
        "raw_fraud_percentage": round(raw_ratio, 4)
    }
    metrics["smote_resampling"] = {
        "train_samples_before_smote": len(X_train),
        "train_fraud_before_smote": train_fraud_before,
        "train_legit_before_smote": train_legit_before,
        "train_samples_after_smote": len(X_train_resampled),
        "train_fraud_after_smote": train_fraud_after,
        "train_legit_after_smote": train_legit_after,
        "test_samples_count": len(X_test),
        "test_fraud_count": test_fraud,
        "test_legit_count": test_legit
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[*] Saved updated metrics to: {metrics_path}")

    elapsed = time.time() - start_time
    print(f"[SUCCESS] SMOTE balancing and EDA pass completed in {elapsed:.2f}s!")

if __name__ == "__main__":
    main()
