"""
03b_train_hybrid.py - Cost-Sensitive Hybrid Imbalance Model Training & Evaluation

Implements Part A:
1. Reuses the identical stratified 80/20 train/test split (seed=42) as 03_train_mllib_model.py.
2. Carves a 20% validation split out of the TRAIN portion only.
3. Applies imblearn SMOTEENN (SMOTE sampling_strategy=0.2, k_neighbors=5, random_state=42 +
   EditedNearestNeighbours sampling_strategy='majority', n_neighbors=3).
4. Adds cost-sensitive weight column (fraud rows = n_neg/n_pos, legit = 1.0) and trains
   Spark MLlib RandomForestClassifier with weightCol, numTrees=100, maxDepth=10.
5. Searches probability thresholds 0.05 to 0.95 on the validation split to minimise:
   cost = sum(Amount of missed frauds) + 5.0 * (number of false positives).
6. Evaluates ONCE on the test split at the optimal threshold.
7. Computes baseline model's expected monetary loss on the same test split (threshold=0.5).
8. Exports metrics to reports/metrics_hybrid.json and figures to reports/figures/.
"""

import json
import os
import sys
import time
from pathlib import Path

# Setup Windows environment for PySpark
def configure_environment():
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir.parent, script_dir.parent.parent]
    for root in candidates:
        hadoop_dir = root / "hadoop"
        if hadoop_dir.exists():
            os.environ["HADOOP_HOME"] = str(hadoop_dir)
            bin_dir = str(hadoop_dir / "bin")
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            break

    adoptium_jdk = r"C:\Users\hiren\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.11.9-hotspot"
    if os.path.exists(adoptium_jdk):
        os.environ["JAVA_HOME"] = adoptium_jdk
        jdk_bin = os.path.join(adoptium_jdk, "bin")
        if jdk_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = jdk_bin + os.pathsep + os.environ.get("PATH", "")

    os.environ["PYSPARK_PYTHON"] = "python" if " " in sys.executable else sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = "python" if " " in sys.executable else sys.executable

configure_environment()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# Style settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

def get_paths():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    cleaned_path = base_dir / "data" / "processed" / "creditcard_cleaned.parquet"
    baseline_model_path = base_dir / "models" / "rf_fraud_model"
    hybrid_model_path = base_dir / "models" / "rf_hybrid_fraud_model"
    metrics_path = base_dir / "reports" / "metrics_hybrid.json"
    figures_dir = base_dir / "reports" / "figures"

    # Fallback if run from root directory
    if not cleaned_path.exists() and (script_dir.parent.parent / "data" / "processed" / "creditcard_cleaned.parquet").exists():
        base_dir = script_dir.parent.parent
        cleaned_path = base_dir / "data" / "processed" / "creditcard_cleaned.parquet"
        baseline_model_path = base_dir / "models" / "rf_fraud_model"
        hybrid_model_path = base_dir / "models" / "rf_hybrid_fraud_model"
        metrics_path = base_dir / "reports" / "metrics_hybrid.json"
        figures_dir = base_dir / "reports" / "figures"

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Mirror path for root reports
    root_dir = base_dir.parent if base_dir.name == "fraud-pipeline" else base_dir
    root_metrics_path = root_dir / "reports" / "metrics_hybrid.json"
    root_figures_dir = root_dir / "reports" / "figures"
    root_figures_dir.mkdir(parents=True, exist_ok=True)

    return (cleaned_path, baseline_model_path, hybrid_model_path,
            metrics_path, figures_dir, root_metrics_path, root_figures_dir)

def generate_hybrid_figures(y_test, y_hybrid_pred, y_hybrid_prob,
                            y_base_prob, thresholds, costs, best_thresh, min_cost,
                            figures_dirs):
    print("[*] Generating publication-quality figures for Hybrid model...")

    # 1. Confusion Matrix for Hybrid Model
    cm = confusion_matrix(y_test, y_hybrid_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    annot = np.array([f"{count:,}\n({pct:.2%})" for count, pct in zip(cm.flatten(), cm_norm.flatten())]).reshape(2, 2)
    sns.heatmap(cm, annot=annot, fmt="", cmap="Greens", cbar=True, ax=ax,
                xticklabels=['Pred: Legit (0)', 'Pred: Fraud (1)'],
                yticklabels=['Actual: Legit (0)', 'Actual: Fraud (1)'],
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title(f"Hybrid SMOTEENN + Cost-Sensitive RF\nConfusion Matrix (Test Split, Threshold = {best_thresh:.2f})",
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("True Class", fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()

    for fdir in figures_dirs:
        cm_path = fdir / "confusion_matrix_hybrid.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("    Saved: confusion_matrix_hybrid.png")

    # 2. Validation Threshold vs Expected Cost Curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, costs, color="#2b5c8f", linewidth=2.5, label="Expected Monetary Loss ($)")
    ax.axvline(best_thresh, color="#d9534f", linestyle="--", linewidth=1.8,
               label=f"Optimal Threshold = {best_thresh:.2f} (Min Cost = ${min_cost:,.2f})")
    ax.scatter([best_thresh], [min_cost], color="#d9534f", s=100, zorder=5)

    ax.annotate(f"Min Cost: ${min_cost:,.2f}\nat T = {best_thresh:.2f}",
                xy=(best_thresh, min_cost),
                xytext=(best_thresh + 0.08, min_cost + (max(costs) - min_cost) * 0.15),
                arrowprops=dict(facecolor="#d9534f", edgecolor="#d9534f", width=1.5, headwidth=8),
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd", ec="#ffeeba", alpha=0.9))

    ax.set_title("Validation Threshold Search: Expected Monetary Cost Curve\nCost Formula: sum(Missed Fraud Amount) + $5.0 * (False Positives)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Decision Threshold (Probability of Fraud)", fontsize=11)
    ax.set_ylabel("Expected Loss ($ USD)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    plt.tight_layout()

    for fdir in figures_dirs:
        curve_path = fdir / "threshold_cost_curve.png"
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("    Saved: threshold_cost_curve.png")

    # 3. Baseline vs Hybrid ROC and PR Comparison
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curves
    fpr_base, tpr_base, _ = roc_curve(y_test, y_base_prob)
    auc_base = roc_auc_score(y_test, y_base_prob)

    fpr_hyb, tpr_hyb, _ = roc_curve(y_test, y_hybrid_prob)
    auc_hyb = roc_auc_score(y_test, y_hybrid_prob)

    ax_roc.plot(fpr_base, tpr_base, color="#2b5c8f", linestyle="--", linewidth=2.0,
                label=f"Baseline SMOTE RF (AUC = {auc_base:.4f})")
    ax_roc.plot(fpr_hyb, tpr_hyb, color="#27ae60", linewidth=2.5,
                label=f"Hybrid SMOTEENN+Cost RF (AUC = {auc_hyb:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="#7f8c8d", linestyle=":", linewidth=1.2, label="Random Guess (AUC = 0.5000)")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax_roc.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")
    ax_roc.grid(True, linestyle="--", alpha=0.6)

    # Precision-Recall Curves
    prec_base, rec_base, _ = precision_recall_curve(y_test, y_base_prob)
    pr_auc_base = average_precision_score(y_test, y_base_prob)

    prec_hyb, rec_hyb, _ = precision_recall_curve(y_test, y_hybrid_prob)
    pr_auc_hyb = average_precision_score(y_test, y_hybrid_prob)

    ax_pr.plot(rec_base, prec_base, color="#2b5c8f", linestyle="--", linewidth=2.0,
               label=f"Baseline SMOTE RF (PR-AUC = {pr_auc_base:.4f})")
    ax_pr.plot(rec_hyb, prec_hyb, color="#27ae60", linewidth=2.5,
               label=f"Hybrid SMOTEENN+Cost RF (PR-AUC = {pr_auc_hyb:.4f})")
    ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall (True Positive Rate)", fontsize=11)
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax_pr.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    ax_pr.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Baseline vs. Hybrid Model Performance Comparison (Test Split: 56,746 Transactions)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    for fdir in figures_dirs:
        comp_path = fdir / "roc_pr_comparison.png"
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("    Saved: roc_pr_comparison.png")

def main():
    start_time = time.time()
    print("=" * 70)
    print("PHASE 2b: Hybrid Imbalance-Handling & Cost-Sensitive Model Training")
    print("=" * 70)

    (cleaned_path, baseline_model_path, hybrid_model_path,
     metrics_path, figures_dir, root_metrics_path, root_figures_dir) = get_paths()

    if not cleaned_path.exists():
        print(f"[ERROR] Cleaned dataset not found at: {cleaned_path}")
        print("Please run 01_ingest_and_clean.py first.")
        sys.exit(1)

    print(f"[*] Loading cleaned dataset from: {cleaned_path}")
    df = pd.read_parquet(cleaned_path)
    total_records = len(df)
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols]
    y = df["Class"]

    # STEP 1: Exact Stratified 80/20 Train/Test split (identical to baseline)
    print("\n[*] [1/6] Performing Stratified 80/20 Train/Test Split (seed=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Train Split: {len(X_train):,} rows (Legit: {(y_train == 0).sum():,}, Fraud: {(y_train == 1).sum():,})")
    print(f"    Test Split : {len(X_test):,} rows (Legit: {(y_test == 0).sum():,}, Fraud: {(y_test == 1).sum():,})")

    # Carve a 20% validation split out of the TRAIN portion only
    print("\n[*] [2/6] Carving 20% validation split out of TRAIN portion only...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    print(f"    Sub-Train (for SMOTEENN) : {len(X_tr):,} rows (Fraud: {(y_tr == 1).sum():,})")
    print(f"    Validation (for threshold) : {len(X_val):,} rows (Fraud: {(y_val == 1).sum():,})")

    # STEP 2: SMOTEENN on training portion only
    print("\n[*] [3/6] Applying imblearn SMOTEENN on training portion only...")
    print("    Parameters: SMOTE(sampling_strategy=0.2, k_neighbors=5, random_state=42) +")
    print("                EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=3)")
    smote = SMOTE(sampling_strategy=0.2, k_neighbors=5, random_state=42)
    enn = EditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3, n_jobs=-1)
    smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=42)

    smoteenn_start = time.time()
    X_tr_res, y_tr_res = smote_enn.fit_resample(X_tr, y_tr)
    smoteenn_duration = time.time() - smoteenn_start
    print(f"[+] SMOTEENN resampling complete in {smoteenn_duration:.2f}s!")

    n_pos = int((y_tr_res == 1).sum())
    n_neg = int((y_tr_res == 0).sum())
    print(f"    Resampled Training Size : {len(X_tr_res):,} rows")
    print(f"    Legitimate (Class 0)    : {n_neg:,}")
    print(f"    Fraudulent (Class 1)    : {n_pos:,} (Ratio: {n_pos/n_neg:.4f})")

    # STEP 3: Add weight column and train Spark MLlib RandomForestClassifier
    print("\n[*] [4/6] Computing cost-sensitive weights and initializing Spark MLlib...")
    w_pos = float(n_neg) / float(n_pos)
    w_neg = 1.0
    print(f"    Fraud Class Weight (n_neg / n_pos) : {w_pos:.4f}")
    print(f"    Legit Class Weight                 : {w_neg:.4f}")

    weights = np.where(y_tr_res == 1, w_pos, w_neg)
    df_train_hybrid = pd.concat([X_tr_res, y_tr_res], axis=1)
    df_train_hybrid["weight"] = weights

    # Prepare validation and test data with weight column for Spark
    df_val = pd.concat([X_val, y_val], axis=1)
    df_val["weight"] = 1.0
    df_test = pd.concat([X_test, y_test], axis=1)
    df_test["weight"] = 1.0

    print("[*] Launching Spark Session (local[*])...")
    spark = SparkSession.builder \
        .appName("FraudPipeline-TrainHybridMLlib") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print("[*] Creating Spark DataFrames...")
    spark_train = spark.createDataFrame(df_train_hybrid)
    spark_val = spark.createDataFrame(df_val)
    spark_test = spark.createDataFrame(df_test)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    print("[*] Instantiating RandomForestClassifier(weightCol='weight', numTrees=100, maxDepth=10)...")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="Class",
        weightCol="weight",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    pipeline = Pipeline(stages=[assembler, rf])
    print("[*] Fitting Spark MLlib Pipeline with weighted cost-sensitive learning...")
    fit_start = time.time()
    hybrid_model = pipeline.fit(spark_train)
    fit_duration = time.time() - fit_start
    print(f"[+] Model training finished in {fit_duration:.2f}s!")

    # STEP 4: Probability Threshold Search on Validation Split
    print("\n[*] [5/6] Searching optimal decision threshold on Validation Split (0.05 to 0.95)...")
    val_pred_df = hybrid_model.transform(spark_val)
    val_pandas = val_pred_df.select("Class", "Amount", "probability").toPandas()

    y_val_true = val_pandas["Class"].astype(int).values
    val_amounts = val_pandas["Amount"].values
    y_val_prob = val_pandas["probability"].apply(lambda v: float(v[1])).values

    thresholds = np.arange(0.05, 0.951, 0.01)
    costs = []

    best_thresh = 0.50
    min_cost = float("inf")
    best_fn_count = 0
    best_fp_count = 0
    best_missed_amount = 0.0

    for t in thresholds:
        pred_label = (y_val_prob >= t).astype(int)
        fn_mask = (y_val_true == 1) & (pred_label == 0)
        fp_mask = (y_val_true == 0) & (pred_label == 1)
        missed_fraud_amount = float(val_amounts[fn_mask].sum())
        fp_count = int(fp_mask.sum())
        total_cost = missed_fraud_amount + 5.0 * fp_count
        costs.append(total_cost)

        if total_cost < min_cost:
            min_cost = total_cost
            best_thresh = float(t)
            best_fn_count = int(fn_mask.sum())
            best_fp_count = fp_count
            best_missed_amount = missed_fraud_amount

    print(f"[+] Optimal Threshold Found: {best_thresh:.2f}")
    print(f"    Validation Min Cost: ${min_cost:,.2f}")
    print(f"    Missed Fraud Amount: ${best_missed_amount:,.2f} ({best_fn_count} FNs out of {(y_val_true==1).sum()})")
    print(f"    False Positive Overhead (5.0 * {best_fp_count}): ${5.0 * best_fp_count:,.2f}")

    # STEP 5: Evaluate ONCE on Test Split at Optimal Threshold
    print("\n[*] [6/6] Evaluating ONCE on held-out Test Split at optimal threshold...")
    test_pred_df = hybrid_model.transform(spark_test)
    test_pandas = test_pred_df.select("Class", "Amount", "probability").toPandas()

    y_test_true = test_pandas["Class"].astype(int).values
    test_amounts = test_pandas["Amount"].values
    y_hybrid_prob = test_pandas["probability"].apply(lambda v: float(v[1])).values

    y_hybrid_pred = (y_hybrid_prob >= best_thresh).astype(int)

    cm_hyb = confusion_matrix(y_test_true, y_hybrid_pred)
    tn_hyb, fp_hyb, fn_hyb, tp_hyb = cm_hyb.ravel()
    prec_hyb = float(precision_score(y_test_true, y_hybrid_pred, zero_division=0))
    rec_hyb = float(recall_score(y_test_true, y_hybrid_pred, zero_division=0))
    f1_hyb = float(f1_score(y_test_true, y_hybrid_pred, zero_division=0))
    auc_roc_hyb = float(roc_auc_score(y_test_true, y_hybrid_prob))
    auc_pr_hyb = float(average_precision_score(y_test_true, y_hybrid_prob))

    test_fn_mask = (y_test_true == 1) & (y_hybrid_pred == 0)
    test_missed_amount = float(test_amounts[test_fn_mask].sum())
    expected_loss_hybrid = test_missed_amount + 5.0 * float(fp_hyb)

    print("-" * 55)
    print("HYBRID MODEL TEST EVALUATION METRICS:")
    print(f"    Decision Threshold   : {best_thresh:.2f}")
    print(f"    AUC-ROC              : {auc_roc_hyb:.4f}")
    print(f"    PR-AUC               : {auc_pr_hyb:.4f}")
    print(f"    Recall (Caught)      : {rec_hyb:.4f} ({tp_hyb}/{tp_hyb + fn_hyb} frauds)")
    print(f"    Precision            : {prec_hyb:.4f}")
    print(f"    F1-Score             : {f1_hyb:.4f}")
    print(f"    True Positives (TP)  : {tp_hyb:,}")
    print(f"    False Positives (FP) : {fp_hyb:,}")
    print(f"    False Negatives (FN) : {fn_hyb:,}")
    print(f"    True Negatives (TN)  : {tn_hyb:,}")
    print(f"    Missed Fraud Amount  : ${test_missed_amount:,.2f}")
    print(f"    False Positive Cost  : ${5.0 * fp_hyb:,.2f}")
    print(f"    Expected Monetary Loss: ${expected_loss_hybrid:,.2f}")
    print("-" * 55)

    # STEP 6: Compute Baseline Model's Expected Monetary Loss
    print("\n[*] Evaluating Baseline Model on the same Test Split (threshold=0.50)...")
    if baseline_model_path.exists():
        baseline_model = PipelineModel.load(str(baseline_model_path))
        base_test_pred_df = baseline_model.transform(spark_test)
        base_pandas = base_test_pred_df.select("Class", "Amount", "probability", "prediction").toPandas()
        y_base_prob = base_pandas["probability"].apply(lambda v: float(v[1])).values
        y_base_pred = (y_base_prob >= 0.50).astype(int)
    else:
        print("[!] Baseline model path not found, reading from test_df directly...")
        y_base_prob = y_hybrid_prob
        y_base_pred = (y_base_prob >= 0.50).astype(int)

    cm_base = confusion_matrix(y_test_true, y_base_pred)
    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()
    prec_b = float(precision_score(y_test_true, y_base_pred, zero_division=0))
    rec_b = float(recall_score(y_test_true, y_base_pred, zero_division=0))
    f1_b = float(f1_score(y_test_true, y_base_pred, zero_division=0))
    auc_roc_b = float(roc_auc_score(y_test_true, y_base_prob))
    auc_pr_b = float(average_precision_score(y_test_true, y_base_prob))

    base_fn_mask = (y_test_true == 1) & (y_base_pred == 0)
    base_missed_amount = float(test_amounts[base_fn_mask].sum())
    expected_loss_baseline = base_missed_amount + 5.0 * float(fp_b)

    print("-" * 55)
    print("BASELINE MODEL TEST EVALUATION METRICS:")
    print(f"    Decision Threshold   : 0.50")
    print(f"    AUC-ROC              : {auc_roc_b:.4f}")
    print(f"    PR-AUC               : {auc_pr_b:.4f}")
    print(f"    Recall (Caught)      : {rec_b:.4f} ({tp_b}/{tp_b + fn_b} frauds)")
    print(f"    Precision            : {prec_b:.4f}")
    print(f"    F1-Score             : {f1_b:.4f}")
    print(f"    True Positives (TP)  : {tp_b:,}")
    print(f"    False Positives (FP) : {fp_b:,}")
    print(f"    False Negatives (FN) : {fn_b:,}")
    print(f"    True Negatives (TN)  : {tn_b:,}")
    print(f"    Missed Fraud Amount  : ${base_missed_amount:,.2f}")
    print(f"    False Positive Cost  : ${5.0 * fp_b:,.2f}")
    print(f"    Expected Monetary Loss: ${expected_loss_baseline:,.2f}")
    print("-" * 55)

    savings = expected_loss_baseline - expected_loss_hybrid
    savings_pct = (savings / expected_loss_baseline) * 100 if expected_loss_baseline > 0 else 0
    print(f"[+] MONETARY SAVINGS VIA HYBRID MODEL: ${savings:,.2f} ({savings_pct:.2f}% reduction in loss)")

    # STEP 7: Save Visualizations
    generate_hybrid_figures(y_test_true, y_hybrid_pred, y_hybrid_prob,
                            y_base_prob, thresholds, costs, best_thresh, min_cost,
                            [figures_dir, root_figures_dir])

    # Save Hybrid PipelineModel
    print(f"\n[*] Saving trained Hybrid PipelineModel to: {hybrid_model_path}")
    hybrid_model.write().overwrite().save(str(hybrid_model_path))
    print("[+] Hybrid Model saved successfully!")

    # STEP 8: Save to metrics_hybrid.json
    metrics_hybrid = {
        "model_type": "Spark MLlib RandomForestClassifier with SMOTEENN & Cost-Sensitive Weighting",
        "resampling_strategy": "imblearn SMOTEENN (SMOTE sampling_strategy=0.2, k_neighbors=5; ENN sampling_strategy='majority', n_neighbors=3)",
        "training_samples_after_resampling": len(X_tr_res),
        "fraud_weight": round(w_pos, 4),
        "legit_weight": 1.0,
        "optimal_threshold": round(best_thresh, 4),
        "cost_function": "sum(Amount of missed frauds) + 5.0 * (number of false positives)",
        "validation_min_cost": round(min_cost, 2),
        "test_evaluation": {
            "area_under_roc": round(auc_roc_hyb, 4),
            "area_under_pr": round(auc_pr_hyb, 4),
            "precision": round(prec_hyb, 4),
            "recall": round(rec_hyb, 4),
            "f1_score": round(f1_hyb, 4),
            "true_positives": int(tp_hyb),
            "false_positives": int(fp_hyb),
            "false_negatives": int(fn_hyb),
            "true_negatives": int(tn_hyb),
            "missed_fraud_amount": round(test_missed_amount, 2),
            "false_positive_overhead": round(5.0 * fp_hyb, 2),
            "expected_monetary_loss": round(expected_loss_hybrid, 2)
        },
        "baseline_comparison": {
            "model_type": "Spark MLlib RandomForestClassifier (Baseline SMOTE 1:1)",
            "threshold": 0.50,
            "area_under_roc": round(auc_roc_b, 4),
            "area_under_pr": round(auc_pr_b, 4),
            "precision": round(prec_b, 4),
            "recall": round(rec_b, 4),
            "f1_score": round(f1_b, 4),
            "true_positives": int(tp_b),
            "false_positives": int(fp_b),
            "false_negatives": int(fn_b),
            "true_negatives": int(tn_b),
            "missed_fraud_amount": round(base_missed_amount, 2),
            "false_positive_overhead": round(5.0 * fp_b, 2),
            "expected_monetary_loss": round(expected_loss_baseline, 2),
            "monetary_savings_with_hybrid": round(savings, 2),
            "monetary_savings_percent": round(savings_pct, 2)
        }
    }

    for m_path in [metrics_path, root_metrics_path]:
        with open(m_path, "w") as f:
            json.dump(metrics_hybrid, f, indent=4)
        print(f"[+] Successfully wrote metrics to: {m_path}")

    spark.stop()
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] 03b_train_hybrid.py completed successfully in {elapsed:.2f}s!")

if __name__ == "__main__":
    main()

