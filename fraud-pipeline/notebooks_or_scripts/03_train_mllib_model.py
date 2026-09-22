"""
03_train_mllib_model.py - Spark MLlib Random Forest Training & Evaluation

Assembles features, trains a Spark MLlib RandomForestClassifier (numTrees=100)
on the balanced training data, evaluates on the held-out test split using AUC-ROC,
PR-AUC, precision, recall, and F1-score, exports the trained PipelineModel to
models/rf_fraud_model/, and updates metrics.json and reports/figures/.
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
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    precision_recall_curve
)

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def get_paths():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    train_path = base_dir / "data" / "processed" / "train_resampled.parquet"
    test_path = base_dir / "data" / "processed" / "test.parquet"
    model_path = base_dir / "models" / "rf_fraud_model"
    metrics_path = base_dir / "reports" / "metrics.json"
    figures_dir = base_dir / "reports" / "figures"

    # Fallback to root layout if running in alternative layout
    if not train_path.exists() and (script_dir.parent.parent / "data" / "processed" / "train_resampled.parquet").exists():
        base_dir = script_dir.parent.parent
        train_path = base_dir / "data" / "processed" / "train_resampled.parquet"
        test_path = base_dir / "data" / "processed" / "test.parquet"
        model_path = base_dir / "models" / "rf_fraud_model"
        metrics_path = base_dir / "reports" / "metrics.json"
        figures_dir = base_dir / "reports" / "figures"

    figures_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return train_path, test_path, model_path, metrics_path, figures_dir

def generate_model_figures(y_true, y_pred, y_prob, feature_names, importances, figures_dir, roc_auc):
    print("[*] Generating model evaluation figures...")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    annot = np.array([f"{count:,}\n({pct:.2%})" for count, pct in zip(cm.flatten(), cm_norm.flatten())]).reshape(2, 2)
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True, ax=ax,
                xticklabels=['Pred: Legit', 'Pred: Fraud'],
                yticklabels=['Actual: Legit', 'Actual: Fraud'],
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title("Spark MLlib Random Forest Confusion Matrix (Test Split)", fontsize=13, fontweight='bold')
    ax.set_ylabel("True Class", fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {cm_path}")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2b5c8f', lw=2.5, label=f'Random Forest ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#888888', lw=1.5, linestyle='--', label='Random Chance Baseline (AUC = 0.5000)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.tight_layout()
    roc_path = figures_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {roc_path}")

    # 3. Top 15 Feature Importances
    sorted_idx = np.argsort(importances)[::-1][:15]
    top_features = [feature_names[i] for i in sorted_idx]
    top_importances = [importances[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_features[::-1], top_importances[::-1], color='#3498db', edgecolor='black', alpha=0.85)
    ax.set_xlabel('Relative Feature Importance (Gini Impurity Reduction)', fontsize=11)
    ax.set_title('Spark MLlib Random Forest - Top 15 Feature Importances', fontsize=13, fontweight='bold')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fi_path = figures_dir / "feature_importance.png"
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fi_path}")

def main():
    start_time = time.time()
    print("=" * 70)
    print("PHASE 2: Spark MLlib Random Forest Classifier Training")
    print("=" * 70)

    train_path, test_path, model_path, metrics_path, figures_dir = get_paths()

    if not train_path.exists() or not test_path.exists():
        print(f"[ERROR] Processed datasets not found at:\n  {train_path}\n  {test_path}")
        print("Please run 02_resample_smote.py first.")
        sys.exit(1)

    print("[*] Initializing PySpark Session (local[*])...")
    spark = SparkSession.builder \
        .appName("FraudPipeline-TrainMLlib") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print(f"[*] Reading training Parquet: {train_path}")
    train_df = spark.read.parquet(str(train_path))
    print(f"[*] Reading test Parquet: {test_path}")
    test_df = spark.read.parquet(str(test_path))

    train_count = train_df.count()
    test_count = test_df.count()
    print(f"[*] Train set rows: {train_count:,} | Test set rows: {test_count:,}")

    # Assemble feature columns (Time, Amount, V1-V28)
    feature_cols = [c for c in test_df.columns if c != "Class"]
    print(f"[*] Assembling {len(feature_cols)} feature columns...")

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    # Instantiate Random Forest Classifier
    # numTrees=100, maxDepth=10 to balance expressive capacity with generalizability
    print("[*] Instantiating RandomForestClassifier (numTrees=100, maxDepth=10)...")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="Class",
        numTrees=100,
        maxDepth=10,
        featureSubsetStrategy="auto",
        seed=42
    )

    # Build Pipeline: Assembler -> RF Classifier
    pipeline = Pipeline(stages=[assembler, rf])

    print("[*] Fitting Spark MLlib Pipeline on training data (this may take a minute)...")
    train_start = time.time()
    model = pipeline.fit(train_df)
    train_duration = time.time() - train_start
    print(f"[*] Training finished in {train_duration:.2f}s!")

    # Evaluate on held-out test split
    print("[*] Transforming held-out test split for predictions...")
    predictions = model.transform(test_df)

    # Spark ML Evaluators
    roc_evaluator = BinaryClassificationEvaluator(
        labelCol="Class",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="Class",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    auc_roc = float(roc_evaluator.evaluate(predictions))
    auc_pr = float(pr_evaluator.evaluate(predictions))
    print(f"\n[*] Spark ML Evaluator Results:")
    print(f"    Area Under ROC (AUC-ROC): {auc_roc:.4f}")
    print(f"    Area Under PR  (PR-AUC) : {auc_pr:.4f}")

    # Extract vectors to compute Precision, Recall, F1, and Confusion Matrix
    print("[*] Computing precision, recall, F1, and confusion matrix...")
    pred_pd = predictions.select("Class", "prediction", "probability").toPandas()
    y_true = pred_pd["Class"].astype(int).values
    y_pred = pred_pd["prediction"].astype(int).values
    # Extract probability of positive class (fraud)
    y_prob = pred_pd["probability"].apply(lambda v: float(v[1])).values

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float((tp + tn) / (tp + tn + fp + fn))

    print("-" * 50)
    print(f"    True Negatives (TN) : {tn:,}")
    print(f"    False Positives (FP): {fp:,}")
    print(f"    False Negatives (FN): {fn:,}")
    print(f"    True Positives (TP) : {tp:,}")
    print("-" * 50)
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}")
    print(f"    F1-Score : {f1:.4f}")
    print(f"    Accuracy : {acc:.4f}")
    print("-" * 50)

    # Extract Feature Importances from the RF model stage
    rf_stage = model.stages[-1]
    importances = rf_stage.featureImportances.toArray()

    # Generate Figures
    generate_model_figures(y_true, y_pred, y_prob, feature_cols, importances, figures_dir, auc_roc)

    # Save Pipeline Model to models/rf_fraud_model/
    print(f"[*] Saving trained PipelineModel to: {model_path}")
    model.write().overwrite().save(str(model_path))
    print(f"[+] Model saved successfully!")

    # Update reports/metrics.json
    metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    metrics["model_evaluation"] = {
        "model_type": "Spark MLlib RandomForestClassifier",
        "num_trees": 100,
        "max_depth": 10,
        "training_duration_seconds": round(train_duration, 2),
        "area_under_roc": round(auc_roc, 4),
        "area_under_pr": round(auc_pr, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "top_features": [
            {"feature": feature_cols[i], "importance": round(float(importances[i]), 5)}
            for i in np.argsort(importances)[::-1][:5]
        ]
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[*] Successfully recorded evaluation metrics to: {metrics_path}")

    total_elapsed = time.time() - start_time
    print(f"[SUCCESS] Phase 2 Model Training & Evaluation completed in {total_elapsed:.2f}s!")
    spark.stop()

if __name__ == "__main__":
    main()
