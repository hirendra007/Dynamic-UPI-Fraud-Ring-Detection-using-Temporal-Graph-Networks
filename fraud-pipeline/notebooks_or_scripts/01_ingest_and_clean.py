"""
01_ingest_and_clean.py - Batch Ingestion & Cleaning with PySpark

Loads the raw Kaggle Credit Card Fraud Detection (ULB) dataset into Apache Spark,
performs schema validation, drops duplicates and nulls, casts column types, and
exports optimized cleaned Parquet storage in data/processed/.
"""

import os
import sys
import time
from pathlib import Path

# Set up Windows environment paths for Hadoop and Java 17
def configure_environment():
    script_dir = Path(__file__).resolve().parent
    # Check project root and fraud-pipeline root
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

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, IntegerType, StructType, StructField

def get_data_paths():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    raw_path = base_dir / "data" / "raw" / "creditcard.csv"
    processed_path = base_dir / "data" / "processed" / "creditcard_cleaned.parquet"

    # Fallback to root data dir if running in alternative layout
    if not raw_path.exists() and (script_dir.parent.parent / "data" / "raw" / "creditcard.csv").exists():
        raw_path = script_dir.parent.parent / "data" / "raw" / "creditcard.csv"
        processed_path = script_dir.parent.parent / "data" / "processed" / "creditcard_cleaned.parquet"

    return raw_path, processed_path

def main():
    start_time = time.time()
    print("=" * 70)
    print("PHASE 1: Spark Data Ingestion & Cleansing")
    print("=" * 70)

    raw_path, processed_path = get_data_paths()
    print(f"[*] Checking input raw dataset: {raw_path}")

    if not raw_path.exists():
        print(f"\n[ERROR] Raw dataset not found at: {raw_path}")
        print("\nPlease ensure creditcard.csv is downloaded into data/raw/")
        print("Kaggle download command:")
        print("  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip")
        print("Or manual download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        sys.exit(1)

    print("[*] Initializing PySpark Session (local[*])...")
    spark = SparkSession.builder \
        .appName("FraudPipeline-IngestClean") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Define explicit schema for Credit Card Fraud dataset
    # Time, V1-V28 (Double), Amount (Double), Class (Integer)
    fields = [StructField("Time", DoubleType(), True)]
    for i in range(1, 29):
        fields.append(StructField(f"V{i}", DoubleType(), True))
    fields.append(StructField("Amount", DoubleType(), True))
    fields.append(StructField("Class", IntegerType(), True))
    schema = StructType(fields)

    print(f"[*] Ingesting raw CSV from: {raw_path}")
    df_raw = spark.read.csv(
        str(raw_path),
        header=True,
        schema=schema,
        mode="DROPMALFORMED"
    )

    initial_count = df_raw.count()
    print(f"[*] Raw row count: {initial_count:,}")

    # Drop nulls across all features
    df_clean = df_raw.na.drop()
    non_null_count = df_clean.count()
    nulls_dropped = initial_count - non_null_count
    print(f"[*] Purged null rows: {nulls_dropped:,}")

    # Drop duplicate records
    df_clean = df_clean.dropDuplicates()
    deduped_count = df_clean.count()
    dupes_dropped = non_null_count - deduped_count
    print(f"[*] Purged duplicate rows: {dupes_dropped:,}")
    print(f"[*] Cleaned valid row count: {deduped_count:,}")

    # Inspect class balance
    class_counts = df_clean.groupBy("Class").count().collect()
    counts_map = {row["Class"]: row["count"] for row in class_counts}
    legit_count = counts_map.get(0, 0)
    fraud_count = counts_map.get(1, 0)
    fraud_ratio = (fraud_count / deduped_count) * 100 if deduped_count > 0 else 0

    print("-" * 50)
    print(f"    Legitimate Transactions (Class 0): {legit_count:,} ({100 - fraud_ratio:.4f}%)")
    print(f"    Fraudulent Transactions (Class 1): {fraud_count:,} ({fraud_ratio:.4f}%)")
    print(f"    Imbalance Ratio: ~{legit_count // max(1, fraud_count)}:1")
    print("-" * 50)

    # Export to Parquet
    print(f"[*] Writing cleaned Parquet to: {processed_path}")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.write.mode("overwrite").parquet(str(processed_path))

    elapsed = time.time() - start_time
    print(f"[SUCCESS] Ingestion and cleansing complete in {elapsed:.2f}s!")
    print(f"Cleaned output saved: {processed_path}")
    spark.stop()

if __name__ == "__main__":
    main()
