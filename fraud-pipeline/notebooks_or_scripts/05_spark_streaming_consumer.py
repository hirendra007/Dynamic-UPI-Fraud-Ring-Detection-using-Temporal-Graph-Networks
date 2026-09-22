"""
05_spark_streaming_consumer.py - Real-Time Micro-Batch Scoring with Spark Structured Streaming

Subscribes to the Kafka 'transactions' topic, parses streaming JSON payloads,
applies the trained Spark MLlib Random Forest model to each micro-batch in real-time,
filters flagged fraudulent transactions (prediction == 1.0), and writes alerts to
both console output and data/processed/flagged/.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Setup Windows environment
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

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

def get_paths():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    model_path = base_dir / "models" / "rf_fraud_model"
    flagged_path = base_dir / "data" / "processed" / "flagged"
    checkpoint_path = base_dir / "data" / "processed" / "checkpoints" / "streaming"

    if not model_path.exists() and (script_dir.parent.parent / "models" / "rf_fraud_model").exists():
        base_dir = script_dir.parent.parent
        model_path = base_dir / "models" / "rf_fraud_model"
        flagged_path = base_dir / "data" / "processed" / "flagged"
        checkpoint_path = base_dir / "data" / "processed" / "checkpoints" / "streaming"

    flagged_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return model_path, flagged_path, checkpoint_path

def main():
    parser = argparse.ArgumentParser(description="Spark Structured Streaming Fraud Detection Consumer")
    parser.add_argument("--bootstrap-servers", default="localhost:9092", help="Kafka broker address")
    parser.add_argument("--topic", default="transactions", help="Kafka topic to consume")
    args = parser.parse_args()

    model_path, flagged_path, checkpoint_path = get_paths()

    print("=" * 70)
    print("PHASE 3: Spark Structured Streaming Real-Time Scoring Consumer")
    print("=" * 70)

    if not model_path.exists():
        print(f"[ERROR] Trained ML model not found at: {model_path}")
        print("Please run 03_train_mllib_model.py first.")
        sys.exit(1)

    print(f"[*] Initializing Spark Session with Kafka Connector (v3.5.1)...")
    spark = SparkSession.builder \
        .appName("FraudPipeline-StreamingConsumer") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
        .config("spark.sql.streaming.checkpointLocation", str(checkpoint_path)) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print(f"[*] Loading trained PipelineModel from: {model_path}")
    model = PipelineModel.load(str(model_path))
    print("[+] Model loaded successfully!")

    # Define Schema for incoming Kafka JSON payload
    fields = [
        StructField("tx_id", StringType(), True),
        StructField("timestamp", DoubleType(), True),
        StructField("Time", DoubleType(), True),
        StructField("Amount", DoubleType(), True),
        StructField("Class", IntegerType(), True)
    ]
    for i in range(1, 29):
        fields.append(StructField(f"V{i}", DoubleType(), True))
    json_schema = StructType(fields)

    print(f"[*] Connecting Structured Stream to Kafka: {args.bootstrap_servers}, topic: {args.topic}")
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", args.bootstrap_servers) \
        .option("subscribe", args.topic) \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()

    # Deserialize JSON from Kafka value column
    parsed_stream = raw_stream \
        .selectExpr("CAST(value AS STRING) as json_payload") \
        .select(from_json(col("json_payload"), json_schema).alias("data")) \
        .select("data.*")

    # Define Micro-Batch Processor
    batch_counter = [0]
    total_flagged_count = [0]

    def process_micro_batch(batch_df, batch_id):
        batch_counter[0] += 1
        count = batch_df.count()
        if count == 0:
            return

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] --- Processing Micro-Batch #{batch_id} ({count} transactions) ---")

        # Apply Spark MLlib PipelineModel
        scored_batch = model.transform(batch_df)

        # Filter fraudulent predictions
        flagged_batch = scored_batch.filter(col("prediction") == 1.0)
        flagged_count = flagged_batch.count()
        total_flagged_count[0] += flagged_count

        print(f"    Scored: {count} | Flagged as Fraud: {flagged_count} (Total Flagged so far: {total_flagged_count[0]})")

        if flagged_count > 0:
            print("\n    >>> [REAL-TIME FRAUD ALERTS DETECTED] <<<")
            alert_records = flagged_batch.select("tx_id", "Amount", "prediction", "Class").toPandas()
            for _, r in alert_records.iterrows():
                ground_truth = "MATCH (Actual Fraud)" if r["Class"] == 1 else "FALSE ALARM (False Positive)"
                print(f"    * ALERT: ID={r['tx_id']} | Amount=${r['Amount']:.2f} | Status=FRAUD FLAGGED | Ground Truth: {ground_truth}")

            # Write flagged transactions to persistent Parquet sink
            flagged_output = flagged_path / f"batch_{batch_id}.parquet"
            flagged_batch.drop("features", "rawPrediction", "probability") \
                .write.mode("append").parquet(str(flagged_output))
            print(f"    Saved flagged records to: {flagged_output}")

    print("[*] Starting Structured Streaming Query with 2-second micro-batch trigger...")
    print("    Awaiting incoming transactions from producer (Press Ctrl+C to terminate)...\n")

    query = parsed_stream.writeStream \
        .foreachBatch(process_micro_batch) \
        .trigger(processingTime="2 seconds") \
        .start()

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n[!] Stopping streaming consumer...")
        query.stop()
        spark.stop()
        print("[+] Streaming consumer gracefully halted.")

if __name__ == "__main__":
    main()
