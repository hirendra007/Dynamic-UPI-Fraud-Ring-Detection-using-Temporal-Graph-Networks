"""
04_kafka_producer.py - Real-Time Transaction Streaming Producer

Reads the held-out test dataset (or streaming sample) and publishes each
transaction row as a JSON message to an Apache Kafka topic ('transactions')
at a controlled rate to simulate live credit card/UPI transactions.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import pandas as pd

def get_kafka_producer(bootstrap_servers):
    # Support both kafka-python and kafka-python-ng
    try:
        from kafka import KafkaProducer
    except ImportError:
        print("[ERROR] kafka module not found. Install kafka-python-ng or kafka-python.")
        sys.exit(1)

    print(f"[*] Connecting to Kafka broker at {bootstrap_servers}...")
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3
            )
            print(f"[+] Successfully connected to Kafka on attempt {attempt}!")
            return producer
        except Exception as e:
            print(f"[!] Kafka connection attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                raise

def get_test_data_path():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    test_path = base_dir / "data" / "processed" / "test.parquet"
    if not test_path.exists() and (script_dir.parent.parent / "data" / "processed" / "test.parquet").exists():
        test_path = script_dir.parent.parent / "data" / "processed" / "test.parquet"
    return test_path

def main():
    parser = argparse.ArgumentParser(description="Kafka Real-Time Transaction Producer")
    parser.add_argument("--bootstrap-servers", default="localhost:9092", help="Kafka bootstrap broker(s)")
    parser.add_argument("--topic", default="transactions", help="Target Kafka topic")
    parser.add_argument("--delay", type=float, default=0.05, help="Inter-message delay in seconds (default: 0.05)")
    parser.add_argument("--max-records", type=int, default=500, help="Maximum records to stream (default: 500)")
    parser.add_argument("--interleave-fraud", action="store_true", default=True,
                        help="Ensure fraud transactions are regularly interleaved for live demonstration")
    args = parser.parse_args()

    test_path = get_test_data_path()
    if not test_path.exists():
        print(f"[ERROR] Held-out test dataset not found at: {test_path}")
        print("Please run 02_resample_smote.py first.")
        sys.exit(1)

    print("=" * 70)
    print("PHASE 3: Kafka Real-Time Transaction Producer")
    print("=" * 70)
    print(f"[*] Loading test transactions from: {test_path}")
    df = pd.read_parquet(test_path)
    print(f"[*] Loaded {len(df):,} test records.")

    # Interleave fraud records to guarantee live alerts during demos
    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]
    print(f"[*] Test pool contains {len(fraud_df):,} actual frauds and {len(legit_df):,} legits.")

    if args.interleave_fraud and len(fraud_df) > 0:
        # Sample so fraud appears regularly (every ~10-15 transactions)
        records_to_stream = []
        fraud_pool = fraud_df.to_dict(orient="records")
        legit_pool = legit_df.to_dict(orient="records")
        f_idx = 0
        l_idx = 0
        
        while len(records_to_stream) < min(args.max_records, len(df)):
            for _ in range(12):
                if l_idx < len(legit_pool):
                    records_to_stream.append(legit_pool[l_idx])
                    l_idx += 1
            if f_idx < len(fraud_pool):
                records_to_stream.append(fraud_pool[f_idx])
                f_idx += 1
            if l_idx >= len(legit_pool) and f_idx >= len(fraud_pool):
                break
    else:
        records_to_stream = df.head(args.max_records).to_dict(orient="records")

    print(f"[*] Prepared stream buffer: {len(records_to_stream)} transactions")

    producer = get_kafka_producer(args.bootstrap_servers)

    print(f"\n[*] Starting real-time dispatch to topic '{args.topic}' (delay={args.delay}s)...")
    print(f"{'TX ID':<10} | {'AMOUNT ($)':<12} | {'GROUND TRUTH':<15} | {'STATUS'}")
    print("-" * 55)

    sent_count = 0
    fraud_sent = 0

    try:
        for idx, row in enumerate(records_to_stream, 1):
            tx_id = f"TX-{int(time.time()*1000) % 10000000:07d}-{idx}"
            is_fraud = int(row.get("Class", 0))
            amount = float(row.get("Amount", 0.0))

            message = {
                "tx_id": tx_id,
                "timestamp": time.time(),
                "Time": float(row.get("Time", 0.0)),
                "Amount": amount,
                "Class": is_fraud
            }
            for i in range(1, 29):
                message[f"V{i}"] = float(row.get(f"V{i}", 0.0))

            producer.send(args.topic, value=message)
            sent_count += 1

            if is_fraud == 1:
                fraud_sent += 1
                status = ">>> [HIGH RISK DISPATCHED] <<<"
            else:
                status = "Dispatched"

            print(f"{tx_id:<10} | {amount:>10.2f}   | {'FRAUD (1)' if is_fraud else 'Legit (0)':<15} | {status}")

            time.sleep(args.delay)

        producer.flush()
        print("\n" + "=" * 55)
        print(f"[+] Finished streaming {sent_count} transactions!")
        print(f"    Legitimate sent: {sent_count - fraud_sent}")
        print(f"    Fraudulent sent: {fraud_sent}")
        print("=" * 55)

    except KeyboardInterrupt:
        print("\n[!] Streaming interrupted by user.")
    finally:
        producer.close()
        print("[*] Kafka producer connection closed.")

if __name__ == "__main__":
    main()
