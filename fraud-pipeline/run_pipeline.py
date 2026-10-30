"""
run_pipeline.py - Master Orchestrator for Real-Time Fraud & Graph Analytics Pipeline

Provides a CLI to verify environment readiness and sequentially run any or all
pipeline phases:
    Phase 1: Spark Ingestion, Cleansing & SMOTE Resampling
    Phase 2: MLlib Random Forest Classifier Training & Evaluation
    Phase 3: Kafka Producer & Spark Structured Streaming Micro-Batch Consumer
    Phase 4: PaySim Graph ETL into Neo4j & Cypher Ring Detection
    Phase 5: Metrics & Academic Report Generation
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "notebooks_or_scripts"
PYTHON_BIN = sys.executable

def print_banner(text):
    print("\n" + "=" * 75)
    print(f"  {text}")
    print("=" * 75)

def run_command(cmd, desc):
    print_banner(f"EXECUTION: {desc}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")
    start = time.time()
    res = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.time() - start
    if res.returncode != 0:
        print(f"\n[!] FAILED: {desc} (exit code: {res.returncode}) after {elapsed:.2f}s")
        return False
    print(f"\n[+] SUCCESS: {desc} completed in {elapsed:.2f}s.")
    return True

def check_env():
    print_banner("ENVIRONMENT HEALTH CHECK")
    
    # 1. Python
    print(f"[*] Python Binary: {sys.executable}")
    print(f"[*] Python Version: {sys.version.split()[0]}")

    # 2. Java
    java_home = os.environ.get("JAVA_HOME", "NOT SET")
    print(f"[*] JAVA_HOME: {java_home}")
    try:
        j_ver = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT, text=True)
        first_line = j_ver.strip().split("\n")[0]
        print(f"[+] Java executable detected: {first_line}")
    except Exception as e:
        print(f"[!] Java check warning: {e}")

    # 3. Hadoop & Winutils
    hadoop_home = os.environ.get("HADOOP_HOME")
    if not hadoop_home:
        cand = BASE_DIR.parent / "hadoop"
        if cand.exists():
            hadoop_home = str(cand)
    print(f"[*] HADOOP_HOME: {hadoop_home}")
    if hadoop_home:
        winutils_path = Path(hadoop_home) / "bin" / "winutils.exe"
        if winutils_path.exists():
            print(f"[+] winutils.exe found at: {winutils_path}")
        else:
            print(f"[!] winutils.exe missing at: {winutils_path}")

    # 4. Datasets
    raw_dir = BASE_DIR / "data" / "raw"
    ulb_csv = raw_dir / "creditcard.csv"
    paysim_csv = raw_dir / "PS_20174392719_1491204439457_log.csv"
    print("\n[*] Dataset Status:")
    print(f"    - ULB Credit Card CSV ({ulb_csv}): {'PRESENT' if ulb_csv.exists() else 'MISSING (Action Required)'}")
    print(f"    - PaySim CSV ({paysim_csv}): {'PRESENT' if paysim_csv.exists() else 'MISSING (Action Required)'}")

    # 5. Docker
    print("\n[*] Docker Status:")
    try:
        docker_res = subprocess.check_output(["docker", "ps"], text=True)
        print("[+] Docker Desktop daemon is RUNNING.")
    except Exception:
        print("[!] Docker Desktop is currently STOPPED. (Start Docker Desktop for Kafka/Neo4j)")

def run_phase1():
    print_banner("RUNNING PHASE 1: BATCH PROCESSING & SMOTE")
    if not run_command([PYTHON_BIN, str(SCRIPTS_DIR / "01_ingest_and_clean.py")], "01 Spark Ingest & Clean"):
        return False
    if not run_command([PYTHON_BIN, str(SCRIPTS_DIR / "02_resample_smote.py")], "02 SMOTE Resample & EDA"):
        return False
    return True

def run_phase2b():
    print_banner("RUNNING PHASE 2b: HYBRID SMOTEENN & COST-SENSITIVE MODEL")
    return run_command([PYTHON_BIN, str(SCRIPTS_DIR / "03b_train_hybrid.py")], "03b Hybrid Model Training & Evaluation")

def run_phase4b():
    print_banner("RUNNING PHASE 4b: GRAPH RISK SCORE & CENTRALITY RANKING")
    return run_command([PYTHON_BIN, str(SCRIPTS_DIR / "compute_graph_risk.py")], "Compute Top 20 Graph Risk Scores")

def run_dashboard():
    print_banner("BUILDING EXECUTIVE LIVE DASHBOARD & SCREENSHOT")
    return run_command([PYTHON_BIN, str(SCRIPTS_DIR / "build_dashboard.py")], "Build Live Dashboard & Capture Screenshot")

def run_phase3():
    print_banner("RUNNING PHASE 3: STREAMING INFRASTRUCTURE")
    print("[*] To test streaming, launch the consumer in one terminal:")
    print(f"    {PYTHON_BIN} {SCRIPTS_DIR / '05_spark_streaming_consumer.py'}")
    print("[*] Then launch the producer in a second terminal:")
    print(f"    {PYTHON_BIN} {SCRIPTS_DIR / '04_kafka_producer.py'} --max-records 100")

def run_phase4(uri="bolt://localhost:7687", password="fraudpassword123"):
    print_banner("RUNNING PHASE 4: GRAPH DB EXTENSION")
    ok = run_command([
        PYTHON_BIN, str(SCRIPTS_DIR / "06_load_to_neo4j.py"),
        "--uri", uri, "--password", password
    ], "06 PaySim Graph ETL into Neo4j")
    if ok:
        return run_command([
            PYTHON_BIN, str(SCRIPTS_DIR / "execute_cypher_analytics.py")
        ], "07 Cypher Fraud Ring Analytics")
    return False

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline Runner")
    parser.add_argument("action", choices=["check-env", "phase1", "phase2", "phase2b", "phase3", "phase4", "phase4b", "dashboard", "all"],
                        default="check-env", nargs="?", help="Action or phase to execute")
    args = parser.parse_args()

    if args.action == "check-env":
        check_env()
    elif args.action == "phase1":
        run_phase1()
    elif args.action == "phase2":
        run_phase2()
    elif args.action == "phase2b":
        run_phase2b()
    elif args.action == "phase3":
        run_phase3()
    elif args.action == "phase4":
        run_phase4()
    elif args.action == "phase4b":
        run_phase4b()
    elif args.action == "dashboard":
        run_dashboard()
    elif args.action == "all":
        check_env()
        if run_phase1():
            if run_phase2():
                print("\n[+] Phases 1 & 2 complete! Follow Phase 3 instructions for Kafka streaming demo.")

if __name__ == "__main__":
    main()
