# Dynamic UPI & Financial Fraud Ring Detection Pipeline

> **Big Data Analytics Course Project — DA1/DA2 Deliverable**  
> *Hybrid Real-Time Ingestion, Spark MLlib Scoring, Kafka Micro-batching & Neo4j Graph Ring Analytics*

Please see the full documentation and execution guide inside [fraud-pipeline/README.md](file:///d:/VS%20code/college/big_data/Dynamic-UPI-Fraud-Ring-Detection-using-Temporal-Graph-Networks/fraud-pipeline/README.md).

### Quick Start:
```powershell
# 1. Environment & Dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Check Environment Status
python run_pipeline.py check-env

# 3. Download Kaggle Datasets to data/raw/ or fraud-pipeline/data/raw/
# creditcard.csv and PS_20174392719_1491204439457_log.csv

# 4. Run Phases
python run_pipeline.py phase1  # PySpark Ingest, Clean, SMOTE Resample & EDA
python run_pipeline.py phase2  # Spark MLlib Random Forest Training & Evaluation
# Phase 3 (Streaming) and Phase 4 (Neo4j Graph ETL) details in fraud-pipeline/README.md
```