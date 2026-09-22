# Hybrid Real-Time Fraud Detection & Graph Analytics Pipeline

> **Big Data Analytics Course Project — DA1/DA2 Deliverable**  
> *Domain: E-Commerce / Digital Payments / Financial Fraud Detection*

---

## 📌 Executive Summary

Modern financial crime syndicates exploit latency in traditional fraud systems by dispersing stolen funds across multi-tier synthetic identity networks and circular mule accounts ("fraud rings"). Isolated transactional classifiers (e.g., standard ML models) evaluate each transfer in a vacuum, completely missing collusive cyclic structures.

This project implements a **Hybrid Dual-Layer Real-Time Pipeline**:
1. **Batch & Fast-Scoring Layer:** Apache Spark MLlib **Random Forest Classifier** trained on SMOTE-balanced transaction streams, scoring incoming events in real-time micro-batches over **Apache Kafka** via **Spark Structured Streaming**.
2. **Graph Analytics & Ring Layer:** **Neo4j Graph Database** ingesting account relational topologies (`:Account -[:TRANSFER]-> :Account`) to discover multi-hop cyclic rings (2–4 hops), high-degree transit hubs, and smurfing accounts via **Cypher queries**.

---

## 🏗️ Architecture & Pipeline Flow

```
                                  [ Raw Transaction Inflow ]
                                               │
                                 ┌─────────────┴─────────────┐
                                 ▼                           ▼
                     [ULB Credit Card Dataset]      [PaySim Network Dataset]
                                 │                           │
                   01_ingest_and_clean.py                    │
                  (Spark Filter, Schema, Parquet)            │
                                 │                           │
                   02_resample_smote.py                      │
                  (Stratified Split + SMOTE)                 │
                                 │                           │
                  03_train_mllib_model.py                    │
                 (Spark RF Model Training)                   │
                                 │                           │
                                 ▼                           │
                       [models/rf_fraud_model]               │
                                 │                           │
                      04_kafka_producer.py                   │
                      (Test Stream -> Kafka)                 │
                                 │                           │
                                 ▼                           │
                     [Kafka Broker: localhost:9092]          │
                                 │                           │
                 05_spark_streaming_consumer.py              │
                  (Spark Structured Streaming)               │
                                 │                           │
                                 ▼                           │
                   [data/processed/flagged/]                 │
                                 │                           │
                                 └─────────────┬─────────────┘
                                               ▼
                                      06_load_to_neo4j.py
                                  (Graph ETL + :Flagged Tagging)
                                               │
                                               ▼
                                  [Neo4j Property Graph DB]
                                  (bolt://localhost:7687)
                                               │
                                               ▼
                                  07_fraud_ring_queries.cypher
                             (2-4 Hop Cycles, Hubs, Communities)
                                               │
                                               ▼
                                 [Academic Deliverable Report]
```

---

## 📂 Repository Structure

```
fraud-pipeline/
├── data/
│   ├── raw/                       # Downloaded raw datasets (untouched)
│   │   ├── creditcard.csv         # ULB Credit Card Fraud dataset
│   │   └── PS_20174392719_1491204439457_log.csv # PaySim Mobile-Money dataset
│   └── processed/                 # Cleaned and resampled Parquet outputs
│       ├── creditcard_cleaned.parquet
│       ├── train_resampled.parquet # SMOTE balanced training data (1:1)
│       ├── test.parquet           # Pristine held-out test split
│       └── flagged/               # Scored transactions flagged as fraud by streaming
├── notebooks_or_scripts/
│   ├── 01_ingest_and_clean.py     # Spark batch ingestion, cleaning, type casting
│   ├── 02_resample_smote.py       # SMOTE class balancing + EDA figure export
│   ├── 03_train_mllib_model.py    # Spark MLlib VectorAssembler + RandomForestClassifier
│   ├── 04_kafka_producer.py       # Simulated real-time test event streaming to Kafka
│   ├── 05_spark_streaming_consumer.py # Structured Streaming micro-batch ML scoring
│   ├── 06_load_to_neo4j.py        # Relational-to-Graph ETL for PaySim + entity tagging
│   └── 07_fraud_ring_queries.cypher # Cypher queries for circular rings, hubs, and smurfing
├── models/
│   └── rf_fraud_model/            # Exported Spark MLlib PipelineModel
├── reports/
│   ├── metrics.json               # Recorded evaluation metrics (AUC, Precision, Recall, F1)
│   ├── summary.md                 # Academic project report (DA1/DA2 deliverable)
│   └── figures/                   # Generated evaluation plots & screenshots
│       ├── class_balance.png      # Before/after SMOTE class distribution
│       ├── amount_distribution.png# Log-scaled transaction amount density
│       ├── correlation_heatmap.png# Top feature correlations with fraud label
│       ├── confusion_matrix.png   # MLlib model confusion matrix
│       ├── roc_curve.png          # Receiver Operating Characteristic curve
│       ├── feature_importance.png # Top 15 Random Forest feature importances
│       └── neo4j_fraud_ring.png   # Neo4j Browser graph visualization screenshot
├── docker-compose.yml             # Single-node KRaft Kafka broker + Neo4j Community DB
├── requirements.txt               # Pinned Python package dependencies
├── run_pipeline.py                # Master CLI orchestrator
└── README.md                      # Comprehensive project guide
```

---

## ⚙️ Environment Prerequisites & Setup

### 1. Requirements
- **Python**: 3.10 or 3.11
- **Java**: Java 17 LTS (configured in `JAVA_HOME`)
- **Docker**: Docker Desktop (for Kafka & Neo4j containers)

### 2. Virtual Environment Installation
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 3. Datasets Setup
Download and place the two required datasets in `data/raw/`:
- **ULB Credit Card Fraud**: Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) $\rightarrow$ `data/raw/creditcard.csv`
- **PaySim Synthetic Financial Network**: Download from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) $\rightarrow$ `data/raw/PS_20174392719_1491204439457_log.csv`

---

## 🚀 Execution Guide (Phase by Phase)

### Phase 1: Ingestion, Cleansing & SMOTE Balancing
```powershell
python notebooks_or_scripts/01_ingest_and_clean.py
python notebooks_or_scripts/02_resample_smote.py
```
- Ingests raw CSV into a PySpark DataFrame, purges missing values and duplicates.
- Generates `creditcard_cleaned.parquet`.
- Splits data 80/20 into train/test, applies SMOTE on the training split to produce `train_resampled.parquet`, and preserves the pristine `test.parquet`.
- Generates EDA plots in `reports/figures/`.

### Phase 2: Spark MLlib Model Training & Evaluation
```powershell
python notebooks_or_scripts/03_train_mllib_model.py
```
- Trains a Spark MLlib `RandomForestClassifier(numTrees=100, maxDepth=10)`.
- Evaluates against the held-out test split using `BinaryClassificationEvaluator`.
- Saves the trained `PipelineModel` to `models/rf_fraud_model/`.
- Records evaluation metrics in `reports/metrics.json` and exports `confusion_matrix.png`, `roc_curve.png`, and `feature_importance.png`.

### Phase 3: Real-Time Kafka Streaming & Scoring
1. **Start Kafka and Neo4j via Docker Compose:**
   ```powershell
   docker compose up -d
   ```
2. **Terminal 1 — Launch Spark Structured Streaming Consumer:**
   ```powershell
   python notebooks_or_scripts/05_spark_streaming_consumer.py
   ```
3. **Terminal 2 — Launch Kafka Transaction Producer:**
   ```powershell
   python notebooks_or_scripts/04_kafka_producer.py --max-records 200 --delay 0.05
   ```
- Producer streams test records to topic `transactions`.
- Structured Streaming consumer loads `models/rf_fraud_model/`, processes incoming micro-batches, prints real-time alerts to the terminal, and writes flagged events to `data/processed/flagged/`.

### Phase 4: Neo4j Graph DB & Ring Analytics
1. **Load PaySim Network into Neo4j:**
   ```powershell
   python notebooks_or_scripts/06_load_to_neo4j.py --sample-size 50000
   ```
2. **Run Cypher Fraud Ring Queries:**
   - Open Neo4j Browser at `http://localhost:7474` (Credentials: `neo4j` / `fraudpassword123`).
   - Run the Cypher queries in `notebooks_or_scripts/07_fraud_ring_queries.cypher`.
   - Inspect detected cyclic rings (`MATCH path = (a:Account)-[:TRANSFER*2..4]->(a) RETURN path LIMIT 25`).

---

## 📊 Summary of Model Performance

| Metric | Target / Benchmark |
| :--- | :--- |
| **Model** | Spark MLlib `RandomForestClassifier` (100 Trees) |
| **Resampling Strategy** | SMOTE (1:1 Balanced Training Split) |
| **Evaluation Split** | Held-Out Test Set (Uncorrupted Pristine Imbalance) |
| **Area Under ROC (AUC-ROC)** | Evaluated in `reports/metrics.json` |
| **Area Under PR (PR-AUC)** | Evaluated in `reports/metrics.json` |
| **Prioritized Objective** | High Recall & Precision over raw Accuracy |

---

## 👥 Contributors & Course Info
- **Project**: Dynamic UPI & Real-Time Financial Fraud Detection Pipeline
- **Course**: Big Data Analytics (DA1 / DA2 Course Project)
