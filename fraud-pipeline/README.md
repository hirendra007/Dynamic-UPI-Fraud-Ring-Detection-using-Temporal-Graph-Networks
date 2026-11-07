# 🛡️ Dynamic UPI & Financial Fraud Ring Detection using Temporal Graph Networks & Spark MLlib

> **Big Data Analytics Course Capstone Project — DA1/DA2 Deliverable**  
> *Domain: E-Commerce, Unified Payments Interface (UPI), Digital Payments & Financial Crime Detection*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5%20MLlib-E25A1C.svg)](https://spark.apache.org/)
[![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-Distributed%20Streaming-231F20.svg)](https://kafka.apache.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x%20Graph%20Database-008CC1.svg)](https://neo4j.com/)
[![Imbalanced-Learn](https://img.shields.io/badge/imbalanced--learn-SMOTEENN-orange.svg)](https://imbalanced-learn.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analytics-blueviolet.svg)](https://networkx.org/)
[![Vis.js](https://img.shields.io/badge/Vis.js-Interactive%20Physics%20Graph-brightgreen.svg)](https://visjs.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 Executive Summary

Modern financial crime syndicates systematically exploit latency in traditional, isolated fraud systems by dispersing illicit capital across multi-tier synthetic identities, transit mules, and circular laundering syndicates (**"fraud rings"**). Conventional tabular machine learning models evaluate transactions as independent, isolated events, leaving an inherent blind spot for collusive structural patterns such as **smurfing** (fan-in aggregation) and **multi-hop cyclical money loops**.

This project establishes a production-grade, end-to-end **Dual-Layer Real-Time Big Data Architecture**:

1. **Transaction Fast-Scoring Layer (Apache Spark MLlib + Kafka Streaming)**:
   - Implements a **Hybrid Imbalance-Handling Engine (`03b_train_hybrid.py`)** combining **SMOTEENN** boundary cleaning, **cost-sensitive loss weighting** ($w_{\text{fraud}} = 4.7233$), and **monetary loss threshold optimization** ($T^* = 0.54$) evaluated on a strict holdout test split of $56,746$ transactions.
   - Deploys **Apache Kafka** and **Spark Structured Streaming** to score incoming transactions in low-latency micro-batches against the trained MLlib PipelineModel.

2. **Graph Network Analytics Layer (Neo4j Property Graph + NetworkX)**:
   - Maps $96,419$ accounts and $50,007$ directed transfer relationships into **Neo4j** to uncover hidden topological structures via **Cypher pattern matching**.
   - Computes a **Composite Graph Risk Score** combining circular ring membership ($40\%$), PageRank flow centrality ($30\%$), and in-degree fan-in accumulation ($30\%$) to rank the **Top 20 high-risk syndicate entities**.

3. **Executive Visual Interfaces**:
   - **Interactive Fullscreen Graph Explorer (`graph_explorer.html`)**: Physics-enabled force-directed canvas (Vis.js ForceAtlas2) with draggable nodes, continuous mouse-wheel zoom, live account ID search, inspector HUD, and high-contrast color palette.
   - **Live Executive Dashboard (`dashboard.html` / `dashboard_app.py`)**: Zero-dependency standalone HTML and Streamlit dashboard with dynamic KPI cards, side-by-side confusion matrices, cost-optimization curves, and embedded interactive topologies.

---

## 📊 Key Results & Empirical Benchmark

Every metric reported in this project is calculated from an actual execution and saved to disk with **zero fabricated numbers**.

### Baseline vs. Hybrid Model Performance (Test Split: 56,746 Rows)

| Metric | Baseline Model (SMOTE 1:1, T=0.50) | Hybrid Model (SMOTEENN+Cost, T=0.54) | Impact & Operational Benefit |
| :--- | :---: | :---: | :--- |
| **Resampling Strategy** | Standard SMOTE (1:1) | **SMOTEENN (Boundary Cleaned)** | Eliminates noisy majority outliers |
| **Class Weighting** | Uniform (1.0 : 1.0) | **Cost-Sensitive ($w=4.7233$)** | $4.72\times$ penalty on missed fraud |
| **True Positives (TP)** | 76 | **75** | Maintains high fraud capture ($78.95\%$ recall) |
| **False Positives (FP)** | 70 | **51** | **-19 False Alarms (-27.14% reduction)** |
| **False Negatives (FN)** | 19 | **20** | Only 1 additional missed fraud |
| **True Negatives (TN)** | 56,581 | **56,600** | +19 genuine transactions uninterrupted |
| **Test Precision** | 52.05% | **59.52%** | **+7.47% absolute gain** in alert accuracy |
| **Area Under ROC (AUC)** | 0.9483 | **0.9644** | **+0.0161 higher overall discrimination** |
| **F1-Score** | 0.6307 | **0.6787** | **+0.0480 improvement** in harmonic balance |
| **Expected Monetary Loss** | \$1,329.99 | **\$1,246.62** | **-\$83.37 net financial savings (+6.27%)** |

### Neo4j Graph Topological Discovery Summary

- **Total Graph Size**: $96,419$ `:Account` nodes | $50,007$ `:TRANSFER` directed edges.
- **Circular Fraud Rings Detected**: 7 circular loops across 2 syndicates:
  - **Alpha Ring (4 Hops)**: `RING_ALPHA_01 -> 02 -> 03 -> 04 -> 01` &bull; **\$1,740,000.00 volume**
  - **Beta Ring (3 Hops)**: `RING_BETA_01 -> 02 -> 03 -> 01` &bull; **\$540,000.00 volume**
- **Smurfing Hubs Flagged**: 13 accumulator accounts with in-degree $\ge 4$ (e.g., account `C141316671` with 5 incoming deposits and 0 outgoing transactions).
- **Top 20 Graph Risk Ranking**: Surfaced in `reports/graph_risk_top20.csv` with scores up to $0.6154$.

---

## 🏗️ End-to-End System Architecture

```
                                  [ RAW DATA INGESTION ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          [ULB Credit Card Dataset]                   [PaySim Financial Network]
           (283,726 rows, 0.1667% fraud)               (6.3M logs -> 50,007 sample)
                       │                                           │
                       ▼                                           │
           [01_ingest_and_clean.py]                                │
          (PySpark Schema, Null Check, Parquet)                    │
                       │                                           │
         ┌─────────────┴─────────────┐                             │
         ▼                           ▼                             │
 [02_resample_smote.py]    [03b_train_hybrid.py]                   │
(Baseline SMOTE 1:1 Split) (SMOTEENN + Cost Weights + T=0.54)      │
         │                           │                             │
         ▼                           ▼                             │
[03_train_mllib_model.py]  [models/rf_hybrid_fraud_model]          │
(Baseline MLlib RF Model)            │                             │
                                     ▼                             │
                          [04_kafka_producer.py]                   │
                          (Streams Test.parquet)                   │
                                     │                             │
                                     ▼                             │
                         [Kafka Topic: transactions]               │
                                     │                             │
                                     ▼                             │
                      [05_spark_streaming_consumer.py]             │
                       (Spark Structured Streaming)                │
                                     │                             │
                                     ▼                             │
                         [data/processed/flagged/]                 │
                                     │                             │
                                     └──────────────┬──────────────┘
                                                    ▼
                                           [06_load_to_neo4j.py]
                                       (Graph ETL + Benchmark Rings)
                                                    │
                                                    ▼
                                         [Neo4j Graph Database]
                                          (bolt://localhost:7687)
                                                    │
                        ┌───────────────────────────┴───────────────────────────┐
                        ▼                                                       ▼
        [execute_cypher_analytics.py]                                 [compute_graph_risk.py]
         (Cypher 2-4 Hop Cycle Queries)                              (Composite Graph Risk Formula)
                        │                                                       │
                        ▼                                                       ▼
        [reports/figures/neo4j_fraud_ring.png]                        [reports/graph_risk_top20.csv]
                        │                                                       │
                        └───────────────────────────┬───────────────────────────┘
                                                    ▼
                                          [build_dashboard.py]
                               (Dynamic Math & Zero Hardcoded Artifacts)
                                                    │
                               ┌────────────────────┴────────────────────┐
                               ▼                                         ▼
                     [graph_explorer.html]                        [dashboard.html]
                  (Fullscreen Interactive App)                  (Executive Dashboard)
                               │                                         │
                               └────────────────────┬────────────────────┘
                                                    ▼
                                            [dashboard_app.py]
                                          (Live Streamlit Web App)
```

---

## 🎨 High-Contrast Visual Topology & Color Identity

The graph visualization across the **interactive explorer** (`graph_explorer.html`), the **executive dashboard** (`dashboard.html`), and the **static publication figure** (`reports/figures/neo4j_fraud_ring.png`) uses a high-contrast palette compliant with WCAG AAA visual standards:

| Entity Role / Element | Hex Color Code | Geometry & Styling | Visual Purpose & Distinction |
| :--- | :---: | :---: | :--- |
| **Alpha Ring (4-Hop)** | `#ff0055` | 30px Circle, 2.0px Crisp White Border | Neon Crimson salience for the \$1.74M circular laundering syndicate. |
| **Beta Ring (3-Hop)** | `#ffd600` | 28px Circle, 2.0px Contrast Black Border | Electric Gold with dark border to separate it from Alpha rings. |
| **Smurfing Mule Hubs** | `#00f0ff` | 26px **Diamond (`shape: 'diamond'`)** | Distinct geometric shape and Electric Cyan color isolate rapid deposit hubs. |
| **Transit Mules** | `#d500f9` | 20px Circle, 2.0px White Border | Bright Magenta provides high luminance for layering intermediaries. |
| **Counterparty Accounts** | `#38bdf8` | 16px Circle, 2.0px White Border | Vivid Sky Blue clearly marks peripheral endpoints. |
| **Transfer Edges** | `#94a3b8` | **2.0px width, 0.85 opacity**, directed arrow | Slate Silver provides strong contrast over dark backgrounds. |
| **Fraud Syndicate Edges**| `#ff0055` / `#ffd600` | **4.0px–4.5px thick curved lines** | Heavy strokes trace the path of illicit funds through the syndicates. |
| **Canvas Background** | `#0f172a` &rarr; `#030712` | Deep Slate to Pitch Black Radial | High-contrast backdrop; includes a `🌓 Toggle Canvas Tone` button. |

---

## 📂 Repository File Structure

```
Dynamic-UPI-Fraud-Ring-Detection-using-Temporal-Graph-Networks/
├── data/
│   ├── raw/                              # Untouched raw datasets
│   │   ├── creditcard.csv                # Kaggle ULB Credit Card Fraud dataset
│   │   └── PS_20174392719_..._log.csv    # Kaggle PaySim financial logs dataset
│   └── processed/                        # Processed Parquet data
│       ├── creditcard_cleaned.parquet    # Cleaned, schema-casted PySpark DataFrame
│       ├── train_resampled.parquet       # SMOTE balanced training data (1:1)
│       ├── test.parquet                  # Pristine held-out test split (56,746 rows)
│       └── flagged/                      # Real-time flagged streaming alerts
├── fraud-pipeline/                       # Core pipeline codebase
│   ├── notebooks_or_scripts/
│   │   ├── 01_ingest_and_clean.py        # Spark ingestion, type casting & Parquet conversion
│   │   ├── 02_resample_smote.py          # Stratified split & baseline SMOTE 1:1
│   │   ├── 03_train_mllib_model.py       # Spark MLlib baseline RandomForestClassifier
│   │   ├── 03b_train_hybrid.py           # SMOTEENN + Cost Weights + Threshold Optimization
│   │   ├── 04_kafka_producer.py          # Real-time Kafka event streaming producer
│   │   ├── 05_spark_streaming_consumer.py# Spark Structured Streaming micro-batch ML consumer
│   │   ├── 06_load_to_neo4j.py           # Neo4j graph loader with controlled benchmark rings
│   │   ├── 07_fraud_ring_queries.cypher  # Production Cypher queries for ring detection
│   │   ├── execute_cypher_analytics.py   # Automated Cypher query runner & NetworkX visualizer
│   │   ├── compute_graph_risk.py         # Computes PageRank, In-Degree, Ring Scores
│   │   └── build_dashboard.py            # Dynamic HTML builder & headless screenshot capture
│   ├── models/
│   │   ├── rf_fraud_model/               # Saved Baseline Spark MLlib PipelineModel
│   │   └── rf_hybrid_fraud_model/        # Saved Hybrid SMOTEENN Spark MLlib PipelineModel
│   ├── reports/
│   │   ├── metrics.json                  # Baseline evaluation metrics & Neo4j graph summary
│   │   ├── metrics_hybrid.json           # Hybrid ML evaluation metrics & baseline comparison
│   │   ├── graph_risk_top20.csv          # Ranked Top 20 high-risk accounts from Neo4j
│   │   ├── graph_interactive.json        # Connected subnetwork topology for Vis.js
│   │   └── figures/                      # High-resolution generated plots & screenshots
│   │       ├── confusion_matrix_hybrid.png  # Confusion matrix plot for hybrid model
│   │       ├── threshold_cost_curve.png     # Validation cost optimization curve (T=0.54)
│   │       ├── roc_pr_comparison.png        # ROC & Precision-Recall curves (Base vs Hybrid)
│   │       ├── neo4j_fraud_ring.png         # High-contrast static graph visualization
│   │       ├── dashboard_live.png           # Full-page screenshot of dashboard
│   │       └── vis-network.min.js           # Offline standalone Vis.js network library
│   ├── docker-compose.yml                # Single-node KRaft Kafka broker + Neo4j Community DB
│   ├── requirements.txt                  # Pinned Python package dependencies
│   ├── run_pipeline.py                   # Master CLI pipeline orchestrator
│   └── SYSTEM_UPDATES_AND_AUDIT_REPORT.md# Exhaustive technical report & zero-hardcoding audit
├── dashboard.html                        # Standalone executive HTML dashboard (mirrored to root)
├── graph_explorer.html                   # Standalone interactive graph explorer (mirrored to root)
├── dashboard_app.py                      # Live reactive Streamlit executive dashboard
├── run_pipeline.py                       # Root launcher pointing to master orchestrator
├── SYSTEM_UPDATES_AND_AUDIT_REPORT.md    # Master technical report & integrity audit
└── README.md                             # THIS MASTER DOCUMENT
```

---

## ⚙️ Prerequisites & Environment Setup

### 1. System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version `3.10` or `3.11`
- **Java**: Java 17 LTS (configured in `JAVA_HOME`)
- **Docker**: Docker Desktop (for Kafka & Neo4j containers)

### 2. Virtual Environment Installation
```powershell
# Clone the repository
git clone https://github.com/<your-username>/Dynamic-UPI-Fraud-Ring-Detection-using-Temporal-Graph-Networks.git
cd Dynamic-UPI-Fraud-Ring-Detection-using-Temporal-Graph-Networks

# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# Install pinned dependencies
pip install -r fraud-pipeline/requirements.txt
```

### 3. Datasets Setup
Download and place the two Kaggle datasets in `fraud-pipeline/data/raw/` (or `data/raw/`):
1. **ULB Credit Card Fraud**: Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) $\rightarrow$ `data/raw/creditcard.csv`
2. **PaySim Synthetic Financial Network**: Download from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) $\rightarrow$ `data/raw/PS_20174392719_1491204439457_log.csv`

### 4. Verify Environment Readiness
```powershell
python run_pipeline.py check-env
```

---

## 🚀 Step-by-Step Execution Guide

You can run each phase individually using the master CLI or via the underlying scripts:

### Phase 1: Ingestion, Cleaning & SMOTE Resampling
```powershell
python run_pipeline.py phase1
```
- Deduplicates, verifies zero nulls, casts data types via PySpark, and outputs `creditcard_cleaned.parquet`.
- Applies a stratified 80/20 train/test split (`random_state=42`) and SMOTE resampling on the training portion to produce `train_resampled.parquet`, preserving the untouched `test.parquet`.

### Phase 2: Baseline Spark MLlib Random Forest
```powershell
python run_pipeline.py phase2
```
- Trains a Spark MLlib `RandomForestClassifier(numTrees=100, maxDepth=10)` on `train_resampled.parquet`.
- Evaluates against the held-out test split, exports the `PipelineModel` to `models/rf_fraud_model/`, and saves baseline metrics to `reports/metrics.json`.

### Phase 2b: Hybrid SMOTEENN + Cost-Sensitive Model Training
```powershell
python run_pipeline.py phase2b
```
- Carves a dedicated $20\%$ validation split out of the training portion.
- Applies **SMOTEENN** (`SMOTE(sampling_strategy=0.2)` + `EditedNearestNeighbours(sampling_strategy='majority')`) to prune noisy boundary majority points.
- Applies cost-sensitive weighting ($w_{\text{fraud}} = 4.7233$) and trains Spark MLlib `RandomForestClassifier` with `weightCol`.
- Evaluates 91 probability thresholds on the validation split minimizing:
  $$\text{Cost}(T) = \sum \text{Amount of Missed Frauds (FN)} + (\$5.00 \times \text{False Positives})$$
  Finds optimal threshold **$T^* = 0.54$** and evaluates once on the test split.
- Saves results to `reports/metrics_hybrid.json` and figures to `reports/figures/`.

### Phase 3: Real-Time Kafka Streaming & Scoring
1. **Start Kafka and Neo4j Containers:**
   ```powershell
   docker compose -f fraud-pipeline/docker-compose.yml up -d
   ```
2. **Terminal 1 — Launch Spark Structured Streaming Consumer:**
   ```powershell
   python fraud-pipeline/notebooks_or_scripts/05_spark_streaming_consumer.py
   ```
3. **Terminal 2 — Launch Kafka Streaming Transaction Producer:**
   ```powershell
   python fraud-pipeline/notebooks_or_scripts/04_kafka_producer.py --max-records 150 --delay 0.05
   ```
- The consumer loads the MLlib model, intercepts micro-batches from Kafka topic `transactions`, classifies fraud in real-time, and sinks flagged alerts to `data/processed/flagged/`.

### Phase 4: Neo4j Graph ETL & Cypher Fraud Ring Detection
1. **Load PaySim Network into Neo4j:**
   ```powershell
   python run_pipeline.py phase4
   ```
2. **Execute Automated Cypher Analytics & NetworkX Plotting:**
   - Runs queries in `07_fraud_ring_queries.cypher`, detects multi-hop circular rings, and exports high-contrast `reports/figures/neo4j_fraud_ring.png`.
3. **Compute Composite Graph Risk Scores:**
   ```powershell
   python run_pipeline.py phase4b
   ```
   - Calculates in-degree, out-degree, PageRank ($\alpha=0.85$), and ring membership across all 96,419 accounts. Saves the top 20 ranked entities to `reports/graph_risk_top20.csv`.

---

## 🖥️ Launching the Interactive Dashboards

### 1. Dedicated Fullscreen Interactive Graph Explorer (`graph_explorer.html`)
```powershell
python run_pipeline.py graph-explorer
# OR directly in your browser:
start graph_explorer.html
```
- **Force-Directed Physics (ForceAtlas2)**: Drag nodes, pan, and zoom smoothly.
- **Freeze Physics**: Click `⏸️ Freeze Physics` to lock nodes in place.
- **One-Click Filtering**: Click `🔴 4-Hop Alpha Ring`, `🟡 3-Hop Beta Ring`, or `🔷 Mule Hubs` to isolate syndicates while dimming unrelated accounts.
- **Live Search**: Enter any account ID (e.g., `C141316671`) to auto-zoom and open the details card.
- **Canvas Tone Toggle**: Switch between radial slate and pitch-dark canvas using `🌓 Toggle Canvas Tone`.

### 2. Standalone Executive HTML Dashboard (`dashboard.html`)
```powershell
start dashboard.html
```
- Displays live KPI cards, embedded interactive Vis.js graph canvas, Top 20 risk score table, side-by-side confusion matrices, and validation cost curves.

### 3. Live Reactive Streamlit Application (`dashboard_app.py`)
```powershell
streamlit run dashboard_app.py
```
- Reactive multi-column web app rendering live metrics from disk, data tables with gradient heatmaps, and embedded interactive topology widgets.

---

## 🎓 Academic Viva Defense & Presentation Guide

When presenting this project for evaluation, highlight these key technical principles:

### 1. "Why SMOTEENN instead of basic SMOTE?"
> *"Standard SMOTE forces artificial 1:1 balance by synthesizing minority points between arbitrary neighbors. In severe 0.17% fraud imbalance, this places synthetic fraud points inside legitimate clusters, creating noisy boundaries and 70 false alarms. SMOTEENN synthesizes fraud instances conservatively (up to 20% of majority) and then applies Edited Nearest Neighbours to prune ambiguous majority instances. Combined with cost-sensitive class weighting ($w=4.7233$) and validation threshold optimization ($T=0.54$), this reduced false alarms by 27.14% (51 vs 70), increased precision to 59.52%, and achieved an Area Under ROC of 0.9644 on untouched test data."*

### 2. "Why combine Tabular Machine Learning with Neo4j Graph Analytics?"
> *"Tabular ML models evaluate transactions in isolation based on numerical features, making them blind to relational money laundering tactics like smurfing and multi-hop layering. By modeling transactions as a directed graph in Neo4j, we compute in-degree, PageRank, and circular ring paths. Combining these into a Composite Graph Risk Score ($0.4 \times \text{Ring} + 0.3 \times \text{PageRank} + 0.3 \times \text{In-Degree}$) successfully flagged both the \$1.74M circular syndicate and the high-volume fan-in smurfing hubs that tabular models miss."*

### 3. "How was the optimal threshold (0.54) determined?"
> *"Rather than assuming an arbitrary 0.50 threshold, we formulated an operational financial loss function: $\text{Loss} = \sum (\text{Missed Fraud Amount}) + \$5.00 \times \text{False Positives}$. We carved a dedicated 20% validation split out of the training portion and evaluated 91 candidate thresholds from 0.05 to 0.95. The cost curve forms a U-shape where 0.54 represents the global financial cost minimum (\$2,338.36). Evaluating the test set once at 0.54 yielded an immediate \$83.37 monetary savings."*

### 4. "Are these values verified or hardcoded?"
> *"Every value in our reports and dashboard is computed dynamically from actual execution runs on disk (`reports/metrics.json`, `reports/metrics_hybrid.json`, and `reports/graph_risk_top20.csv`). We conducted an exhaustive repository audit across all scripts. The only synthetic additions are the two controlled benchmark rings in `06_load_to_neo4j.py`, which are necessary because the Kaggle PaySim dataset is a Directed Acyclic Graph with zero natural circular loops. All downstream algorithms detect and score these rings through live Cypher pattern matching and NetworkX cycle traversals."*

---

## 📜 Full Documentation Links
- Detailed Technical Audit & Hardcoding Resolution: [**`SYSTEM_UPDATES_AND_AUDIT_REPORT.md`**](file:///d:/VS%20code/college/big_data/Dynamic-UPI-Fraud-Ring-Detection-using-Temporal-Graph-Networks/SYSTEM_UPDATES_AND_AUDIT_REPORT.md)
- Complete Project Walkthrough & Demo Guide: [**`PROJECT_EXPLANATION_AND_DEMO_GUIDE.md`**](file:///d:/VS%20code/college/big_data/Dynamic-UPI-Fraud-Ring-Detection-using-Temporal-Graph-Networks/PROJECT_EXPLANATION_AND_DEMO_GUIDE.md)

---

## 👥 Authors & Course Metadata
- **Project**: Dynamic UPI & Real-Time Financial Fraud Detection Pipeline
- **Course**: Big Data Analytics (DA1 / DA2 Deliverable)
- **Frameworks**: Apache Spark 3.5, Apache Kafka, Neo4j Community 5.x, Python 3.11, Vis.js, Streamlit
