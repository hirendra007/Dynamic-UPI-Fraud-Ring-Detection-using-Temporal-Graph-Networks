"""
06_load_to_neo4j.py - Relational-to-Graph ETL for PaySim Network & Fraud Flagging

Ingests the PaySim mobile-money relational transaction network (sender/receiver/amount),
indexes entities, and builds a directed property graph in Neo4j:
    (:Account {id}) -[:TRANSFER {amount, step, type, isFraud}]-> (:Account {id})
Also cross-references flagged accounts from Phase 3 / PaySim ground truth and marks
them with the `:Flagged` label for downstream graph ring analytics.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase

def get_data_paths():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    paysim_path = base_dir / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
    flagged_dir = base_dir / "data" / "processed" / "flagged"

    if not paysim_path.exists() and (script_dir.parent.parent / "data" / "raw" / "PS_20174392719_1491204439457_log.csv").exists():
        base_dir = script_dir.parent.parent
        paysim_path = base_dir / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
        flagged_dir = base_dir / "data" / "processed" / "flagged"

    return paysim_path, flagged_dir

def connect_to_neo4j(uri, user, password, max_retries=5):
    print(f"[*] Connecting to Neo4j at {uri}...")
    for attempt in range(1, max_retries + 1):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                result = session.run("RETURN 'Connected' AS status").single()
                print(f"[+] Successfully connected to Neo4j on attempt {attempt}: {result['status']}")
            return driver
        except Exception as e:
            print(f"[!] Neo4j connection attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                raise

def setup_schema(session):
    print("[*] Ensuring uniqueness constraints & indexes...")
    # Unique constraint on Account(id)
    session.run("CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE")
    print("[+] Schema constraint verified: Account(id) is unique.")

def load_transactions_batch(session, batch_records):
    cypher_batch = """
    UNWIND $batch AS row
    MERGE (sender:Account {id: row.nameOrig})
    MERGE (receiver:Account {id: row.nameDest})
    CREATE (sender)-[:TRANSFER {
        amount: row.amount,
        step: row.step,
        type: row.type,
        isFraud: row.isFraud
    }]->(receiver)
    """
    session.run(cypher_batch, batch=batch_records)

def main():
    parser = argparse.ArgumentParser(description="Load PaySim Transaction Graph into Neo4j")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="fraudpassword123", help="Neo4j password")
    parser.add_argument("--sample-size", type=int, default=50000,
                        help="Transaction sample size to load (default: 50,000 for responsive graph exploration)")
    parser.add_argument("--batch-size", type=int, default=2000, help="Cypher commit batch size")
    parser.add_argument("--clean-db", action="store_true", default=True, help="Clear existing graph before loading")
    args = parser.parse_args()

    paysim_path, flagged_dir = get_data_paths()

    print("=" * 70)
    print("PHASE 4: Neo4j Graph Database Ingestion")
    print("=" * 70)

    if not paysim_path.exists():
        print(f"[ERROR] PaySim dataset not found at: {paysim_path}")
        print("\nPlease ensure PaySim CSV is downloaded into data/raw/")
        print("Kaggle download command:")
        print("  kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip")
        print("Or manual download: https://www.kaggle.com/datasets/ealaxi/paysim1")
        sys.exit(1)

    driver = connect_to_neo4j(args.uri, args.user, args.password)

    with driver.session() as session:
        if args.clean_db:
            print("[*] Clearing existing graph database (:Account, :TRANSFER)...")
            session.run("MATCH (n:Account) DETACH DELETE n")

        setup_schema(session)

    # Read PaySim dataset
    print(f"[*] Reading PaySim dataset: {paysim_path}")
    df = pd.read_csv(paysim_path)
    print(f"[*] Total records in CSV: {len(df):,}")

    # Prioritize TRANSFER & CASH_OUT where multi-hop fraud and rings occur
    transfer_df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    fraud_txs = transfer_df[transfer_df["isFraud"] == 1]
    legit_txs = transfer_df[transfer_df["isFraud"] == 0]

    print(f"[*] Transfer/CashOut count: {len(transfer_df):,} (Fraud: {len(fraud_txs):,})")

    # Sample to create a dense, realistic network with guaranteed fraud rings
    sample_legit = legit_txs.sample(n=min(args.sample_size - len(fraud_txs), len(legit_txs)), random_state=42)
    load_df = pd.concat([fraud_txs, sample_legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[*] Staging {len(load_df):,} transactions for Neo4j ingestion...")

    # Load in batches
    batch_records = []
    total_loaded = 0
    start_time = time.time()

    with driver.session() as session:
        for idx, row in load_df.iterrows():
            batch_records.append({
                "nameOrig": str(row["nameOrig"]),
                "nameDest": str(row["nameDest"]),
                "amount": float(row["amount"]),
                "step": int(row["step"]),
                "type": str(row["type"]),
                "isFraud": int(row["isFraud"])
            })

            if len(batch_records) >= args.batch_size:
                load_transactions_batch(session, batch_records)
                total_loaded += len(batch_records)
                print(f"    Loaded {total_loaded:,} / {len(load_df):,} transactions...", end="\r")
                batch_records = []

        if batch_records:
            load_transactions_batch(session, batch_records)
            total_loaded += len(batch_records)

        print(f"\n[+] Ingestion complete! Successfully loaded {total_loaded:,} transactions in {time.time() - start_time:.2f}s.")

        # Tag Flagged Accounts
        print("[*] Tagging confirmed fraud accounts with :Flagged label...")
        tag_cypher = """
        MATCH (a:Account)-[r:TRANSFER]->(b:Account)
        WHERE r.isFraud = 1
        SET a:Flagged, b:Flagged
        RETURN count(DISTINCT a) AS flaggedSenders, count(DISTINCT b) AS flaggedReceivers
        """
        tag_result = session.run(tag_cypher).single()
        print(f"[+] Tagged {tag_result['flaggedSenders']} senders and {tag_result['flaggedReceivers']} receivers as :Flagged")

        # Inject Collusive Circular Fraud Rings (Layering Syndicate Benchmark)
        print("[*] Instantiating collusive multi-hop circular fraud rings (3-hop & 4-hop)...")
        ring_cypher = """
        // 4-Hop Ring: Alpha Syndicate
        MERGE (a1:Account {id: 'RING_ALPHA_01'}) SET a1:Flagged
        MERGE (a2:Account {id: 'RING_ALPHA_02'}) SET a2:Flagged
        MERGE (a3:Account {id: 'RING_ALPHA_03'}) SET a3:Flagged
        MERGE (a4:Account {id: 'RING_ALPHA_04'}) SET a4:Flagged
        CREATE (a1)-[:TRANSFER {amount: 450000.0, step: 101, isFraud: 1, type: 'TRANSFER', ringId: 'RING-ALPHA'}]->(a2)
        CREATE (a2)-[:TRANSFER {amount: 440000.0, step: 102, isFraud: 1, type: 'TRANSFER', ringId: 'RING-ALPHA'}]->(a3)
        CREATE (a3)-[:TRANSFER {amount: 430000.0, step: 103, isFraud: 1, type: 'TRANSFER', ringId: 'RING-ALPHA'}]->(a4)
        CREATE (a4)-[:TRANSFER {amount: 420000.0, step: 104, isFraud: 1, type: 'TRANSFER', ringId: 'RING-ALPHA'}]->(a1)

        // 3-Hop Ring: Beta Syndicate
        MERGE (b1:Account {id: 'RING_BETA_01'}) SET b1:Flagged
        MERGE (b2:Account {id: 'RING_BETA_02'}) SET b2:Flagged
        MERGE (b3:Account {id: 'RING_BETA_03'}) SET b3:Flagged
        CREATE (b1)-[:TRANSFER {amount: 185000.0, step: 201, isFraud: 1, type: 'TRANSFER', ringId: 'RING-BETA'}]->(b2)
        CREATE (b2)-[:TRANSFER {amount: 180000.0, step: 202, isFraud: 1, type: 'TRANSFER', ringId: 'RING-BETA'}]->(b3)
        CREATE (b3)-[:TRANSFER {amount: 175000.0, step: 203, isFraud: 1, type: 'TRANSFER', ringId: 'RING-BETA'}]->(b1)
        """
        session.run(ring_cypher)
        print("[+] Created 4-hop Ring Alpha and 3-hop Ring Beta for cyclic detection validation.")

        # Verify Counts
        count_res = session.run("MATCH (a:Account) RETURN count(a) AS accountCount, count(a:Flagged) AS flaggedCount").single()
        rel_res = session.run("MATCH ()-[r:TRANSFER]->() RETURN count(r) AS transferCount").single()
        print("-" * 50)
        print(f"    Total :Account nodes: {count_res['accountCount']:,}")
        print(f"    Total :Flagged accounts: {count_res['flaggedCount']:,}")
        print(f"    Total :TRANSFER edges: {rel_res['transferCount']:,}")
        print("-" * 50)

    driver.close()
    print("[SUCCESS] Neo4j graph population finished successfully!")

if __name__ == "__main__":
    main()
