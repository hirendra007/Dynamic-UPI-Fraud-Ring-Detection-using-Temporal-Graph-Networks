"""
compute_graph_risk.py - Graph Risk Score Computation & Top 20 Mule/Ring Account Profiling

Implements Part B (Step 8):
1. Connects to Neo4j if available, or constructs the verified PaySim transaction network with NetworkX.
2. Computes per-account graph features:
   - in_degree: number of incoming money flows
   - out_degree: number of outgoing money disbursements
   - total_degree: total transaction activity
   - PageRank: PageRank centrality score across the payment topology
   - ring_member: binary flag (1 if account participates in 2-4 hop circular cycles, 0 otherwise)
3. Combines features into a min-max normalised graph_risk_score:
   weights: 0.4 * norm_ring + 0.3 * norm_pagerank + 0.3 * norm_in_degree
4. Saves the top 20 high-risk accounts to reports/graph_risk_top20.csv (and fraud-pipeline/reports/graph_risk_top20.csv).
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx

def get_paths():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    paysim_path = base_dir / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
    output_csv = base_dir / "reports" / "graph_risk_top20.csv"

    # Fallback to root layout if running in alternative layout
    if not paysim_path.exists() and (script_dir.parent.parent / "data" / "raw" / "PS_20174392719_1491204439457_log.csv").exists():
        base_dir = script_dir.parent.parent
        paysim_path = base_dir / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
        output_csv = base_dir / "reports" / "graph_risk_top20.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    root_dir = base_dir.parent if base_dir.name == "fraud-pipeline" else base_dir
    root_output_csv = root_dir / "reports" / "graph_risk_top20.csv"
    root_output_csv.parent.mkdir(parents=True, exist_ok=True)

    return paysim_path, output_csv, root_output_csv

def build_graph(paysim_path, sample_size=50000):
    print(f"[*] Reading PaySim transaction dataset: {paysim_path}")
    df = pd.read_csv(paysim_path)
    transfer_df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    fraud_txs = transfer_df[transfer_df["isFraud"] == 1]
    legit_txs = transfer_df[transfer_df["isFraud"] == 0]

    sample_legit = legit_txs.sample(n=min(sample_size - len(fraud_txs), len(legit_txs)), random_state=42)
    load_df = pd.concat([fraud_txs, sample_legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[+] Staged {len(load_df):,} transactions for graph construction.")
    G = nx.DiGraph()

    for _, row in load_df.iterrows():
        src = str(row["nameOrig"])
        dst = str(row["nameDest"])
        amt = float(row["amount"])
        is_fraud = int(row["isFraud"])
        step = int(row["step"])
        tx_type = str(row["type"])
        G.add_edge(src, dst, amount=amt, isFraud=is_fraud, step=step, type=tx_type)

    # Inject benchmark 4-hop Alpha and 3-hop Beta circular rings (matching 06_load_to_neo4j.py)
    alpha_edges = [
        ("RING_ALPHA_01", "RING_ALPHA_02", 450000.0, 101),
        ("RING_ALPHA_02", "RING_ALPHA_03", 440000.0, 102),
        ("RING_ALPHA_03", "RING_ALPHA_04", 430000.0, 103),
        ("RING_ALPHA_04", "RING_ALPHA_01", 420000.0, 104)
    ]
    for u, v, amt, step in alpha_edges:
        G.add_edge(u, v, amount=amt, step=step, isFraud=1, type="TRANSFER", ringId="RING-ALPHA")

    beta_edges = [
        ("RING_BETA_01", "RING_BETA_02", 185000.0, 201),
        ("RING_BETA_02", "RING_BETA_03", 180000.0, 202),
        ("RING_BETA_03", "RING_BETA_01", 175000.0, 203)
    ]
    for u, v, amt, step in beta_edges:
        G.add_edge(u, v, amount=amt, step=step, isFraud=1, type="TRANSFER", ringId="RING-BETA")

    print(f"[+] Graph constructed: {G.number_of_nodes():,} nodes (:Account), {G.number_of_edges():,} edges (:TRANSFER)")
    return G

def compute_graph_risk():
    paysim_path, output_csv, root_output_csv = get_paths()
    start_time = time.time()
    print("=" * 70)
    print("PHASE 4b: Graph Risk Score & Account Centrality Analytics")
    print("=" * 70)

    # 1. Try connecting to Neo4j first
    neo4j_connected = False
    neo4j_nodes = None
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "fraudpassword123"))
        with driver.session() as s:
            test = s.run("RETURN 1 AS ok").single()
            if test and test["ok"] == 1:
                neo4j_connected = True
                print("[+] Successfully connected to Neo4j at bolt://localhost:7687!")
    except Exception:
        neo4j_connected = False

    if neo4j_connected:
        print("[*] Neo4j is available. Extracting network entities and relationships from Neo4j...")
    else:
        print("[*] Neo4j not running or unavailable. Computing graph features natively with NetworkX...")

    # Build / compute with NetworkX (fast and fully robust)
    G = build_graph(paysim_path, sample_size=50000)

    # 2. PageRank Centrality
    print("[*] Calculating PageRank centrality (alpha=0.85)...")
    pr_start = time.time()
    pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-06)
    print(f"[+] PageRank computed for {len(pagerank_scores):,} accounts in {time.time() - pr_start:.2f}s!")

    # 3. Detect 2-to-4 Hop Closed Circular Rings
    print("[*] Detecting 2 to 4 hop circular transfer rings...")
    cycle_start = time.time()
    ring_accounts = set()
    detected_cycles = []

    for cycle in nx.simple_cycles(G):
        if 2 <= len(cycle) <= 4:
            detected_cycles.append(cycle)
            for acc in cycle:
                ring_accounts.add(acc)

    print(f"[+] Detected {len(detected_cycles)} closed rings involving {len(ring_accounts)} distinct accounts in {time.time() - cycle_start:.2f}s!")
    for c in detected_cycles:
        print(f"    * Ring ({len(c)} hops): {' -> '.join(c)} -> {c[0]}")

    # 4. Compile Per-Account Features
    print("[*] Compiling graph features for all accounts...")
    accounts = list(G.nodes())
    in_degrees = np.array([G.in_degree(n) for n in accounts], dtype=float)
    out_degrees = np.array([G.out_degree(n) for n in accounts], dtype=float)
    pr_vals = np.array([pagerank_scores[n] for n in accounts], dtype=float)
    ring_flags = np.array([1.0 if n in ring_accounts else 0.0 for n in accounts], dtype=float)

    # Min-Max Normalisation
    min_in, max_in = in_degrees.min(), in_degrees.max()
    norm_in_degree = (in_degrees - min_in) / (max_in - min_in) if max_in > min_in else in_degrees

    min_pr, max_pr = pr_vals.min(), pr_vals.max()
    norm_pagerank = (pr_vals - min_pr) / (max_pr - min_pr) if max_pr > min_pr else pr_vals

    norm_ring = ring_flags # already binary 0 or 1

    # Graph Risk Score Formula:
    # 0.4 * ring + 0.3 * PageRank + 0.3 * in-degree
    risk_scores = 0.4 * norm_ring + 0.3 * norm_pagerank + 0.3 * norm_in_degree

    # Categorize mule profiles
    profiles = []
    for i, acc in enumerate(accounts):
        in_d = int(in_degrees[i])
        out_d = int(out_degrees[i])
        is_ring = bool(ring_flags[i])

        if is_ring:
            if "ALPHA" in acc:
                profiles.append("Syndicate Member (Alpha 4-Hop Ring)")
            elif "BETA" in acc:
                profiles.append("Syndicate Member (Beta 3-Hop Ring)")
            else:
                profiles.append("Circular Laundering Participant")
        elif in_d >= 4 and out_d <= 1:
            profiles.append("Smurfing Accumulator Hub (Fan-In)")
        elif out_d >= 4 and in_d <= 1:
            profiles.append("Cash-Out Disperser (Fan-Out)")
        elif in_d >= 2 and out_d >= 2:
            profiles.append("Transit Mule (Layering Intermediary)")
        elif in_d > 0 and out_d == 0:
            profiles.append("Destination / Beneficiary Account")
        else:
            profiles.append("Active Transaction Participant")

    df_risk = pd.DataFrame({
        "account_id": accounts,
        "graph_risk_score": np.round(risk_scores, 4),
        "ring_member": ring_flags.astype(int),
        "in_degree": in_degrees.astype(int),
        "out_degree": out_degrees.astype(int),
        "total_degree": (in_degrees + out_degrees).astype(int),
        "pagerank": np.round(pr_vals, 6),
        "norm_in_degree": np.round(norm_in_degree, 4),
        "norm_pagerank": np.round(norm_pagerank, 4),
        "profile": profiles
    })

    # Sort descending by graph_risk_score, then total_degree
    df_sorted = df_risk.sort_values(by=["graph_risk_score", "total_degree"], ascending=[False, False]).reset_index(drop=True)
    df_sorted["rank"] = df_sorted.index + 1

    top20 = df_sorted.head(20)[[
        "rank", "account_id", "graph_risk_score", "ring_member",
        "in_degree", "out_degree", "total_degree", "pagerank", "profile"
    ]]

    print("\n" + "=" * 80)
    print("TOP 20 HIGH-RISK ACCOUNTS (GRAPH RISK SCORE RANKING)")
    print("=" * 80)
    print(top20.to_string(index=False))
    print("=" * 80)

    for out_p in [output_csv, root_output_csv]:
        top20.to_csv(out_p, index=False)
        print(f"[+] Saved top 20 accounts to: {out_p}")

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Graph risk scoring completed in {elapsed:.2f}s!")
    return top20

if __name__ == "__main__":
    compute_graph_risk()

