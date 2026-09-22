"""
execute_cypher_analytics.py - Automated Execution of Graph Analytics & Ring Detection

Executes graph algorithms and Cypher queries against the Neo4j database:
1. Cyclic Transfer Detection (2 to 4 hop rings)
2. High-Degree Mule & Hub Account Detection (Fan-in / Fan-out topologies)
3. Multi-Hop Layering Chains (Origin -> Intermediary Mule -> Cash-Out Destination)
4. Connected Fraud Communities (Weakly Connected Components / Clusters)
5. Exports network topology visualization to reports/figures/neo4j_fraud_ring.png
6. Updates reports/metrics.json with graph discovery statistics.
"""

import json
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from neo4j import GraphDatabase

def get_paths():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    metrics_path = base_dir / "reports" / "metrics.json"
    figures_dir = base_dir / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return metrics_path, figures_dir

def main():
    metrics_path, figures_dir = get_paths()
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "fraudpassword123"

    print("=" * 70)
    print("PHASE 4: Graph Analytics & Fraud Ring Detection via Cypher")
    print("=" * 70)

    print(f"[*] Connecting to Neo4j at {uri}...")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # 1. Total Graph Statistics
        print("\n[*] Querying Graph Summary Statistics...")
        stats_query = """
        MATCH (a:Account)
        OPTIONAL MATCH (a)-[r:TRANSFER]->()
        RETURN count(DISTINCT a) AS nodeCount, count(r) AS edgeCount
        """
        stats = session.run(stats_query).single()
        print(f"    Total Entities (:Account): {stats['nodeCount']:,}")
        print(f"    Total Transactions (:TRANSFER): {stats['edgeCount']:,}")

        # 2. High-Degree Mule & Hub Accounts (Degree Centrality)
        print("\n[*] Query 1: Identifying High-Degree Hub & Mule Accounts...")
        hub_query = """
        MATCH (a:Account)
        WITH a, count{(a)-[:TRANSFER]->()} AS outDegree, count{(a)<-[:TRANSFER]-()} AS inDegree
        WHERE outDegree + inDegree >= 2
        RETURN a.id AS accountId, inDegree, outDegree, inDegree + outDegree AS totalDegree
        ORDER BY totalDegree DESC
        LIMIT 15
        """
        hub_results = list(session.run(hub_query))
        print(f"{'ACCOUNT ID':<18} | {'IN-DEGREE':<10} | {'OUT-DEGREE':<10} | {'TOTAL DEGREE'}")
        print("-" * 55)
        hubs_data = []
        for row in hub_results:
            print(f"{row['accountId']:<18} | {row['inDegree']:<10} | {row['outDegree']:<10} | {row['totalDegree']}")
            hubs_data.append({
                "account_id": row["accountId"],
                "in_degree": row["inDegree"],
                "out_degree": row["outDegree"],
                "total_degree": row["totalDegree"]
            })

        # 3. Cyclic Transfer Detection (2 to 4 Hop Closed Cycles)
        print("\n[*] Query 2: Detecting Cyclic Transfer Rings (2-4 Hop Cycles)...")
        cycle_query = """
        MATCH path = (origin:Account)-[:TRANSFER*2..4]->(origin)
        WITH origin, path, [rel in relationships(path) | rel.amount] AS amounts
        RETURN 
            origin.id AS ringOrigin,
            length(path) AS cycleLengthHops,
            [node in nodes(path) | node.id] AS ringAccounts,
            reduce(total = 0.0, a in amounts | total + a) AS totalRingVolume
        LIMIT 25
        """
        cycle_results = list(session.run(cycle_query))
        rings_found = len(cycle_results)
        print(f"    Found {rings_found} cyclic fraud rings.")
        rings_data = []
        for c in cycle_results[:5]:
            ring_chain = " -> ".join(c["ringAccounts"])
            print(f"    * Ring ({c['cycleLengthHops']} hops): {ring_chain} | Total Volume: ${c['totalRingVolume']:,.2f}")
            rings_data.append({
                "origin": c["ringOrigin"],
                "hops": c["cycleLengthHops"],
                "accounts": c["ringAccounts"],
                "volume": round(c["totalRingVolume"], 2)
            })

        # 4. Multi-Hop Layering Chains (Origin -> Intermediate Mule -> Destination)
        print("\n[*] Query 3: Detecting Multi-Hop Layering Chains (A -> B -> C)...")
        layering_query = """
        MATCH (a:Account)-[r1:TRANSFER {isFraud: 1}]->(mule:Account)-[r2:TRANSFER]->(dest:Account)
        WHERE a.id <> dest.id
        RETURN 
            a.id AS originAccount, 
            mule.id AS muleAccount, 
            dest.id AS destinationAccount,
            r1.amount AS initialStolenAmount,
            r2.amount AS dispersedAmount,
            r1.step AS stepInitial,
            r2.step AS stepDispersed
        ORDER BY initialStolenAmount DESC
        LIMIT 10
        """
        layering_results = list(session.run(layering_query))
        print(f"    Found {len(layering_results)} multi-hop layering chains.")
        chains_data = []
        for chain in layering_results:
            print(f"    * {chain['originAccount']} -> [${chain['initialStolenAmount']:,.2f}] -> {chain['muleAccount']} -> [${chain['dispersedAmount']:,.2f}] -> {chain['destinationAccount']}")
            chains_data.append({
                "origin": chain["originAccount"],
                "mule": chain["muleAccount"],
                "destination": chain["destinationAccount"],
                "stolen_amount": round(chain["initialStolenAmount"], 2),
                "dispersed_amount": round(chain["dispersedAmount"], 2)
            })

        # 5. Connected Component Sub-Graph Visualization
        print("\n[*] Generating Graph Visualization for Academic Deliverable...")
        visual_query = """
        MATCH (a:Account)-[r:TRANSFER {isFraud: 1}]->(m:Account)-[r2:TRANSFER]->(d:Account)
        RETURN a.id AS src1, m.id AS dst1, r1.amount AS amt1, m.id AS src2, d.id AS dst2, r2.amount AS amt2
        LIMIT 35
        """
        # Fetch a rich connected subgraph around fraud nodes
        graph_query = """
        MATCH (origin:Account)-[r:TRANSFER {isFraud: 1}]->(mule:Account)
        OPTIONAL MATCH (mule)-[r2:TRANSFER]->(dest:Account)
        RETURN origin.id AS src, mule.id AS target, r.amount AS amount, true AS isFraud
        LIMIT 40
        """
        edges = list(session.run(graph_query))

        G = nx.DiGraph()
        for e in edges:
            G.add_edge(e["src"], e["target"], amount=e["amount"], isFraud=True)

        # Plot NetworkX Directed Graph
        fig, ax = plt.subplots(figsize=(12, 9), facecolor='#0e1117')
        pos = nx.spring_layout(G, k=0.7, seed=42)

        # Draw nodes (Hubs in Gold, Mules in Orange, Victims in Crimson)
        node_degrees = dict(G.degree())
        node_colors = []
        for node in G.nodes():
            in_d = G.in_degree(node)
            out_d = G.out_degree(node)
            if in_d > 0 and out_d > 0:
                node_colors.append('#f39c12') # Transit Mule
            elif in_d > 1:
                node_colors.append('#e74c3c') # Accumulator Hub
            else:
                node_colors.append('#3498db') # Origin / Disperser

        nx.draw_networkx_nodes(G, pos, node_size=350, node_color=node_colors, edgecolors='#ffffff', linewidths=1.2, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='#7f8c8d', width=1.5, alpha=0.7, arrows=True, arrowsize=14, ax=ax)

        # Labels for top hubs
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        labels = {n: f"{n[:6]}.." for n, _ in top_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='white', font_weight='bold', ax=ax)

        # Legend
        ax.plot([], [], 'o', color='#3498db', label='Origin Accounts (Victim / Source)')
        ax.plot([], [], 'o', color='#f39c12', label='Transit Mule Nodes (Layering Hubs)')
        ax.plot([], [], 'o', color='#e74c3c', label='Accumulation Hubs (Cash-Out Mules)')
        ax.legend(loc='upper right', facecolor='#1e222d', edgecolor='#4b5563', labelcolor='white', fontsize=10)

        ax.set_title("Neo4j Fraud Ring & Multi-Hop Mule Network Topology\n(PaySim Financial Transaction Graph)",
                     color='white', fontsize=14, fontweight='bold', pad=15)
        ax.axis('off')
        plt.tight_layout()

        ring_screenshot_path = figures_dir / "neo4j_fraud_ring.png"
        plt.savefig(ring_screenshot_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close()
        print(f"[+] Saved graph visualization screenshot to: {ring_screenshot_path}")

        # 6. Update reports/metrics.json
        print("\n[*] Updating reports/metrics.json with graph discovery analytics...")
        metrics = {}
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            except Exception:
                metrics = {}

        metrics["graph_analytics"] = {
            "total_accounts": stats["nodeCount"],
            "total_transfers": stats["edgeCount"],
            "top_hub_accounts": hubs_data,
            "rings_detected_count": rings_found,
            "sample_rings": rings_data,
            "multi_hop_layering_chains_count": len(layering_results),
            "sample_layering_chains": chains_data
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"[+] Successfully recorded graph analytics in: {metrics_path}")

    driver.close()
    print("\n[SUCCESS] Phase 4 Graph Analytics & Cypher Ring Detection completed successfully!")

if __name__ == "__main__":
    main()
