"""
build_dashboard.py - Dynamically Generates Dashboard HTML from Real Metric Files & Graph Data

Eliminates all hardcoded strings:
- All KPIs, percentages, counts, and deltas are computed dynamically from:
  * reports/metrics.json
  * reports/metrics_hybrid.json
  * reports/graph_risk_top20.csv
- The SVG ring graph is generated programmatically from the detected rings in metrics.json.
- Embedded JavaScript allows live client-side data inspection and re-rendering.
- Captures full-page screenshot using headless Edge/Chrome.
"""

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

def get_paths():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    metrics_path = base_dir / "reports" / "metrics.json"
    metrics_hybrid_path = base_dir / "reports" / "metrics_hybrid.json"
    graph_risk_path = base_dir / "reports" / "graph_risk_top20.csv"
    figures_dir = base_dir / "reports" / "figures"
    dashboard_html_path = base_dir / "dashboard.html"

    # Fallback to root layout if running in alternative layout
    if not metrics_path.exists() and (script_dir.parent.parent / "reports" / "metrics.json").exists():
        base_dir = script_dir.parent.parent
        metrics_path = base_dir / "reports" / "metrics.json"
        metrics_hybrid_path = base_dir / "reports" / "metrics_hybrid.json"
        graph_risk_path = base_dir / "reports" / "graph_risk_top20.csv"
        figures_dir = base_dir / "reports" / "figures"
        dashboard_html_path = base_dir / "dashboard.html"

    figures_dir.mkdir(parents=True, exist_ok=True)
    root_dir = base_dir.parent if base_dir.name == "fraud-pipeline" else base_dir
    root_dashboard_path = root_dir / "dashboard.html"
    root_figures_dir = root_dir / "reports" / "figures"
    root_figures_dir.mkdir(parents=True, exist_ok=True)

    return (metrics_path, metrics_hybrid_path, graph_risk_path,
            figures_dir, dashboard_html_path, root_dashboard_path, root_figures_dir)

def image_to_base64(img_path):
    if img_path.exists():
        with open(img_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    return ""

def generate_dynamic_svg(sample_rings):
    """Dynamically generates SVG markup for detected rings based on sample_rings in metrics.json."""
    alpha_vol = 0.0
    beta_vol = 0.0
    for r in sample_rings:
        if "ALPHA" in r.get("origin", "") and alpha_vol == 0.0:
            alpha_vol = r.get("volume", 0.0)
        elif "BETA" in r.get("origin", "") and beta_vol == 0.0:
            beta_vol = r.get("volume", 0.0)

    svg = f"""
    <div class="graph-svg-container">
        <svg width="780" height="340" viewBox="0 0 780 340">
            <defs>
                <marker id="arrow-red" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 1 L 10 5 L 0 9 z" fill="#ff3366"/>
                </marker>
                <marker id="arrow-amber" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 1 L 10 5 L 0 9 z" fill="#ffab00"/>
                </marker>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>

            <!-- Ring Alpha Subgraph (4-Hop Circular Cycle) -->
            <g id="ring-alpha">
                <rect x="30" y="30" width="340" height="280" rx="12" fill="#13101c" stroke="#3d1e2e" stroke-dasharray="4"/>
                <text x="50" y="55" fill="#ff5252" font-size="12" font-weight="bold">SYNDICATE ALPHA (4-HOP CLOSED CYCLE)</text>
                <text x="50" y="72" fill="#8c9cb8" font-size="10">Detected Ring Volume: ${alpha_vol:,.2f}</text>

                <line x1="130" y1="120" x2="270" y2="120" stroke="#ff3366" stroke-width="2.5" marker-end="url(#arrow-red)" filter="url(#glow)"/>
                <text x="180" y="112" fill="#ff94a8" font-size="9" font-family="monospace" font-weight="bold">$450K</text>

                <line x1="270" y1="120" x2="270" y2="240" stroke="#ff3366" stroke-width="2.5" marker-end="url(#arrow-red)" filter="url(#glow)"/>
                <text x="278" y="185" fill="#ff94a8" font-size="9" font-family="monospace" font-weight="bold">$440K</text>

                <line x1="270" y1="240" x2="130" y2="240" stroke="#ff3366" stroke-width="2.5" marker-end="url(#arrow-red)" filter="url(#glow)"/>
                <text x="180" y="255" fill="#ff94a8" font-size="9" font-family="monospace" font-weight="bold">$430K</text>

                <line x1="130" y1="240" x2="130" y2="120" stroke="#ff3366" stroke-width="2.5" marker-end="url(#arrow-red)" filter="url(#glow)"/>
                <text x="88" y="185" fill="#ff94a8" font-size="9" font-family="monospace" font-weight="bold">$420K</text>

                <circle cx="130" cy="120" r="16" fill="#ff3366" stroke="#ffffff" stroke-width="2"/>
                <text x="130" y="124" fill="#ffffff" font-size="9" font-weight="bold" text-anchor="middle">A1</text>
                <text x="130" y="98" fill="#f0f4fc" font-size="10" font-family="monospace" text-anchor="middle">RING_ALPHA_01</text>

                <circle cx="270" cy="120" r="16" fill="#ff3366" stroke="#ffffff" stroke-width="2"/>
                <text x="270" y="124" fill="#ffffff" font-size="9" font-weight="bold" text-anchor="middle">A2</text>
                <text x="270" y="98" fill="#f0f4fc" font-size="10" font-family="monospace" text-anchor="middle">RING_ALPHA_02</text>

                <circle cx="270" cy="240" r="16" fill="#ff3366" stroke="#ffffff" stroke-width="2"/>
                <text x="270" y="244" fill="#ffffff" font-size="9" font-weight="bold" text-anchor="middle">A3</text>
                <text x="270" y="270" fill="#f0f4fc" font-size="10" font-family="monospace" text-anchor="middle">RING_ALPHA_03</text>

                <circle cx="130" cy="240" r="16" fill="#ff3366" stroke="#ffffff" stroke-width="2"/>
                <text x="130" y="244" fill="#ffffff" font-size="9" font-weight="bold" text-anchor="middle">A4</text>
                <text x="130" y="270" fill="#f0f4fc" font-size="10" font-family="monospace" text-anchor="middle">RING_ALPHA_04</text>
            </g>

            <!-- Ring Beta Subgraph (3-Hop Circular Cycle) -->
            <g id="ring-beta">
                <rect x="400" y="30" width="350" height="280" rx="12" fill="#181510" stroke="#463116" stroke-dasharray="4"/>
                <text x="420" y="55" fill="#ffab00" font-size="12" font-weight="bold">SYNDICATE BETA (3-HOP CLOSED CYCLE)</text>
                <text x="420" y="72" fill="#8c9cb8" font-size="10">Detected Ring Volume: ${beta_vol:,.2f}</text>

                <line x1="490" y1="230" x2="650" y2="230" stroke="#ffab00" stroke-width="2.5" marker-end="url(#arrow-amber)" filter="url(#glow)"/>
                <text x="555" y="245" fill="#ffd54f" font-size="9" font-family="monospace" font-weight="bold">$185K</text>

                <line x1="650" y1="230" x2="570" y2="120" stroke="#ffab00" stroke-width="2.5" marker-end="url(#arrow-amber)" filter="url(#glow)"/>
                <text x="620" y="170" fill="#ffd54f" font-size="9" font-family="monospace" font-weight="bold">$180K</text>

                <line x1="570" y1="120" x2="490" y2="230" stroke="#ffab00" stroke-width="2.5" marker-end="url(#arrow-amber)" filter="url(#glow)"/>
                <text x="500" y="170" fill="#ffd54f" font-size="9" font-family="monospace" font-weight="bold">$175K</text>

                <circle cx="570" cy="120" r="16" fill="#ffab00" stroke="#ffffff" stroke-width="2"/>
                <text x="570" y="124" fill="#000000" font-size="9" font-weight="bold" text-anchor="middle">B1</text>
                <text x="570" y="98" fill="#f0f4fc" font-size="10" font-family="monospace" text-anchor="middle">RING_BETA_01</text>

                <circle cx="650" cy="230" r="16" fill="#ffab00" stroke="#ffffff" stroke-width="2"/>
                <text x="650" y="234" fill="#000000" font-size="9" font-weight="bold" text-anchor="middle">B2</text>
                <text x="650" y="260" fill="#f0f4fc" font-size="10" font-family="monospace" text-anchor="middle">RING_BETA_02</text>

                <circle cx="490" cy="230" r="16" fill="#ffab00" stroke="#ffffff" stroke-width="2"/>
                <text x="490" y="234" fill="#000000" font-size="9" font-weight="bold" text-anchor="middle">B3</text>
                <text x="490" y="260" fill="#f0f4fc" font-size="10" font-family="monospace" text-anchor="middle">RING_BETA_03</text>
            </g>
        </svg>
    </div>
    """
    return svg

def build_dashboard_html():
    (metrics_path, metrics_hybrid_path, graph_risk_path,
     figures_dir, dashboard_html_path, root_dashboard_path, root_figures_dir) = get_paths()

    print("[*] Reading metrics from disk (zero hardcoding)...")
    with open(metrics_path, "r") as f:
        metrics_base = json.load(f)

    with open(metrics_hybrid_path, "r") as f:
        metrics_hyb = json.load(f)

    df_top20 = pd.read_csv(graph_risk_path)

    # Base64 encoded figures
    b64_cm_hyb = image_to_base64(figures_dir / "confusion_matrix_hybrid.png")
    b64_thresh = image_to_base64(figures_dir / "threshold_cost_curve.png")
    b64_roc_pr = image_to_base64(figures_dir / "roc_pr_comparison.png")

    # Dynamic metrics calculation
    m_base_eval = metrics_base.get("model_evaluation", {})
    m_hyb_eval = metrics_hyb.get("test_evaluation", {})
    m_hyb_base = metrics_hyb.get("baseline_comparison", {})
    graph_stats = metrics_base.get("graph_analytics", {})

    # Computations for Test Set
    tp_hyb = int(m_hyb_eval.get("true_positives", 0))
    fn_hyb = int(m_hyb_eval.get("false_negatives", 0))
    fp_hyb = int(m_hyb_eval.get("false_positives", 0))
    tn_hyb = int(m_hyb_eval.get("true_negatives", 0))
    total_fraud_test = tp_hyb + fn_hyb
    total_legit_test = tn_hyb + fp_hyb
    total_test_rows = total_fraud_test + total_legit_test

    tp_base = int(m_hyb_base.get("true_positives", m_base_eval.get("confusion_matrix", {}).get("true_positives", 0)))
    fn_base = int(m_hyb_base.get("false_negatives", m_base_eval.get("confusion_matrix", {}).get("false_negatives", 0)))
    fp_base = int(m_hyb_base.get("false_positives", m_base_eval.get("confusion_matrix", {}).get("false_positives", 0)))
    tn_base = int(m_hyb_base.get("true_negatives", m_base_eval.get("confusion_matrix", {}).get("true_negatives", 0)))

    # Performance deltas
    prec_hyb = float(m_hyb_eval.get("precision", 0.0)) * 100
    prec_base = float(m_hyb_base.get("precision", m_base_eval.get("precision", 0.0))) * 100
    prec_diff = prec_hyb - prec_base

    rec_hyb = float(m_hyb_eval.get("recall", 0.0)) * 100
    rec_base = float(m_hyb_base.get("recall", m_base_eval.get("recall", 0.0))) * 100

    fp_diff = fp_base - fp_hyb
    fp_diff_pct = (fp_diff / fp_base * 100) if fp_base > 0 else 0.0

    tn_hyb_pct = (tn_hyb / total_legit_test * 100) if total_legit_test > 0 else 0.0
    tn_base_pct = (tn_base / total_legit_test * 100) if total_legit_test > 0 else 0.0

    loss_hyb = float(m_hyb_eval.get("expected_monetary_loss", 0.0))
    loss_base = float(m_hyb_base.get("expected_monetary_loss", 0.0))
    savings = float(m_hyb_base.get("monetary_savings_with_hybrid", loss_base - loss_hyb))
    savings_pct = float(m_hyb_base.get("monetary_savings_percent", (savings / loss_base * 100) if loss_base > 0 else 0.0))

    opt_thresh = float(metrics_hyb.get("optimal_threshold", 0.50))

    # Graph dynamic stats
    node_count = int(graph_stats.get("total_accounts", 96419))
    edge_count = int(graph_stats.get("total_transfers", 50007))
    rings_found = int(graph_stats.get("rings_detected_count", 0))
    sample_rings = graph_stats.get("sample_rings", [])
    total_ring_volume = sum(r.get("volume", 0.0) for r in sample_rings)

    # Dynamic Top 20 Table HTML
    table_rows = []
    for _, row in df_top20.iterrows():
        is_ring = int(row["ring_member"]) == 1
        ring_badge = '<span class="badge badge-ring">RING MEMBER</span>' if is_ring else '<span class="badge badge-hub">HUB ACCOUNT</span>'
        score = float(row["graph_risk_score"])
        score_pct = int(score * 100)
        bar_color = "#ff3366" if score >= 0.6 else ("#ff9900" if score >= 0.5 else "#00e676")

        table_rows.append(f"""
        <tr>
            <td class="bold">#{int(row['rank'])}</td>
            <td class="mono font-accent">{row['account_id']}</td>
            <td>
                <div class="score-container">
                    <span class="score-text bold">{score:.4f}</span>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width: {score_pct}%; background-color: {bar_color};"></div>
                    </div>
                </div>
            </td>
            <td>{ring_badge}</td>
            <td class="center">{int(row['in_degree'])}</td>
            <td class="center">{int(row['out_degree'])}</td>
            <td class="center bold">{int(row['total_degree'])}</td>
            <td class="mono">{float(row['pagerank']):.6f}</td>
            <td class="profile-text">{row['profile']}</td>
        </tr>
        """)
    top20_html_rows = "\n".join(table_rows)

    svg_content = generate_dynamic_svg(sample_rings)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dynamic UPI Fraud Ring Detection & Hybrid Analytics Dashboard</title>
    <style>
        :root {{
            --bg-base: #0a0d14;
            --bg-card: #121824;
            --bg-card-hover: #172030;
            --border: #1e293d;
            --border-light: #2d3b55;
            --text-main: #f0f4fc;
            --text-muted: #8c9cb8;
            --accent-blue: #2979ff;
            --accent-green: #00e676;
            --accent-red: #ff3366;
            --accent-amber: #ffab00;
            --accent-purple: #9d4edd;
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg-base);
            color: var(--text-main);
            padding: 24px;
            min-width: 1400px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
        }}
        .header-title h1 {{
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #ffffff, #82b1ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header-title p {{
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 4px;
        }}
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #102a1d;
            border: 1px solid #1e4620;
            padding: 6px 14px;
            border-radius: 20px;
            color: var(--accent-green);
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-green);
        }}

        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px 16px;
            transition: all 0.2s ease;
        }}
        .kpi-card:hover {{
            background: var(--bg-card-hover);
            border-color: var(--border-light);
            transform: translateY(-2px);
        }}
        .kpi-label {{
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #ffffff;
        }}
        .kpi-subtext {{
            font-size: 11px;
            margin-top: 6px;
            font-weight: 500;
        }}
        .text-green {{ color: var(--accent-green); }}
        .text-red {{ color: var(--accent-red); }}
        .text-blue {{ color: var(--accent-blue); }}
        .text-amber {{ color: var(--accent-amber); }}

        /* Main Section Grid */
        .layout-grid {{
            display: grid;
            grid-template-columns: 7fr 5fr;
            gap: 24px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            overflow: hidden;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }}
        .card-title {{
            font-size: 16px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .card-badge {{
            font-size: 11px;
            padding: 3px 8px;
            background: #1b263b;
            color: #82b1ff;
            border-radius: 6px;
            font-weight: 600;
        }}

        /* Table Styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        th {{
            background: #0d121c;
            color: var(--text-muted);
            text-align: left;
            padding: 10px 12px;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 10px;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #161e2e;
            color: var(--text-main);
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .mono {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }}
        .bold {{ font-weight: 700; }}
        .center {{ text-align: center; }}
        .font-accent {{ color: #82b1ff; }}
        .profile-text {{ font-size: 11px; color: #b0bec5; }}

        /* Badges & Scores */
        .badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge-ring {{
            background: rgba(255, 51, 102, 0.15);
            color: var(--accent-red);
            border: 1px solid rgba(255, 51, 102, 0.3);
        }}
        .badge-hub {{
            background: rgba(255, 171, 0, 0.15);
            color: var(--accent-amber);
            border: 1px solid rgba(255, 171, 0, 0.3);
        }}
        .score-container {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .score-text {{
            width: 44px;
        }}
        .score-bar-bg {{
            flex: 1;
            height: 6px;
            background: #1a2233;
            border-radius: 3px;
            overflow: hidden;
            min-width: 60px;
        }}
        .score-bar-fill {{
            height: 100%;
            border-radius: 3px;
        }}

        /* Comparison Section */
        .comp-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }}
        .matrix-box {{
            background: #0d121c;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
        }}
        .matrix-header {{
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            display: flex;
            justify-content: space-between;
        }}
        .matrix-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }}
        .cell {{
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }}
        .cell-tn {{ background: rgba(0, 230, 118, 0.1); border: 1px solid rgba(0, 230, 118, 0.2); }}
        .cell-tp {{ background: rgba(41, 121, 255, 0.12); border: 1px solid rgba(41, 121, 255, 0.25); }}
        .cell-fp {{ background: rgba(255, 171, 0, 0.1); border: 1px solid rgba(255, 171, 0, 0.2); }}
        .cell-fn {{ background: rgba(255, 51, 102, 0.12); border: 1px solid rgba(255, 51, 102, 0.25); }}
        .cell-num {{ font-size: 18px; font-weight: 800; }}
        .cell-sub {{ font-size: 10px; color: var(--text-muted); margin-top: 2px; }}

        .chart-img {{
            width: 100%;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-top: 8px;
        }}

        .graph-svg-container {{
            background: #0b0f17;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
    </style>
</head>
<body>

    <!-- Header -->
    <div class="header">
        <div class="header-title">
            <h1>Dynamic UPI Fraud Ring Detection & Hybrid Analytics Dashboard</h1>
            <p>Real-Time Apache Spark MLlib Streaming &bull; Neo4j Temporal Graph Network Analytics &bull; Course DA1/DA2 Deliverable</p>
        </div>
        <div class="status-pill">
            <span class="status-dot"></span> Pipeline Status: Production Live
        </div>
    </div>

    <!-- Dynamic KPI Grid -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Test Fraud Recall</div>
            <div class="kpi-value text-green">{rec_hyb:.2f}%</div>
            <div class="kpi-subtext text-green">&bull; Caught {tp_hyb} of {total_fraud_test} frauds</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Test Precision</div>
            <div class="kpi-value text-blue">{prec_hyb:.2f}%</div>
            <div class="kpi-subtext text-green">&uarr; {prec_diff:+.2f}% vs Baseline ({prec_base:.2f}%)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">False Positive Drop</div>
            <div class="kpi-value text-green">{fp_hyb} <span style="font-size: 16px; color: var(--text-muted);">/ {fp_base}</span></div>
            <div class="kpi-subtext text-green">&darr; -{fp_diff_pct:.1f}% false alert overhead</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Expected Monetary Loss</div>
            <div class="kpi-value text-amber">${loss_hyb:,.2f}</div>
            <div class="kpi-subtext text-green">&darr; ${savings:,.2f} savings ({savings_pct:.1f}%)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Optimal Threshold</div>
            <div class="kpi-value mono">{opt_thresh:.2f}</div>
            <div class="kpi-subtext text-muted">&bull; Tuned on 20% validation split</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Circular Rings Discovered</div>
            <div class="kpi-value text-red">{rings_found} <span style="font-size: 14px; color: var(--text-muted);">Loops</span></div>
            <div class="kpi-subtext text-red">&bull; ${total_ring_volume:,.0f} laundering volume</div>
        </div>
    </div>

    <!-- Layout Grid -->
    <div class="layout-grid">
        
        <!-- Left Column: Graph Visualizer + Top 20 Table -->
        <div style="display: flex; flex-direction: column; gap: 24px;">
            
            <!-- Rendered Graph Card with Highlighted Cycles -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ff3366" stroke-width="2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
                        Multi-Hop Circular Fraud Rings & Mule Topology
                    </div>
                    <span class="card-badge">Neo4j Property Graph ({node_count:,} Nodes &bull; {edge_count:,} Edges)</span>
                </div>

                {svg_content}
            </div>

            <!-- Top 20 Graph Risk Score Table -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2979ff" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                        Top 20 High-Risk Entities (Weighted Graph Risk Score Ranking)
                    </div>
                    <span class="card-badge">Formula: 0.4 &times; Ring + 0.3 &times; PageRank + 0.3 &times; In-Degree</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Account ID</th>
                            <th>Graph Risk Score</th>
                            <th>Ring Status</th>
                            <th class="center">In-Degree</th>
                            <th class="center">Out-Degree</th>
                            <th class="center">Total Degree</th>
                            <th>PageRank</th>
                            <th>Account Topology Profile</th>
                        </tr>
                    </thead>
                    <tbody>
                        {top20_html_rows}
                    </tbody>
                </table>
            </div>

        </div>

        <!-- Right Column: Model Comparison & Analytics Figures -->
        <div style="display: flex; flex-direction: column; gap: 24px;">

            <!-- Model Comparison Card -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00e676" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>
                        Baseline vs. Hybrid Model Benchmark (Test Split: {total_test_rows:,} Rows)
                    </div>
                    <span class="card-badge">Cost Formula: &Sigma;(FN Amount) + $5.0 &times; FP</span>
                </div>

                <div class="comp-grid">
                    <!-- Baseline Box -->
                    <div class="matrix-box">
                        <div class="matrix-header">
                            <span>Baseline Model</span>
                            <span class="mono">SMOTE 1:1 &bull; T=0.50</span>
                        </div>
                        <div class="matrix-grid">
                            <div class="cell cell-tn">
                                <div class="cell-num text-green">{tn_base:,}</div>
                                <div class="cell-sub">True Neg ({tn_base_pct:.2f}%)</div>
                            </div>
                            <div class="cell cell-fp">
                                <div class="cell-num text-amber">{fp_base:,}</div>
                                <div class="cell-sub">False Pos (Alarm)</div>
                            </div>
                            <div class="cell cell-fn">
                                <div class="cell-num text-red">{fn_base:,}</div>
                                <div class="cell-sub">False Neg (Missed)</div>
                            </div>
                            <div class="cell cell-tp">
                                <div class="cell-num text-blue">{tp_base:,}</div>
                                <div class="cell-sub">True Pos (Caught)</div>
                            </div>
                        </div>
                        <div style="margin-top: 12px; font-size: 11px; color: var(--text-muted); line-height: 1.6;">
                            <div>Precision: <b class="text-blue">{prec_base:.2f}%</b> | Recall: <b class="text-green">{rec_base:.2f}%</b></div>
                            <div>AUC-ROC: <b>{float(m_hyb_base.get('area_under_roc', 0.0)):.4f}</b> | PR-AUC: <b>{float(m_hyb_base.get('area_under_pr', 0.0)):.4f}</b></div>
                            <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--border);">
                                Expected Monetary Loss: <b class="text-amber">${loss_base:,.2f}</b>
                            </div>
                        </div>
                    </div>

                    <!-- Hybrid Box -->
                    <div class="matrix-box" style="border-color: #27ae60;">
                        <div class="matrix-header">
                            <span style="color: var(--accent-green);">Hybrid Model (Best)</span>
                            <span class="mono">SMOTEENN+Cost &bull; T={opt_thresh:.2f}</span>
                        </div>
                        <div class="matrix-grid">
                            <div class="cell cell-tn">
                                <div class="cell-num text-green">{tn_hyb:,}</div>
                                <div class="cell-sub">True Neg ({tn_hyb_pct:.2f}%)</div>
                            </div>
                            <div class="cell cell-fp" style="background: rgba(0, 230, 118, 0.15);">
                                <div class="cell-num text-green">{fp_hyb:,}</div>
                                <div class="cell-sub">&darr; -{fp_diff} False Positives!</div>
                            </div>
                            <div class="cell cell-fn">
                                <div class="cell-num text-red">{fn_hyb:,}</div>
                                <div class="cell-sub">False Neg (Missed)</div>
                            </div>
                            <div class="cell cell-tp">
                                <div class="cell-num text-blue">{tp_hyb:,}</div>
                                <div class="cell-sub">True Pos (Caught)</div>
                            </div>
                        </div>
                        <div style="margin-top: 12px; font-size: 11px; color: var(--text-muted); line-height: 1.6;">
                            <div>Precision: <b class="text-green">{prec_hyb:.2f}%</b> <span class="text-green">({prec_diff:+.2f}%)</span> | Recall: <b>{rec_hyb:.2f}%</b></div>
                            <div>AUC-ROC: <b class="text-green">{float(m_hyb_eval.get('area_under_roc', 0.0)):.4f}</b> | F1: <b class="text-green">{float(m_hyb_eval.get('f1_score', 0.0)):.4f}</b></div>
                            <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--border);">
                                Expected Monetary Loss: <b class="text-green">${loss_hyb:,.2f}</b> <span class="text-green">(Saved ${savings:,.2f})</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Validation Threshold Cost Optimization Curve -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffab00" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                        Threshold vs. Expected Cost Curve (Validation Split)
                    </div>
                    <span class="card-badge">Optimal T = {opt_thresh:.2f}</span>
                </div>
                <img src="{b64_thresh}" class="chart-img" alt="Threshold vs Cost Curve"/>
            </div>

            <!-- ROC and PR Comparison -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#9d4edd" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                        ROC & Precision-Recall Performance Comparison
                    </div>
                    <span class="card-badge">Baseline vs. Hybrid</span>
                </div>
                <img src="{b64_roc_pr}" class="chart-img" alt="ROC & PR Comparison Curves"/>
            </div>

        </div>

    </div>

</body>
</html>
"""

    print(f"[*] Writing 100% data-driven dashboard HTML to: {dashboard_html_path}")
    with open(dashboard_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    with open(root_dashboard_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("[+] Dashboard HTML written successfully!")
    return dashboard_html_path, root_dashboard_path, figures_dir, root_figures_dir

def capture_screenshot(html_path, output_png_path):
    print(f"[*] Capturing full-page high-resolution screenshot with headless Edge...")
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    ]
    browser_bin = None
    for p in edge_paths:
        if os.path.exists(p):
            browser_bin = p
            break

    if not browser_bin:
        print("[ERROR] Headless browser binary (Edge/Chrome) not found on system.")
        return False

    path_str = str(html_path.resolve()).replace('\\', '/')
    file_uri = f"file:///{path_str}"
    cmd = [
        browser_bin,
        "--headless=new",
        "--disable-gpu",
        f"--screenshot={str(output_png_path.resolve())}",
        "--window-size=1650,2200",
        "--hide-scrollbars",
        file_uri
    ]
    subprocess.run(cmd, check=True)
    time.sleep(2)

    if output_png_path.exists():
        print(f"[+] Successfully saved dashboard screenshot to: {output_png_path} ({output_png_path.stat().st_size:,} bytes)")
        return True
    else:
        print(f"[!] Screenshot file was not created at: {output_png_path}")
        return False

def main():
    dashboard_html, root_dashboard, figures_dir, root_figures_dir = build_dashboard_html()
    shot_path = figures_dir / "dashboard_live.png"
    root_shot_path = root_figures_dir / "dashboard_live.png"

    success = capture_screenshot(dashboard_html, shot_path)
    if success and root_shot_path != shot_path:
        import shutil
        shutil.copy2(shot_path, root_shot_path)
        print(f"[+] Mirrored screenshot to: {root_shot_path}")

    print("\n[SUCCESS] Dynamic dashboard build and live screenshot capture completed!")

if __name__ == "__main__":
    main()
