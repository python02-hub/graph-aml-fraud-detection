#!/usr/bin/env python3
"""
Turn the AML pipeline's output CSVs into an interactive, zoomable, draggable
graph you open in a web browser (no internet needed after the one-time
`pip install pyvis`).

Run this AFTER aml_gnn.py has produced a run folder (e.g. runs/aml or
runs/hi_small), which must contain flagged_transactions.csv and
ring_candidates.csv.

Usage:
    python visualize_graph.py --run runs/aml
    python visualize_graph.py --run runs/hi_small --ring 1   # just one ring
    python visualize_graph.py --run runs/aml --max-rings 10  # more rings

Output:
    <run>/graph.html   <- double-click this file to open it in your browser
"""
import argparse
from pathlib import Path

import pandas as pd

try:
    from pyvis.network import Network
except ImportError:
    raise SystemExit("Missing dependency. Run:  pip install pyvis")


def load(run):
    run = Path(run)
    tx_path, ring_path = run / "flagged_transactions.csv", run / "ring_candidates.csv"
    if not tx_path.exists() or not ring_path.exists():
        raise SystemExit(f"Couldn't find {tx_path} and/or {ring_path}.\n"
                         f"Run aml_gnn.py first (e.g. python aml_gnn.py --demo) "
                         f"and point --run at its output folder.")
    tx = pd.read_csv(tx_path)
    rings = pd.read_csv(ring_path)
    return tx, rings


def account_key(row, prefix):
    """Match the aml_gnn.py node identity: bank + account number together."""
    return f"{row[f'{prefix}_bank']}_{row[f'{prefix}_acct']}"


def build_graph(tx, rings, ring_id, max_rings, min_edges, max_edges_per_ring):
    rings = rings.sort_values(["max_score", "n_edges"], ascending=False)
    if ring_id is not None:
        rings = rings[rings.ring_id == ring_id]
        if rings.empty:
            raise SystemExit(f"No ring with id {ring_id} in ring_candidates.csv")
    else:
        rings = rings[rings.n_edges >= min_edges].head(max_rings)
    if rings.empty:
        raise SystemExit("No rings meet the --min-edges threshold. Try lowering it, "
                         "or re-run aml_gnn.py with a higher --neg-ratio / more --epochs.")

    net = Network(height="820px", width="100%", directed=True, bgcolor="#11131a",
                  font_color="#e9ecf5", notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-4200, central_gravity=0.25, spring_length=140,
                   spring_strength=0.02, damping=0.25)

    tx["src_key"] = tx.apply(lambda r: account_key(r, "src"), axis=1)
    tx["dst_key"] = tx.apply(lambda r: account_key(r, "dst"), axis=1)

    added = set()
    acct_score = {}
    for _, r in tx.iterrows():
        acct_score[r.src_key] = max(acct_score.get(r.src_key, 0), r.gnn_score)
        acct_score[r.dst_key] = max(acct_score.get(r.dst_key, 0), r.gnn_score)

    def node_color(score, is_confirmed):
        if is_confirmed:
            return "#ff3b3b"        # confirmed laundering label in the data
        if score > 0.8:
            return "#ff9d3b"        # high model risk score, unconfirmed
        return "#5b8cff"            # normal / lower-risk account

    # ring_candidates.csv doesn't store each ring's account list (it's dropped before
    # saving in aml_gnn.py), so we rebuild connected components directly from the
    # flagged transactions themselves - this exactly mirrors what aml_gnn.py did.
    import networkx as nx
    G = nx.DiGraph()
    for _, r in tx.iterrows():
        if G.has_edge(r.src_key, r.dst_key):
            G[r.src_key][r.dst_key]["amount"] += abs(r.amount_paid)
            G[r.src_key][r.dst_key]["score"] = max(G[r.src_key][r.dst_key]["score"], r.gnn_score)
            G[r.src_key][r.dst_key]["label"] = max(G[r.src_key][r.dst_key]["label"], int(r.label))
        else:
            G.add_edge(r.src_key, r.dst_key, amount=abs(r.amount_paid),
                      score=r.gnn_score, label=int(r.label))

    comps = [c for c in nx.weakly_connected_components(G) if G.subgraph(c).number_of_edges() >= 2]
    comps.sort(key=lambda c: (max(d["score"] for _, _, d in G.subgraph(c).edges(data=True)),
                              len(c)), reverse=True)
    if ring_id is not None:
        comps = comps[ring_id - 1:ring_id] if ring_id - 1 < len(comps) else []
    else:
        comps = [c for c in comps if G.subgraph(c).number_of_edges() >= min_edges][:max_rings]
    if not comps:
        raise SystemExit("No connected clusters of flagged transactions were found to draw.")

    drawn_edges = 0
    for comp in comps:
        H = G.subgraph(comp)
        if H.number_of_edges() > max_edges_per_ring:
            keep = sorted(H.edges(data=True), key=lambda e: e[2]["score"], reverse=True)[:max_edges_per_ring]
            H = nx.DiGraph(); H.add_edges_from(keep)
            print(f"[note] trimmed a {len(comp)}-account cluster down to its "
                 f"{max_edges_per_ring} highest-risk transfers so it stays readable "
                 f"(raise --max-edges-per-ring to see more)")
        drawn_edges += H.number_of_edges()
        is_cycle = not nx.is_directed_acyclic_graph(H)
        for node in H.nodes():
            if node in added:
                continue
            added.add(node)
            sc = acct_score.get(node, 0)
            degree = H.degree(node)
            net.add_node(node, label=node[-6:], title=f"Account {node}\nmax risk score {sc:.2f}",
                        color=node_color(sc, False), size=14 + 4 * degree,
                        borderWidth=3 if is_cycle else 1)
        for u, v, d in H.edges(data=True):
            confirmed = d["label"] == 1
            color = "#ff3b3b" if confirmed else ("#ff9d3b" if d["score"] > 0.8 else "#6f7694")
            net.add_edge(u, v, value=max(d["score"], 0.05),
                        title=f"${d['amount']:,.0f}  |  model risk {d['score']:.2f}"
                              f"{'  |  CONFIRMED LAUNDERING' if confirmed else ''}",
                        color=color, arrows="to")

    net.set_options("""
    {
      "nodes": {"shape": "dot", "font": {"size": 12}},
      "interaction": {"hover": true, "tooltipDelay": 80, "navigationButtons": true, "keyboard": true},
      "physics": {"stabilization": {"iterations": 150}}
    }
    """)
    return net, len(comps), drawn_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output folder from aml_gnn.py, e.g. runs/aml")
    ap.add_argument("--ring", type=int, default=None, help="draw only this ring_id from ring_candidates.csv")
    ap.add_argument("--max-rings", type=int, default=15, help="how many top rings to include")
    ap.add_argument("--min-edges", type=int, default=2, help="skip clusters smaller than this")
    ap.add_argument("--max-edges-per-ring", type=int, default=80,
                    help="if a cluster has more transactions than this, keep only its "
                         "highest-risk edges so the graph stays readable")
    args = ap.parse_args()

    tx, rings = load(args.run)
    net, n_rings, n_edges = build_graph(tx, rings, args.ring, args.max_rings, args.min_edges,
                                        args.max_edges_per_ring)
    out_path = Path(args.run) / "graph.html"
    # Write explicitly as UTF-8: pyvis's own write_html() uses the OS default
    # encoding, which on Windows (cp1252) can't handle some characters it
    # generates and raises UnicodeEncodeError.
    out_path.write_text(net.generate_html(notebook=False), encoding="utf-8")
    print(f"Drew {n_rings} ring(s), {n_edges} transactions.")
    print(f"\nOpen this file in any web browser (double-click it, or drag it into a browser tab):")
    print(f"  {out_path.resolve()}")
    print("\nLegend: red = confirmed laundering label | orange = high model risk, unconfirmed | "
          "blue = lower risk | thick node border = part of a circular transfer chain.")
    print("Drag nodes, scroll to zoom, hover an edge for amount/score.")


if __name__ == "__main__":
    main()
