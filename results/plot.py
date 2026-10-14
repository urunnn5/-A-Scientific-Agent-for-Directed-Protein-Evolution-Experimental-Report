#!/usr/bin/env python3
"""Draw a real training-evidence subgraph from the final DMS project.

Place this file in DMS/, then run:
    python plot_knowledge_graph.py --root . --out-dir results/knowledge_graph

Requirements are already listed in DMS/requirements.txt:
    pandas, matplotlib, networkx, biopython

No API request or MLP prediction is made. All fitness values come from
data/train.csv. The figure shows a readable subset of the full GraphKnowledge
graph, not an invented causal diagram. Property values live on graph edges.
English node labels avoid a dependency on a Chinese font on other computers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath
import pandas as pd


def project_root(requested: Path | None) -> Path:
    if requested is not None:
        options = [requested, requested / "DMS"]
    else:
        here = Path(__file__).resolve().parent
        options = [here, here / "DMS", Path.cwd(), Path.cwd() / "DMS"]
    for folder in options:
        folder = folder.resolve()
        if (folder / "agent/graph.py").is_file() and (folder / "data/train.csv").is_file():
            return folder
    raise SystemExit("DMS project not found. Pass --root /path/to/DMS.")


def relation_data(graph, source: str, target: str, relation: str):
    for attrs in (graph.get_edge_data(source, target) or {}).values():
        if attrs.get("relation") == relation:
            return attrs
    raise ValueError(f"Missing actual graph edge: {source} -> {target} ({relation})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="DMS project directory")
    parser.add_argument("--out-dir", type=Path, default=Path("results/knowledge_graph"))
    parser.add_argument("--mutation", default="L221V", help="Mutation with a training single observation")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    root = project_root(args.root)
    sys.path.insert(0, str(root))
    from agent.graph import GraphKnowledge

    train = pd.read_csv(root / "data/train.csv")
    knowledge = GraphKnowledge(root / "data/aa_props.csv", train)
    graph = knowledge.graph
    name = args.mutation
    mutation = f"mutation:{name}"
    single = f"variant:{name}"
    if mutation not in graph or single not in graph:
        raise SystemExit(f"{name} has no training single observation in this graph.")
    evidence = knowledge.evidence(name)["mutations"][0]
    multi = [o for o in knowledge._observations(name) if o["mutation_count"] > 1 and o["high_top10"]]
    if not multi:
        raise SystemExit(f"{name} has no high-scoring multi-mutant example in the selected graph.")
    combo = f"variant:{multi[0]['variant']}"
    old, new = f"aa:{name[0]}", f"aa:{name[-1]}"
    position = f"position:{evidence['position']}"
    properties = [f"property:{p}" for p in ("Volume", "Hydropathy", "BetaSheet")]
    observations = [f"fitness:{name}", f"fitness:{multi[0]['variant']}"]
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    node_counts = Counter(a["type"] for _, a in graph.nodes(data=True))
    edge_counts = Counter(a["relation"] for *_, a in graph.edges(data=True))
    stats = {**knowledge.stats, "node_types": dict(node_counts), "edge_relations": dict(edge_counts),
             "focus_mutation": name, "focus_evidence": evidence,
             "note": "Full graph counts include parallel edges from the original MultiDiGraph."}
    (out_dir / "graph_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(12.6, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(-0.5, 13.0)
    ax.set_ylim(-0.8, 8.4)
    ax.axis("off")

    colors = {"Variant": "#E8EFF9", "FitnessObservation": "#E5F3EA", "Mutation": "#FCEBD8",
              "Position": "#F1EDF8", "AminoAcid": "#FFF4D5", "Property": "#EAF4F5"}
    specs = {
        observations[1]: (1.15, 6.65, 2.65, 0.85),
        combo: (4.3, 6.65, 2.95, 0.85),
        observations[0]: (1.15, 4.65, 2.65, 0.85),
        single: (4.3, 4.65, 2.2, 0.85),
        mutation: (7.0, 5.65, 1.85, 0.92),
        position: (10.05, 7.35, 2.15, 0.85),
        old: (6.15, 2.1, 1.75, 0.85),
        new: (8.2, 2.1, 1.75, 0.85),
        properties[0]: (11.15, 3.85, 2.45, 0.82),
        properties[1]: (11.15, 2.1, 2.45, 0.82),
        properties[2]: (11.15, 0.35, 2.45, 0.82),
    }

    def draw_node(node: str):
        x, y, w, h = specs[node]
        attrs = graph.nodes[node]
        kind = attrs["type"]
        if kind == "FitnessObservation":
            label = f"Measured train fitness\n{attrs['score']:.5f}"
        elif kind == "Variant":
            label = f"Variant\n{attrs['mutant']}"
        elif kind == "Mutation":
            label = f"Mutation\n{attrs['name']}"
        elif kind == "Position":
            label = f"Position\n{attrs['index']}"
        elif kind == "AminoAcid":
            label = f"Amino acid\n{attrs['symbol']}"
        else:
            label = f"Property\n{attrs['name']}"
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                             facecolor=colors[kind], edgecolor="#8B9AAC", linewidth=1.15, zorder=4)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=14.3, zorder=5)

    def draw_edge(source, target, relation, label=None, curvature=0.0, label_at=None):
        data = relation_data(graph, source, target, relation)
        sx, sy, sw, sh = specs[source]
        tx, ty, tw, th = specs[target]
        if abs(tx-sx) > abs(ty-sy):
            sign = 1 if tx > sx else -1
            start = (sx + sign*sw/2, sy)
            end = (tx - sign*tw/2, ty)
        else:
            sign = 1 if ty > sy else -1
            start = (sx, sy + sign*sh/2)
            end = (tx, ty - sign*th/2)
        if relation == "has_high_single_observation":
            route = PlotPath([(sx, sy-sh/2), (sx, 3.45), (tx, 3.45), (tx, ty-th/2)],
                             [PlotPath.MOVETO, PlotPath.LINETO, PlotPath.LINETO, PlotPath.LINETO])
            arrow = FancyArrowPatch(path=route, arrowstyle="-|>", mutation_scale=12,
                                    color="#62748A", linewidth=1.05, zorder=1)
        else:
            arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, shrinkA=4, shrinkB=5,
                                    connectionstyle=f"arc3,rad={curvature}", color="#62748A",
                                    linewidth=1.05, zorder=1)
        ax.add_patch(arrow)
        lx, ly = label_at if label_at else ((start[0]+end[0])/2, (start[1]+end[1])/2 + 0.13)
        ax.text(lx, ly, label or relation, ha="center", va="center", fontsize=11.0,
                color="#3D4A5A", bbox={"facecolor":"white", "edgecolor":"none", "pad":1.4}, zorder=3)
        return data

    draw_edge(combo, observations[1], "has_fitness", label_at=(2.65, 7.28))
    draw_edge(single, observations[0], "has_fitness", label_at=(2.75, 5.30))
    draw_edge(combo, mutation, "contains", label_at=(6.25, 7.28))
    draw_edge(single, mutation, "contains", label_at=(5.80, 4.00))
    draw_edge(mutation, position, "occurs_at", label_at=(8.5, 7.0))
    draw_edge(mutation, old, "from_aa", label_at=(6.45, 3.45))
    draw_edge(mutation, new, "to_aa", label_at=(7.95, 3.45))
    for i, prop in enumerate(properties):
        before = relation_data(graph, old, prop, "has_property")["value"]
        after = relation_data(graph, new, prop, "has_property")["value"]
        draw_edge(new, prop, "has_property", f"has_property\n{after:g}",
                  label_at=(9.35, specs[prop][1]+0.63), curvature=0.03*(i-1))
        # L values are displayed separately to keep the figure readable; both
        # before and after values are checked against actual full graph edges.
        ax.text(11.15, specs[prop][1]-0.57, f"{name[0]}={before:g}; {name[-1]}={after:g}; delta={after-before:+g}",
                ha="center", va="center", fontsize=11.7, color="#465260")
    if any(a.get("relation") == "has_high_single_observation" for a in (graph.get_edge_data(mutation, observations[0]) or {}).values()):
        draw_edge(mutation, observations[0], "has_high_single_observation",
                  "has_high_single_observation", curvature=-0.35, label_at=(3.7, 3.65))
    for node in specs:
        draw_node(node)
    fig.suptitle(f"Training evidence subgraph for {name}", x=0.5, y=0.98, fontsize=17, fontweight="bold")
    fig.text(0.5, 0.925,
             f"Full graph: {knowledge.stats['selected_variants']:,} variants | {knowledge.stats['nodes']:,} nodes | "
             f"{knowledge.stats['edges']:,} edges | top-10% threshold = {knowledge.threshold:.5f}",
             ha="center", fontsize=13.3, color="#485768")
    fig.text(0.5, 0.035, "All scores are measured training observations. High scores do not establish improvement over WT.",
             ha="center", fontsize=12.0, color="#485768")
    fig.text(0.5, 0.010, "The plot shows 3 of 6 property types. No unverified Mutation-improves-Fitness edge is added.",
             ha="center", fontsize=11.8, color="#485768")
    fig.subplots_adjust(left=0.025, right=0.99, top=0.88, bottom=0.08)
    output = out_dir / "knowledge_graph.png"
    temporary = out_dir / "knowledge_graph_rendering.png"
    fig.savefig(temporary, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    temporary.replace(output)
    print(f"Saved: {output}")
    print(json.dumps(knowledge.stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
