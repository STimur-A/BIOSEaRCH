#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Step 2 - Predict candidate genes ("guilt by association").

For a set of seed genes, rank their network neighbors by how many seed genes
each neighbor connects to. Genes with many connections to the seed set are
the candidate genes. This same script is reused unmodified in step 4 to
rank neighbors of each *random* seed set, which builds the null model.
"""

import argparse
from collections import Counter

import networkx as nx


def parse_args():
    p = argparse.ArgumentParser(description="Predict candidate genes by seed-gene connectivity")
    p.add_argument("-n", "--network", required=True, help="path/to/network")
    p.add_argument("-t", "--ntop", type=int, default=100, help="number of top candidate genes to keep")
    p.add_argument("-g", "--genes", required=True, help="path/to/seed genes")
    p.add_argument("-o", "--output", required=True, help="path/to/file to save output")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.genes) as fh:
        genes = [line.strip() for line in fh]

    G = nx.read_edgelist(args.network)

    # Keep only seed genes actually present in the network
    genes = [g for g in genes if g in G]

    # Count, for every neighbor of a seed gene, how many seed genes it touches
    nb_counts = Counter()
    for gene in genes:
        for nb in G.neighbors(gene):
            if nb not in genes:
                nb_counts[nb] += 1

    ranked_nb = sorted(nb_counts.items(), key=lambda x: -x[1])

    with open(args.output, "w") as fh:
        for gene, cnt in ranked_nb[: args.ntop]:
            print(f"{gene} {cnt}", file=fh)


if __name__ == "__main__":
    main()
