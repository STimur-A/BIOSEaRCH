#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Step 3 - Generate random seed-gene sets (null model).

Draws `--replics` random gene sets of size `--ngenes` from the network's
node list. Each replicate is later fed through step 2's ranking logic
(via step 4) to see how often each gene shows up as a "candidate" purely
by chance.
"""

import argparse

import networkx as nx
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Generate N random gene sets for a given network")
    p.add_argument("-n", "--network", required=True, help="path/to/network")
    p.add_argument("-g", "--ngenes", required=True, type=int, help="size of each random gene set (match real seed-set size)")
    p.add_argument("-r", "--replics", default=1000, type=int, help="number of random replicates to generate")
    p.add_argument("-o", "--output", required=True, help="output prefix, e.g. out2/rg -> out2/rg-0.txt, out2/rg-1.txt, ...")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(42)

    G = nx.read_edgelist(args.network)
    nodes = list(G.nodes())

    for i in range(args.replics):
        random_genes = np.random.choice(nodes, args.ngenes, replace=False)
        with open(f"{args.output}-{i}.txt", "w") as fh:
            for g in random_genes:
                print(g, file=fh)


if __name__ == "__main__":
    main()
