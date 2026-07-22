#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Step 1 - Explore the PPI network and the seed gene list.

Loads a protein-protein interaction network and a gene list, reports basic
network stats, finds the genes that are actually present in the network
(the real "seed" set used by step 2), and plots the degree distribution.
"""

import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx


def parse_args():
    p = argparse.ArgumentParser(description="Explore a PPI network and intersect it with a gene list")
    p.add_argument("-n", "--network", default="/data/networks/HS-PI.clean.tsv", help="path/to/network edgelist")
    p.add_argument("-g", "--genes", default="/data/genes/human-orthologs-02.txt", help="path/to/gene list")
    p.add_argument("-o", "--output", default="common_nodes.txt", help="where to write the seed genes found in the network")
    p.add_argument("-p", "--pics-dir", default="pics", help="folder to save degree-distribution plots")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.pics_dir, exist_ok=True)

    G = nx.read_edgelist(args.network)
    print("No of nodes:", G.number_of_nodes())
    print("No of edges:", G.number_of_edges())

    degrees = dict(G.degree())
    graph_nodes = set(degrees.keys())

    with open(args.genes) as fh:
        genes = [line.strip() for line in fh if line.strip()]
    genes_set = set(genes)

    common_nodes = graph_nodes & genes_set
    print(f"Number of common nodes: {len(common_nodes)}")

    with open(args.output, "w") as f:
        for node in sorted(common_nodes):
            f.write(node + "\n")
    print(f"Seed genes written to {args.output}")

    plt.figure(figsize=(6, 4))
    plt.hist(degrees.values(), bins=50)
    plt.xlabel("Степень узла")
    plt.ylabel("Число генов")
    plt.savefig(os.path.join(args.pics_dir, "degree_distribution.png"))

    plt.figure(figsize=(6, 4))
    plt.hist(degrees.values(), bins=50)
    plt.xlabel("Степень узла")
    plt.ylabel("log10(Числа генов)")
    plt.yscale("log")
    plt.savefig(os.path.join(args.pics_dir, "degree_distribution2.png"))


if __name__ == "__main__":
    main()
