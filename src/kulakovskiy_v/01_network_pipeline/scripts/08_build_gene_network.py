#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Step 8 (independent branch) - Build a gene co-occurrence network from the
null-model files and analyze it.

Instead of testing significance for each real candidate gene individually
(steps 6-8), this treats every null-model replicate as an "experiment" and
asks: which genes tend to occur together across experiments (hub genes,
correlated genes, communities)? Produces centrality, correlation, clustering
and plots under --output.
"""

import os, glob, argparse
import pandas as pd
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser(description="Gene co-occurrence network analysis across null-model replicates")
    p.add_argument("-i", "--input", default="out3", help="folder with null-model candidate files (step 4 output)")
    p.add_argument("-o", "--output", default="analysis", help="output folder")
    p.add_argument("-m", "--min_exp", type=int, default=20, help="min replicates a gene must appear in to be analyzed")
    p.add_argument("-c", "--corr_thresh", type=float, default=0.6, help="Spearman correlation threshold for network edges")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    out = lambda f: os.path.join(args.output, f)

    print("1. Loading data...")
    files = glob.glob(f"{args.input}/*.txt")
    if not files:
        print(f"Error: No .txt files found in '{args.input}' folder."); return

    dfs = [pd.read_csv(f, sep=r'\s+', header=None, names=['Gene', i], index_col='Gene') for i, f in enumerate(files)]
    matrix = pd.concat(dfs, axis=1).fillna(0)
    matrix.to_csv(out("matrix.csv"))

    print("2. Calculating statistics...")
    counts = (matrix > 0).sum(axis=1)
    stats = pd.DataFrame({'count': counts, 'mean': matrix.mean(axis=1), 'median': matrix.median(axis=1),
                          'max': matrix.max(axis=1), 'min': matrix.min(axis=1), 'std': matrix.std(axis=1)})
    stats.to_csv(out("gene_statistics.csv"))

    print("3. Filtering & Correlation...")
    filtered_matrix = matrix[counts >= args.min_exp]
    if filtered_matrix.shape[0] < 2:
        print("Error: Too few genes passed the min_exp filter. Decrease --min_exp."); return

    corr_matrix = filtered_matrix.T.corr(method='spearman')

    corr_flat = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .rename_axis(['Gene1', 'Gene2'])
        .reset_index(name='Correlation')
    )
    corr_flat = corr_flat.sort_values(by='Correlation', ascending=False)
    corr_flat.to_csv(out("correlations.csv"), index=False)

    max_corr = corr_flat['Correlation'].max()
    print(f"   [INFO] Maximum correlation in your data: {max_corr:.4f}")

    print("4. Building Network...")
    edges = corr_flat[corr_flat['Correlation'] >= args.corr_thresh]
    print(f"   [INFO] Edges found above threshold {args.corr_thresh}: {len(edges)}")

    G = nx.from_pandas_edgelist(edges, 'Gene1', 'Gene2', edge_attr='Correlation')
    nx.write_graphml(G, out("network.graphml"))

    if len(G.nodes) == 0:
        print(f"Network is empty with threshold {args.corr_thresh}. Run again with --corr_thresh lower than {max_corr:.2f}")
        return

    print("5. Network Metrics & 6. Communities...")
    centrality = pd.DataFrame({
        'Degree': dict(G.degree()),
        'Weighted_Degree': dict(G.degree(weight='Correlation')),
        'Betweenness': nx.betweenness_centrality(G),
        'Closeness': nx.closeness_centrality(G),
        'Eigenvector': nx.eigenvector_centrality(G, max_iter=1000, tol=1e-3),
        'PageRank': nx.pagerank(G)
    })
    centrality.to_csv(out("centrality.csv"))

    louvain_sets = nx_comm.louvain_communities(G, weight='Correlation')
    clusters = {node: idx for idx, community_set in enumerate(louvain_sets) for node in community_set}
    pd.DataFrame.from_dict(clusters, orient='index', columns=['Cluster_ID']).to_csv(out("clusters.csv"))

    print("7. Visualizations...")
    plt.figure()
    sns.histplot(corr_flat['Correlation'].dropna(), bins=50)
    plt.title("Correlation Distribution")
    plt.savefig(out("histogram.png")); plt.close()

    plt.figure(figsize=(12, 6))
    stats['count'].nlargest(50).plot(kind='bar')
    plt.title("Top-50 frequently occurring genes")
    plt.tight_layout()
    plt.savefig(out("top_genes.png")); plt.close()

    top_nodes = list(dict(sorted(G.degree(), key=lambda item: item[1], reverse=True)[:50]).keys())
    if len(top_nodes) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix.loc[top_nodes, top_nodes],
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8}
        )
        plt.tight_layout()
        plt.savefig(out("heatmap.png")); plt.close()

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15, seed=42)
    node_color = [clusters.get(node, 0) for node in G.nodes()]

    deg_dict = dict(G.degree())
    max_deg = max(deg_dict.values()) if deg_dict else 1
    node_size = [(deg_dict[node] / max_deg) * 500 + 100 for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, cmap=plt.cm.tab20, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=[d['Correlation'] * 2 for u, v, d in G.edges(data=True)])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out("network.png")); plt.close()

    print("Done! All results successfully saved to the output folder.")


if __name__ == "__main__":
    main()
