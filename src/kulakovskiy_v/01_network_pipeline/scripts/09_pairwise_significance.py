#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Step 9 (independent branch) - Pairwise gene co-occurrence significance.

Builds a binary presence/absence vector for every gene across all
null-model replicates, then uses a hypergeometric test to ask: do two
genes co-occur across replicates more often than chance would predict?
Aggregates significant pairs per gene into a hub score (n-edges) and a
best p-value.

This supersedes legacy/p_value.py, which has a typo in its input path
(out3-1ы) and never actually found any files.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def parse_args():
    p = argparse.ArgumentParser(description="Pairwise co-occurrence significance across null-model replicates")
    p.add_argument("-i", "--input-dir", default="out3-1", help="folder with null-model candidate files (step 4 output)")
    p.add_argument("-o", "--output", default="analysis/gene_summary.csv", help="output CSV: gene_id, n-edges, p_value")
    p.add_argument("-m", "--min-occurrences", type=int, default=20, help="min replicates a gene must appear in to be tested")
    p.add_argument("-a", "--alpha", type=float, default=0.05, help="significance threshold for a pair's p-value")
    return p.parse_args()


def main():
    args = parse_args()

    all_files = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))
    N = len(all_files)
    print(f"Found {N} files in {args.input_dir}")
    if N == 0:
        raise SystemExit(f"No files found in '{args.input_dir}' - check the path")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("Building presence/absence vectors...")
    gene_dict = {}
    for file_idx, file_path in enumerate(all_files):
        with open(file_path) as f:
            genes_in_file = {line.strip() for line in f if line.strip()}
        for gene in genes_in_file:
            if gene not in gene_dict:
                gene_dict[gene] = np.zeros(N, dtype=int)
            gene_dict[gene][file_idx] = 1

    df_vectors = pd.DataFrame.from_dict(gene_dict, orient='index')

    genes = df_vectors.index.tolist()
    frequent_genes = [g for g in genes if df_vectors.loc[g].sum() >= args.min_occurrences]
    print(f"{len(frequent_genes)} genes appear in >= {args.min_occurrences} replicates, testing pairs...")

    edges_list = []
    for i in range(len(frequent_genes)):
        for j in range(i + 1, len(frequent_genes)):
            g1, g2 = frequent_genes[i], frequent_genes[j]
            vec1, vec2 = df_vectors.loc[g1], df_vectors.loc[g2]
            A, B = vec1.sum(), vec2.sum()
            k = (vec1 & vec2).sum()
            if k == 0:
                continue
            p_val = hypergeom.sf(k - 1, N, A, B)
            if p_val < args.alpha:
                edges_list.append({'gene_id': g1, 'p_value': p_val})
                edges_list.append({'gene_id': g2, 'p_value': p_val})

    if not edges_list:
        print("No significant co-occurring pairs found - output will be empty.")
        pd.DataFrame(columns=['gene_id', 'n-edges', 'p_value']).to_csv(args.output, index=False)
        return

    df_edges = pd.DataFrame(edges_list)
    df_final = df_edges.groupby('gene_id').agg(
        n_edges=('p_value', 'count'),
        p_value=('p_value', 'min')
    ).reset_index()
    df_final.columns = ['gene_id', 'n-edges', 'p_value']
    df_final = df_final.sort_values(by='n-edges', ascending=False)
    df_final.to_csv(args.output, index=False)

    print(f"Written to {args.output}")
    print(df_final.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
