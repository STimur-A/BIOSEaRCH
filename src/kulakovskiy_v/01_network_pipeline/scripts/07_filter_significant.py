#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Step 7 - Filter and sort genes by null-model frequency.

Keeps genes whose null-model frequency is <= --threshold (i.e. genes that
rarely appear by chance -> more likely to be real signal), then sorts by
number of real connections (descending), tie-broken by frequency (ascending).
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(description="Filter candidate genes by null-model frequency")
    p.add_argument("-i", "--input", default="out5-1/candidate_gene.txt", help="step 7 output: gene edges raw_count frequency")
    p.add_argument("-o", "--output", default="out5-1/candidate_gene_significant.txt", help="filtered/sorted output")
    p.add_argument("-t", "--threshold", type=float, default=0.05, help="keep genes with frequency <= threshold")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file '{args.input}' not found (run step 7 first)")

    genes_list = []
    with open(args.input) as infile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            try:
                gene_id = parts[0]
                edges = int(parts[1])
                raw_count = int(parts[2])
                frequency = float(parts[3])
            except ValueError:
                continue
            if frequency <= args.threshold:
                genes_list.append((gene_id, edges, raw_count, frequency))

    genes_list.sort(key=lambda x: (-x[1], x[3]))

    with open(args.output, "w") as outfile:
        for gene in genes_list:
            outfile.write(f"{gene[0]} {gene[1]} {gene[2]} {gene[3]:.4f}\n")

    print(f"{len(genes_list)} genes with frequency <= {args.threshold} written to {args.output}")
    if not genes_list:
        print("No genes passed the filter - try a higher --threshold, "
              "or check that --input actually has varying frequency values.")


if __name__ == "__main__":
    main()
