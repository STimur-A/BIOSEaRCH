#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Step 5 - Count how often each real candidate gene shows up in the null model.

For every gene in the real candidate list (step 2), count in how many of the
null-model files (step 4) it also appears as a candidate. A gene that shows
up in almost every random replicate is not a specific finding; a gene that
rarely shows up at random but does appear in the real run is a stronger
candidate (tested formally in step 8).
"""

import argparse
import glob
import os
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(description="Count real-candidate occurrence in the null model")
    p.add_argument("-i", "--null-dir", default="out3-1", help="folder with null-model candidate files (step 4 output)")
    p.add_argument("-c", "--candidates", default="out1-1/candidate-genes.txt", help="real candidate gene file (step 2 output)")
    p.add_argument("-o", "--output", default="out4-1/candidate_gene.txt", help="output file: gene edges raw_count")
    return p.parse_args()


def main():
    args = parse_args()

    all_files = sorted(glob.glob(os.path.join(args.null_dir, "*.txt")))
    print(f"Found {len(all_files)} null-model files in {args.null_dir}")
    if not all_files:
        raise SystemExit(f"No files found in '{args.null_dir}'")

    occurrence_counts = Counter()
    for file_path in all_files:
        with open(file_path) as f:
            genes_in_file = {line.split()[0] for line in f if line.strip()}
        for gene in genes_in_file:
            occurrence_counts[gene] += 1

    if not os.path.exists(args.candidates):
        raise SystemExit(f"Candidates file '{args.candidates}' not found")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    matches = 0
    with open(args.candidates) as infile, open(args.output, "w") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            gene_id = parts[0]
            edges = parts[1] if len(parts) > 1 else "0"
            count = occurrence_counts.get(gene_id, 0)
            if count > 0:
                matches += 1
            outfile.write(f"{gene_id} {edges} {count}\n")

    print(f"{matches} candidate genes matched at least once in the null model")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
