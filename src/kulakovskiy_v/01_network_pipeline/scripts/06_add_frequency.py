#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Step 6 - Turn the raw null-model count (step 6) into a frequency.

frequency = raw_count / n_replicates
A low frequency means the gene rarely appears as a candidate by chance.
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(description="Add a frequency column to the candidate/null-count file")
    p.add_argument("-i", "--input", default="out4-1/candidate_gene.txt", help="step 6 output: gene edges raw_count")
    p.add_argument("-o", "--output", default="out5-1/candidate_gene.txt", help="output: gene edges raw_count frequency")
    p.add_argument("-r", "--replics", type=int, default=10000, help="number of null-model replicates (denominator)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file '{args.input}' not found (run step 6 first)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    count_lines = 0
    with open(args.input) as infile, open(args.output, "w") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                gene_id, edges, raw_count = parts[0], parts[1], parts[2]
                frequency = float(raw_count) / args.replics
                outfile.write(f"{gene_id} {edges} {raw_count} {frequency:.4f}\n")
                count_lines += 1
            else:
                outfile.write(line + "\n")

    print(f"{count_lines} genes written with frequency to {args.output}")


if __name__ == "__main__":
    main()
