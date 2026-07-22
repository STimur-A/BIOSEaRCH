#!/bin/bash
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

# Step 4 - Run step 2's candidate-ranking logic on every random gene set
# produced by step 3, in parallel. This builds the null-model distribution
# used by steps 6-8 to test significance.
#
# Usage:
#   ./04_run_null_model.sh <network> <random_sets_prefix> <output_dir> <n_replicates>
#
# Example (matches the 10000-replicate run used in this project):
#   ./04_run_null_model.sh /data/networks/HS-PI.clean.tsv out2-1/rg out3-1 10000
set -euo pipefail

NETWORK=$1
RANDOM_PREFIX=$2   # e.g. out2-1/rg  (files named rg-0.txt, rg-1.txt, ...)
OUT_DIR=$3
N=$4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

seq 0 $((N - 1)) | parallel \
    python3 "$SCRIPT_DIR/02_predict_candidates.py" \
        -n "$NETWORK" \
        -g "${RANDOM_PREFIX}-{}.txt" \
        -o "$OUT_DIR/candidate-genes-{}.txt"
