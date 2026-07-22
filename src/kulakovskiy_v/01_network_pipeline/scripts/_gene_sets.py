#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Shared helper for scripts 10 (bar chart) and 11 (Euler diagram): load the
two significant-gene sets being compared throughout this project — the
project's own permutation pipeline vs DIAMOnD — from their respective
output files.
"""


def load_mine(path):
    """out5-1/candidate_gene_significant.txt -> {gene_id: frequency}"""
    genes = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 4:
                gene_id, _edges, _raw, freq = parts
                genes[gene_id] = float(freq)
    return genes


def load_diamond(path, alpha):
    """out6/candidate-genes-with-pvalue.txt -> {gene_id: p_hyper}, filtered to p <= alpha"""
    genes = {}
    with open(path) as fh:
        next(fh)  # header
        for line in fh:
            _rank, gene_id, p = line.strip().split("\t")
            p = float(p)
            if p <= alpha:
                genes[gene_id] = p
    return genes
