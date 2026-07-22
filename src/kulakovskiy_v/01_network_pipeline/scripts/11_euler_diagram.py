#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Euler/Venn diagram of significant-gene overlap between the project's own
permutation pipeline (out5-1/candidate_gene_significant.txt) and DIAMOnD
(out6/candidate-genes-with-pvalue.txt), both filtered to p <= alpha.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib_venn import venn2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_sets import load_diamond, load_mine


def parse_args():
    p = argparse.ArgumentParser(description="Euler diagram of significant genes: own pipeline vs DIAMOnD")
    p.add_argument("--mine", default="out5-1/candidate_gene_significant.txt",
                    help="step 7 output: gene edges raw_count frequency (already filtered)")
    p.add_argument("--diamond", default="out6/candidate-genes-with-pvalue.txt",
                    help="DIAMOnD output with p_hyper column")
    p.add_argument("-a", "--alpha", type=float, default=0.05, help="significance threshold")
    p.add_argument("-o", "--output", default="analysis/diamond_vs_mine_euler.png")
    return p.parse_args()


args = parse_args()
ALPHA = args.alpha

my_significant = set(load_mine(args.mine))
diamond_significant = set(load_diamond(args.diamond, ALPHA))

common = my_significant & diamond_significant

fig, ax = plt.subplots(figsize=(7, 6))
v = venn2(
    [my_significant, diamond_significant],
    set_labels=(f"Мой пайплайн\n(permutation, p_value<={ALPHA})", f"DIAMOnD\n(p_value<={ALPHA})"),
    set_colors=("#02AB9C", "#9B2295"),
    alpha=0.7,
    ax=ax,
)

# annotate the overlap region with the actual gene IDs
if v.get_label_by_id("11") is not None and common:
    v.get_label_by_id("11").set_text(f"\n".join(sorted(common)))
    v.get_label_by_id("11").set_fontsize(8)

for label_id in ("10", "01"):
    lbl = v.get_label_by_id(label_id)
    if lbl is not None:
        lbl.set_fontsize(13)

ax.set_title(f"Пересечение значимых генов (p <= {ALPHA})\n"
             f"мой пайплайн: {len(my_significant)} | DIAMOnD: {len(diamond_significant)} | общих: {len(common)}")

plt.tight_layout()
plt.savefig(args.output, dpi=150)
print(f"mine={len(my_significant)} diamond={len(diamond_significant)} common={len(common)}")
print(f"saved to {args.output}")