#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Compare gene significance between the project's own permutation-test
pipeline (out5-1/candidate_gene_significant.txt, frequency used as an
empirical p-value) and the DIAMOnD algorithm's analytic connectivity
p-value (out6/candidate-genes-with-pvalue.txt). Both are pre-filtered to
p <= 0.05.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_sets import load_diamond, load_mine


def parse_args():
    p = argparse.ArgumentParser(description="Compare gene significance: own permutation pipeline vs DIAMOnD")
    p.add_argument("--mine", default="out5-1/candidate_gene_significant.txt",
                    help="step 7 output: gene edges raw_count frequency (already filtered)")
    p.add_argument("--diamond", default="out6/candidate-genes-with-pvalue.txt",
                    help="DIAMOnD output with p_hyper column (see external/DIAMOnD_with_pvalue.py)")
    p.add_argument("-a", "--alpha", type=float, default=0.05, help="significance threshold")
    p.add_argument("-o", "--output", default="analysis/diamond_vs_mine_pvalue_comparison.png")
    return p.parse_args()


args = parse_args()
ALPHA = args.alpha

my_significant = load_mine(args.mine)
diamond_pvalues = load_diamond(args.diamond, ALPHA)

common_genes = sorted(set(my_significant) & set(diamond_pvalues),
                       key=lambda g: my_significant[g])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [3, 2]})

# --- Panel 1: paired p-value comparison for genes significant in BOTH methods ---
ax = axes[0]
x = np.arange(len(common_genes))
width = 0.35

# frequency of exactly 0.0000 can't be shown on a log axis -> floor it for plotting only
my_vals = [max(my_significant[g], 1e-4) for g in common_genes]
dia_vals = [diamond_pvalues[g] for g in common_genes]

ax.bar(x - width/2, my_vals, width, label="Мой пайплайн (частотный p, permutation)", color="#02AB9C")
ax.bar(x + width/2, dia_vals, width, label="DIAMOnD (p_hyper)", color="#9B2295")

ax.set_yscale("log")
ax.set_ylim(1e-15, 1)
ax.axhline(ALPHA, color="red", linestyle="--", linewidth=1.2, label=f"порог p = {ALPHA}")
ax.set_xticks(x)
ax.set_xticklabels(common_genes, rotation=0)
ax.set_xlabel("Ген (Entrez ID)")
ax.set_ylabel("p-value (log scale)")
ax.set_title(f"Гены, значимые в ОБОИХ методах (n={len(common_genes)})")
ax.legend(fontsize=8, loc="lower left", framealpha=0.95)

for i, g in enumerate(common_genes):
    ax.annotate(f"{my_significant[g]:.4f}", (x[i]-width/2, my_vals[i]), textcoords="offset points",
                xytext=(0, 4), ha='center', fontsize=7)
    ax.annotate(f"{dia_vals[i]:.1e}", (x[i]+width/2, dia_vals[i]), textcoords="offset points",
                xytext=(0, 4), ha='center', fontsize=7)

# --- Panel 2: overlap summary ---
ax2 = axes[1]
ax2.axis("off")
n_mine = len(my_significant)
n_diamond = len(diamond_pvalues)
n_common = len(common_genes)

summary_text = (
    f"Мой пайплайн (p <= {ALPHA}): {n_mine} генов\n"
    f"DIAMOnD (p <= {ALPHA}): {n_diamond} генов\n"
    f"Пересечение: {n_common} генов\n\n"
    f"Общие гены:\n{', '.join(common_genes)}\n\n"
    "DIAMOnD жадно добавляет узлы по\n"
    "наилучшему p на каждом шаге - почти\n"
    "все отобранные узлы статистически\n"
    "значимы по его собственному тесту.\n\n"
    "Мой permutation-тест намного строже:\n"
    f"из 100 кандидатов только {n_mine} показали\n"
    "частоту случайного совпадения <= 0.05."
)
ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, fontsize=10,
         va="top", ha="left", family="monospace")
ax2.set_title("Сводка сравнения")

plt.tight_layout()
plt.savefig(args.output, dpi=150)
print(f"my_significant={n_mine} diamond_significant={n_diamond} common={n_common}")
print(f"saved to {args.output}")
