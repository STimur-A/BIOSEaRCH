#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
EN: Plot ROC-AUC of the three classifiers versus embedding dimensionality (Y axis fixed to 0..1).

plot_roc_auc.py
===========================================================================
Отдельная картинка: ROC-AUC трёх алгоритмов по размерностям, ось Y от 0 до 1.

ЗАПУСК:
  python3 plot_roc_auc.py -i МЕТРИКИ/quality_all_models.csv -o МЕТРИКИ/roc_auc_0to1.png
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MODELS = {
    "logreg": ("Логистическая регрессия", "#4C72B0"),
    "svm":    ("Опорные векторы (SVM)",   "#DD8452"),
    "tree":   ("Дерево решений",          "#55A868"),
}

p = argparse.ArgumentParser()
p.add_argument("-i", "--input", default="МЕТРИКИ/quality_all_models.csv")
p.add_argument("-o", "--output", default="МЕТРИКИ/roc_auc_0to1.png")
args = p.parse_args()

df = pd.read_csv(args.input)
dims = sorted(df["dim"].unique())
x = np.arange(len(dims))

fig, ax = plt.subplots(figsize=(9, 6.5))
for key, (label, color) in MODELS.items():
    sub = df[df["model"] == key].sort_values("dim")
    ax.plot(x, sub["roc_auc"], "o-", lw=2.5, ms=9, color=color, label=label)
    for xi, v in zip(x, sub["roc_auc"]):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=color)

ax.set_xticks(x); ax.set_xticklabels([str(d) for d in dims])
ax.set_xlabel("Размерность эмбеддинга", fontsize=12)
ax.set_ylabel("ROC-AUC", fontsize=12)
ax.set_ylim(0, 1.03)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.6, label="случайное угадывание (0.5)")
ax.set_title("Зависимость ROC-AUC трёх алгоритмов от размерности эмбеддинга",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(args.output, dpi=170, bbox_inches="tight")
print(f"Сохранено -> {args.output}")
