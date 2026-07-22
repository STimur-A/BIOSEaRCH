#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
EN: Defense-slide figure: AUC spread per method plus candidate-gene reliability.

slide_figure.py
===========================================================================
Картинка для слайда защиты, в формате "разброс AUC по методам + надёжность
генов-кандидатов" (по образцу слайда 11 презентации "Большие вызовы").

ЗАПУСК:
  python3 slide_figure.py -q МЕТРИКИ/quality_all_models.csv -c results/consensus_top100_named.csv -o МЕТРИКИ/slide_final.png
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("-q", "--quality", default="МЕТРИКИ/quality_all_models.csv")
p.add_argument("-c", "--consensus", default="results/consensus_top100_named.csv")
p.add_argument("-o", "--output", default="МЕТРИКИ/slide_final.png")
p.add_argument("-n", "--ntop", type=int, default=10)
args = p.parse_args()

q = pd.read_csv(args.quality)
c = pd.read_csv(args.consensus)
n_total_lists = int(c["sources"].str.split(",").str.len().max())  # на всякий случай
# точнее берём максимум n_lists как знаменатель, если он совпадает с числом уникальных файлов
n_total = 21  # 3 алгоритма x 7 размерностей

MODEL_ORDER = ["svm", "logreg", "tree"]
MODEL_LABEL = {"svm": "svm", "logreg": "logreg", "tree": "dtree"}

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# ---------- левая панель: boxplot AUC по методам ----------
ax = axes[0]
data = [q[q["model"] == m]["roc_auc"].values for m in MODEL_ORDER]
bp = ax.boxplot(data, tick_labels=[MODEL_LABEL[m] for m in MODEL_ORDER],
                patch_artist=True, showmeans=True,
                meanprops=dict(marker="^", markerfacecolor="green", markeredgecolor="green", markersize=8),
                medianprops=dict(color="darkorange", linewidth=2))
for patch in bp["boxes"]:
    patch.set_facecolor("white"); patch.set_edgecolor("black")
ax.set_ylabel("AUC (по всем embedding size)", fontsize=11)
ax.set_title("Разброс AUC каждого метода по всем размерам эмбеддинга", fontsize=12)
ax.grid(alpha=0.25)

# ---------- правая панель: надёжность топ-генов ----------
ax = axes[1]
top = c.sort_values("n_lists", ascending=False).head(args.ntop).iloc[::-1]
bars = ax.barh(top["symbol"], top["n_lists"], color="#3E9C8F")
ax.set_xlabel(f"Число запусков, в которых ген попал в топ-100 (из {n_total})", fontsize=11, fontweight="bold")
ax.set_title("Надёжность найденных генов-кандидатов", fontsize=13, fontweight="bold")
for b, v in zip(bars, top["n_lists"]):
    ax.text(v + 0.15, b.get_y() + b.get_height()/2, f"{v}/{n_total}",
            va="center", fontsize=9)
ax.set_xlim(0, n_total + 2)

plt.tight_layout()
plt.savefig(args.output, dpi=170, bbox_inches="tight")
print(f"Сохранено -> {args.output}")

print("\nТоп для текста слайда:")
for _, r in top.iloc[::-1].iterrows():
    print(f"  {r['symbol']:<8} {r['n_lists']}/{n_total}  {r['full_name']}")
