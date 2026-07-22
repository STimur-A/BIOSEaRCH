#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
EN: Four-panel consensus overview figure: algorithm convergence, robustness, top-20 genes, and a gene x list heatmap.

plot_consensus.py
===========================================================================
Итоговая картинка по консенсус-анализу: 4 панели, рассказывающие всю работу.

  A) Сходимость алгоритмов с ростом размерности эмбеддинга
  B) Устойчивость каждого алгоритма к смене размерности
  C) Топ-20 консенсусных генов (в скольки из 15 списков встретился)
  D) Тепловая карта: топ-генов x 15 списков (где именно всплыл каждый ген)

ЗАПУСК:
  python3 plot_consensus.py -r results -o results/consensus_overview.png
"""

import argparse
import json
import os
import urllib.parse
import urllib.request
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

DIMS = [8, 16, 32, 64, 128, 256, 512]
MODELS = ["logreg", "svm", "tree"]
MODEL_RU = {"logreg": "Логистическая\nрегрессия", "svm": "Опорные\nвекторы (SVM)", "tree": "Дерево\nрешений"}


def parse_args():
    p = argparse.ArgumentParser(description="Итоговая визуализация консенсус-анализа")
    p.add_argument("-r", "--results-dir", default="results", help="папка с top100_*.txt и consensus_top100.tsv")
    p.add_argument("-o", "--output", default="results/consensus_overview.png")
    p.add_argument("-n", "--ntop", type=int, default=20, help="сколько генов показать на панелях C/D")
    return p.parse_args()


def load_top(path):
    genes = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if parts:
                genes.append(parts[0])
    return genes


def get_symbols(ids):
    """Entrez -> символ гена через mygene.info (если не вышло — вернём сами id)."""
    try:
        req = urllib.request.Request(
            "https://mygene.info/v3/gene",
            data=urllib.parse.urlencode({"ids": ",".join(ids), "fields": "symbol"}).encode(),
            method="POST")
        res = json.load(urllib.request.urlopen(req, timeout=60))
        m = {g["_id"]: g.get("symbol", g["_id"]) for g in res if "_id" in g}
        return {i: m.get(i, i) for i in ids}
    except Exception as e:
        print("  (не удалось получить символы генов:", e, ")")
        return {i: i for i in ids}


def main():
    args = parse_args()
    R = args.results_dir

    # ---- собираем все 15 списков ----
    lists = {}
    for d in DIMS:
        for m in MODELS:
            path = os.path.join(R, f"top100_{m}_{d}d.txt")
            if os.path.exists(path):
                lists[(m, d)] = set(load_top(path))
    print(f"Загружено списков: {len(lists)}")

    # ---- A: сходимость алгоритмов по размерности ----
    agree2, agree3 = [], []
    for d in DIMS:
        c = Counter()
        for m in MODELS:
            c.update(lists.get((m, d), set()))
        agree2.append(sum(1 for v in c.values() if v >= 2))
        agree3.append(sum(1 for v in c.values() if v == 3))

    # ---- B: устойчивость алгоритма по размерностям ----
    stability = []
    for m in MODELS:
        c = Counter()
        for d in DIMS:
            c.update(lists.get((m, d), set()))
        stability.append(sum(1 for v in c.values() if v == len(DIMS)))

    # ---- C/D: консенсус ----
    cons_path = os.path.join(R, "consensus_top100.tsv")
    rows = []
    with open(cons_path) as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            rows.append((p[0], int(p[1])))
    top = rows[: args.ntop]
    sym = get_symbols([g for g, _ in top])

    # ================= ФИГУРА =================
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # --- A ---
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(DIMS))
    ax.plot(x, agree2, "o-", lw=2.5, ms=9, color="#4C72B0", label="в ≥2 алгоритмах из 3")
    ax.plot(x, agree3, "s--", lw=2, ms=8, color="#C44E52", label="во всех 3 алгоритмах")
    for i, v in enumerate(agree2):
        ax.annotate(str(v), (x[i], v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10, color="#4C72B0", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{d}D" for d in DIMS])
    ax.set_xlabel("Размерность эмбеддинга", fontsize=11)
    ax.set_ylabel("Число совпавших генов", fontsize=11)
    ax.set_title("A. Чем больше измерений — тем сильнее\nсходятся разные алгоритмы", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # --- B ---
    ax = fig.add_subplot(gs[0, 1])
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(range(len(MODELS)), stability, color=colors, alpha=0.85)
    for b, v in zip(bars, stability):
        ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_RU[m] for m in MODELS], fontsize=10)
    ax.set_ylabel(f"Генов, стабильных во всех {len(DIMS)} размерностях", fontsize=11)
    ax.set_title("B. Устойчивость алгоритма к смене\nразмерности эмбеддинга", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # --- C ---
    ax = fig.add_subplot(gs[1, 0])
    labels = [sym[g] for g, _ in top][::-1]
    vals = [n for _, n in top][::-1]
    ax.barh(range(len(vals)), vals, color="#8172B3", alpha=0.85)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"В скольки из {len(lists)} списков встретился", fontsize=11)
    ax.set_title(f"C. Топ-{args.ntop} консенсусных генов\n(устойчивы к алгоритму и размерности)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    # --- D ---
    ax = fig.add_subplot(gs[1, 1])
    cols = [(m, d) for d in DIMS for m in MODELS]
    mat = np.array([[1 if g in lists.get(c, set()) else 0 for c in cols] for g, _ in top])
    ax.imshow(mat, cmap="Purples", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([sym[g] for g, _ in top], fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{m[:4]}·{d}" for m, d in cols], rotation=90, fontsize=7)
    for i in range(len(top)):
        for j in range(len(cols)):
            if mat[i, j]:
                ax.text(j, i, "•", ha="center", va="center", fontsize=7, color="white")
    ax.set_title("D. В каких именно списках всплыл ген\n(алгоритм · размерность)", fontsize=12, fontweight="bold")

    fig.suptitle("Консенсус-анализ: поиск генов-кандидатов ишемического инсульта\n"
                 f"3 алгоритма × {len(DIMS)} размерностей эмбеддинга = {len(lists)} независимых топ-100",
                 fontsize=14, fontweight="bold", y=0.985)
    plt.savefig(args.output, dpi=170, bbox_inches="tight")
    print(f"Картинка сохранена -> {args.output}")


if __name__ == "__main__":
    main()
