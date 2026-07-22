#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
EN: Compare the three classifiers across all embedding dimensions via 5-fold cross-validation and plot the metrics.

compare_models.py
===========================================================================
Сравнение качества ТРЁХ алгоритмов на ВСЕХ размерностях эмбеддинга.
Считает метрики (5-фолдовая кросс-валидация) и рисует простой график.

ЗАПУСК:
  python3 compare_models.py -d data -g /data/genes/human-orthologs-02.txt -o МЕТРИКИ
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DIMS = [8, 16, 32, 64, 128, 256, 512]
MODELS = {
    "logreg": ("Логистическая регрессия", "#4C72B0"),
    "svm":    ("Опорные векторы (SVM)",   "#DD8452"),
    "tree":   ("Дерево решений",          "#55A868"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Сравнение качества 3 моделей по размерностям")
    p.add_argument("-d", "--data-dir", default="data")
    p.add_argument("-g", "--genes", required=True)
    p.add_argument("-o", "--output-dir", default="МЕТРИКИ")
    return p.parse_args()


def build(name):
    if name == "logreg":
        return Pipeline([("s", StandardScaler()),
                         ("m", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
    if name == "svm":
        return Pipeline([("s", StandardScaler()),
                         ("m", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE))])
    return Pipeline([("m", DecisionTreeClassifier(random_state=RANDOM_STATE))])


def make_labels(X, nodes, seed):
    y = np.full(len(nodes), -1, dtype=int)
    y[np.isin(nodes, list(seed))] = 1
    n_pos = int((y == 1).sum())
    unl = y == -1
    far = np.argsort(cdist(X[y == 1], X[unl]).sum(axis=0))[-n_pos:]
    y[np.isin(nodes, list(nodes[unl][far]))] = 0
    return y


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda f: os.path.join(args.output_dir, f)

    with open(args.genes) as fh:
        seed = {l.strip() for l in fh if l.strip()}

    rows = []
    for dim in DIMS:
        path = os.path.join(args.data_dir, f"emb{dim}.csv")
        if not os.path.exists(path):
            continue
        data = pd.read_csv(path, header=None)
        X, nodes = data.iloc[:, 1:].values, data.iloc[:, 0].astype(str).values
        y = make_labels(X, nodes, seed)
        tr = (y == 0) | (y == 1)
        Xt, yt = X[tr], y[tr]

        for key in MODELS:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            r = cross_validate(build(key), Xt, yt, cv=cv,
                               scoring=["accuracy", "f1", "roc_auc"], n_jobs=1)
            rows.append({
                "dim": dim, "model": key,
                "accuracy": r["test_accuracy"].mean(),
                "f1": r["test_f1"].mean(),
                "f1_std": r["test_f1"].std(),
                "roc_auc": r["test_roc_auc"].mean(),
            })
            print(f"[{dim:>3}D] {key:<7} acc={rows[-1]['accuracy']:.4f}  "
                  f"F1={rows[-1]['f1']:.4f}  AUC={rows[-1]['roc_auc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out("quality_all_models.csv"), index=False, float_format="%.4f")

    # ---------- простой график: 3 метрики x 3 модели ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(DIMS))
    for ax, metric, title in zip(axes,
                                 ["accuracy", "f1", "roc_auc"],
                                 ["Accuracy (доля верных)", "F1-мера", "ROC-AUC"]):
        for key, (label, color) in MODELS.items():
            sub = df[df["model"] == key].sort_values("dim")
            ax.plot(x, sub[metric], "o-", lw=2.5, ms=8, color=color, label=label)
        ax.set_xticks(x); ax.set_xticklabels([f"{d}" for d in DIMS])
        ax.set_xlabel("Размерность эмбеддинга", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.80, 1.005)
    axes[0].set_ylabel("Значение метрики", fontsize=11)
    axes[0].legend(fontsize=9, loc="lower left")

    fig.suptitle("Сравнение трёх алгоритмов по размерностям эмбеддинга\n"
                 "(5-фолдовая кросс-валидация)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out("comparison_metrics.png"), dpi=170, bbox_inches="tight")
    plt.close()

    # ---------- сводка: лучшее ----------
    lines = ["ЛУЧШИЕ РЕЗУЛЬТАТЫ", "=" * 55, ""]
    for key, (label, _) in MODELS.items():
        sub = df[df["model"] == key]
        b = sub.loc[sub["f1"].idxmax()]
        lines.append(f"{label}:")
        lines.append(f"   лучшая размерность: {int(b['dim'])}D   "
                     f"F1={b['f1']:.4f}  acc={b['accuracy']:.4f}  AUC={b['roc_auc']:.4f}")
    best = df.loc[df["f1"].idxmax()]
    lines += ["", "-" * 55,
              f"АБСОЛЮТНЫЙ ЛУЧШИЙ: {MODELS[best['model']][0]} на {int(best['dim'])}D "
              f"(F1={best['f1']:.4f})", "",
              "ВАЖНО: отрицательный класс построен искусственно (гены, максимально",
              "далёкие от seed), поэтому задача лёгкая и метрики близки к 1.",
              "Они показывают качество разделения обучающих классов, а не точность",
              "предсказания новых генов-кандидатов."]
    txt = "\n".join(lines)
    open(out("ЛУЧШЕЕ.txt"), "w").write(txt)
    print("\n" + txt)
    print(f"\nСохранено в папку: {args.output_dir}/")


if __name__ == "__main__":
    main()
