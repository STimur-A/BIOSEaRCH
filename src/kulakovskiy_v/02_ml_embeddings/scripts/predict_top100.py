#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
EN: Rank the top-100 stroke candidate genes with one classifier (logreg / SVM / decision tree), trained on the full node embedding.

predict_top100.py
===========================================================================
Топ-100 генов-кандидатов инсульта одним из трёх ML-методов.

Обучение идёт на ПОЛНОМ эмбеддинге (все признаки, не t-SNE!).
  1 (положительный) = seed-гены инсульта (-g)
  0 (отрицательный) = столько же генов, максимально далёких от seed
 -1 (кандидаты)     = все остальные; из них и берём топ-100 по вероятности.

ЗАПУСК (один и тот же скрипт, меняется только --model):
  python3 predict_top100.py -e data/emb8.csv -g /data/genes/human-orthologs-02.txt --model logreg -o results/top100_logreg_8d.txt
  python3 predict_top100.py -e data/emb8.csv -g /data/genes/human-orthologs-02.txt --model svm    -o results/top100_svm_8d.txt
  python3 predict_top100.py -e data/emb8.csv -g /data/genes/human-orthologs-02.txt --model tree   -o results/top100_tree_8d.txt

Для 16/32-мерного эмбеддинга — то же самое, меняешь только -e data/emb16.csv и имя -o.

Формат эмбеддинга: CSV без заголовка, 1-я колонка = gene_id, дальше признаки.
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def parse_args():
    p = argparse.ArgumentParser(description="Топ-100 генов-кандидатов одним из трёх методов")
    p.add_argument("-e", "--embedding", required=True, help="CSV эмбеддинга без заголовка (1-я колонка = gene_id)")
    p.add_argument("-g", "--genes", required=True, help="TXT seed-генов (положительный класс)")
    p.add_argument("--model", required=True, choices=["logreg", "svm", "tree"],
                   help="logreg = логистическая регрессия, svm = опорные векторы, tree = дерево решений")
    p.add_argument("-o", "--output", required=True, help="куда сохранить топ-100 (txt: gene_id + вероятность)")
    p.add_argument("-n", "--top", type=int, default=100, help="сколько генов оставить (по умолчанию 100)")
    return p.parse_args()


def build_model(name):
    """Возвращает модель. Для logreg/svm признаки масштабируем, для дерева — нет."""
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE), True
    if name == "tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE), False


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"\n=== Топ-{args.top} методом '{args.model}' на {args.embedding} ===")

    # 1. загрузка
    data = pd.read_csv(args.embedding, header=None)
    X = data.iloc[:, 1:].values
    nodes = data.iloc[:, 0].astype(str).values
    print(f"Эмбеддинг: {len(nodes)} генов x {X.shape[1]} признаков")

    with open(args.genes) as fh:
        seed = {l.strip() for l in fh if l.strip()}

    # 2. разметка 1 / 0 / -1
    y = np.full(len(nodes), -1, dtype=int)
    y[np.isin(nodes, list(seed))] = 1
    n_pos = int((y == 1).sum())
    print(f"Seed-генов найдено в эмбеддинге: {n_pos}")
    if n_pos == 0:
        raise SystemExit("ОШИБКА: seed-гены не найдены в эмбеддинге (проверь id).")

    pos_emb = X[y == 1]
    unl_mask = y == -1
    unl_emb, unl_nodes = X[unl_mask], nodes[unl_mask]
    far_idx = np.argsort(cdist(pos_emb, unl_emb).sum(axis=0))[-n_pos:]
    neg = set(unl_nodes[far_idx])
    y[np.isin(nodes, list(neg))] = 0
    print("Разметка:", dict(Counter(y)), "(1=seed, 0=дальние, -1=кандидаты)")

    # 3. обучение на {0,1}
    train = (y == 0) | (y == 1)
    model, scale = build_model(args.model)
    if scale:
        scaler = StandardScaler().fit(X[train])
        Xtr, Xall = scaler.transform(X[train]), scaler.transform(X)
    else:
        Xtr, Xall = X[train], X
    model.fit(Xtr, y[train])
    proba = model.predict_proba(Xall)[:, 1]
    print(f"Модель обучена на {int(train.sum())} генах, вероятности посчитаны для всех.")

    # 4. топ-N среди кандидатов (label -1)
    res = pd.DataFrame({"gene_id": nodes, "class": y, "probability": proba})
    cand = res[res["class"] == -1].sort_values("probability", ascending=False).head(args.top)
    cand[["gene_id", "probability"]].to_csv(args.output, sep="\t", index=False,
                                            header=False, float_format="%.4f")
    print(f"\nТоп-{args.top} кандидатов сохранён -> {args.output}")
    print("Первая пятёрка:")
    for _, r in cand.head(5).iterrows():
        print(f"    {r['gene_id']:>10}   p = {r['probability']:.4f}")


if __name__ == "__main__":
    main()
