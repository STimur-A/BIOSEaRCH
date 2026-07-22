#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
EN: Evaluate SVM quality across embedding dimensions (5-fold stratified CV + held-out 20% test).

evaluate_svm.py
===========================================================================
Оценка КАЧЕСТВА метода опорных векторов (SVM) на всех размерностях эмбеддинга.

Для каждой размерности:
  1. размечает классы (1 = seed-гены ишемии, 0 = максимально далёкие гены);
  2. гоняет 5-фолдовую стратифицированную кросс-валидацию SVM (rbf);
  3. считает accuracy, precision, recall, F1, ROC-AUC (среднее ± стандартное
     отклонение по фолдам);
  4. дополнительно — отложенная тестовая выборка 20% для честной проверки.

ЗАПУСК:
  python3 evaluate_svm.py -d data -g /data/genes/human-orthologs-02.txt -o results/svm_quality.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

RANDOM_STATE = 42
DIMS = [8, 16, 32, 64, 128, 256, 512]


def parse_args():
    p = argparse.ArgumentParser(description="Качество SVM по всем размерностям")
    p.add_argument("-d", "--data-dir", default="data", help="папка с emb{N}.csv")
    p.add_argument("-g", "--genes", required=True, help="seed-гены (положительный класс)")
    p.add_argument("-o", "--output", default="results/svm_quality.csv")
    return p.parse_args()


def make_labels(X, nodes, seed_genes):
    """1 = seed, 0 = самые далёкие от seed (столько же), -1 = остальные."""
    y = np.full(len(nodes), -1, dtype=int)
    y[np.isin(nodes, list(seed_genes))] = 1
    n_pos = int((y == 1).sum())
    unl = y == -1
    far = np.argsort(cdist(X[y == 1], X[unl]).sum(axis=0))[-n_pos:]
    y[np.isin(nodes, list(nodes[unl][far]))] = 0
    return y, n_pos


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.genes) as fh:
        seed = {l.strip() for l in fh if l.strip()}

    rows = []
    for dim in DIMS:
        path = os.path.join(args.data_dir, f"emb{dim}.csv")
        if not os.path.exists(path):
            print(f"[{dim}D] файл {path} не найден — пропускаю")
            continue

        data = pd.read_csv(path, header=None)
        X = data.iloc[:, 1:].values
        nodes = data.iloc[:, 0].astype(str).values
        y, n_pos = make_labels(X, nodes, seed)

        train = (y == 0) | (y == 1)
        Xt, yt = X[train], y[train]

        # SVM всегда со шкалированием -> кладём в Pipeline, чтобы шкалирование
        # считалось внутри каждого фолда (иначе утечка данных)
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                                     probability=True, random_state=RANDOM_STATE))])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        cvres = cross_validate(pipe, Xt, yt, cv=cv, scoring=scoring, n_jobs=1)

        # + отложенная выборка 20%
        Xtr, Xte, ytr, yte = train_test_split(Xt, yt, test_size=0.2,
                                              random_state=RANDOM_STATE, stratify=yt)
        pipe.fit(Xtr, ytr)
        yp = pipe.predict(Xte)
        yprob = pipe.predict_proba(Xte)[:, 1]

        row = {
            "dim": dim,
            "n_train": len(yt),
            "cv_accuracy": cvres["test_accuracy"].mean(),
            "cv_accuracy_std": cvres["test_accuracy"].std(),
            "cv_precision": cvres["test_precision"].mean(),
            "cv_recall": cvres["test_recall"].mean(),
            "cv_f1": cvres["test_f1"].mean(),
            "cv_f1_std": cvres["test_f1"].std(),
            "cv_roc_auc": cvres["test_roc_auc"].mean(),
            "holdout_accuracy": accuracy_score(yte, yp),
            "holdout_precision": precision_score(yte, yp),
            "holdout_recall": recall_score(yte, yp),
            "holdout_f1": f1_score(yte, yp),
            "holdout_roc_auc": roc_auc_score(yte, yprob),
        }
        rows.append(row)

        print(f"[{dim:>3}D] обучение на {len(yt)} генах | "
              f"CV: acc={row['cv_accuracy']:.4f}±{row['cv_accuracy_std']:.4f}  "
              f"F1={row['cv_f1']:.4f}  AUC={row['cv_roc_auc']:.4f} | "
              f"holdout: F1={row['holdout_f1']:.4f}  AUC={row['holdout_roc_auc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False, float_format="%.4f")
    print(f"\nТаблица качества сохранена -> {args.output}")

    print("\n================ ИТОГ: качество SVM по размерностям ================")
    print(f"{'разм.':>6} {'CV acc':>9} {'CV F1':>9} {'CV AUC':>9} {'hold F1':>9} {'hold AUC':>9}")
    for _, r in df.iterrows():
        print(f"{int(r['dim']):>5}D {r['cv_accuracy']:>9.4f} {r['cv_f1']:>9.4f} "
              f"{r['cv_roc_auc']:>9.4f} {r['holdout_f1']:>9.4f} {r['holdout_roc_auc']:>9.4f}")

    best = df.loc[df["cv_f1"].idxmax()]
    print(f"\nЛучшая размерность по CV F1: {int(best['dim'])}D (F1 = {best['cv_f1']:.4f})")


if __name__ == "__main__":
    main()
