#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
EN: Predict stroke candidate genes from PPI node embeddings and render three ML visualisations (LightGBM heatmap, SVM zones, LogReg boundary).

02_predict_and_visualize.py
===========================================================================
Предсказание генов-кандидатов инсульта по эмбеддингам узлов PPI-сети
и ТРИ визуализации разными ML-подходами.

ЧТО ДЕЛАЕТ (по шагам):
  1. Читает эмбеддинг всех генов сети (data/emb8.csv: gene_id + 8 признаков).
  2. Размечает классы (PU-разметка):
        1  (положительный) = seed-гены инсульта из файла -g
        0  (отрицательный) = столько же генов, максимально ДАЛЁКИХ от seed
                             в пространстве эмбеддинга (надёжный "не-инсульт")
       -1  (кандидаты)     = все остальные — их модель и будет оценивать
  3. Обучает 3 модели на классах {0,1} и рисует по визуализации на каждую:
        (A) LightGBM   -> тепловая карта вероятности поверх t-SNE
        (B) SVM (rbf)  -> нелинейные разделяющие зоны
        (C) LogReg     -> линейная разделяющая граница
  4. Сохраняет таблицу всех генов, отсортированную по вероятности "инсультный".

ЗАПУСК:
  python3 02_predict_and_visualize.py \
      -e data/emb8.csv \
      -g /data/genes/human-orthologs-02.txt \
      -o results

Формат эмбеддинга: CSV, БЕЗ заголовка, 1-я колонка = gene_id, дальше признаки.
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

RANDOM_STATE = 42


def step(msg):
    """Печать заметного заголовка шага, чтобы в логе было понятно, где мы."""
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(description="Предсказание генов-кандидатов + 3 визуализации")
    p.add_argument("-e", "--embedding", required=True,
                   help="CSV эмбеддинга БЕЗ заголовка (1-я колонка = gene_id)")
    p.add_argument("-g", "--genes", required=True,
                   help="TXT со seed-генами (по одному id в строке) — положительный класс")
    p.add_argument("-o", "--output-dir", default="results",
                   help="папка для всех результатов (создаётся автоматически)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)

    # ------------------------------------------------------------------
    step("ШАГ 1/6. Загрузка эмбеддинга и seed-генов")

    data = pd.read_csv(args.embedding, header=None)
    embeddings = data.iloc[:, 1:].values
    nodes = data.iloc[:, 0].astype(str).values
    print(f"  Эмбеддинг: {data.shape[0]} генов x {embeddings.shape[1]} признаков")

    with open(args.genes) as fh:
        seed_genes = {line.strip() for line in fh if line.strip()}
    print(f"  Seed-генов в файле: {len(seed_genes)}")

    # ------------------------------------------------------------------
    step("ШАГ 2/6. Разметка классов (1 = seed, 0 = дальние, -1 = кандидаты)")

    outcomes = np.full(len(nodes), -1, dtype=int)
    for i, node in enumerate(nodes):
        if node in seed_genes:
            outcomes[i] = 1

    n_pos = int((outcomes == 1).sum())
    print(f"  Положительный класс (seed, найдено в эмбеддинге): {n_pos}")
    if n_pos == 0:
        raise SystemExit("ОШИБКА: ни один seed-ген не найден в эмбеддинге. "
                         "Проверь, что id совпадают (Entrez vs символы).")

    # отрицательный класс = n_pos генов, максимально далёких от seed
    pos_emb = embeddings[outcomes == 1]
    unl_mask = outcomes == -1
    unl_emb = embeddings[unl_mask]
    unl_nodes = nodes[unl_mask]

    print(f"  Считаю расстояния от {n_pos} seed-генов до {len(unl_nodes)} остальных...")
    total_dist = cdist(pos_emb, unl_emb).sum(axis=0)
    far_idx = np.argsort(total_dist)[-n_pos:]          # самые далёкие
    negative_genes = set(unl_nodes[far_idx])
    for i, node in enumerate(nodes):
        if node in negative_genes:
            outcomes[i] = 0

    counts = Counter(outcomes)
    print(f"  Итог разметки: 1 (seed) = {counts[1]}, "
          f"0 (дальние) = {counts[0]}, -1 (кандидаты) = {counts[-1]}")

    # ------------------------------------------------------------------
    step("ШАГ 3/6. Обучение LightGBM и оценка ВСЕХ генов сети")

    train_mask = (outcomes == 1) | (outcomes == 0)
    X_train, y_train = embeddings[train_mask], outcomes[train_mask]
    print(f"  Обучающая выборка: {len(y_train)} генов ({counts[1]} поз. + {counts[0]} отр.)")

    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05,
                          random_state=RANDOM_STATE, verbose=-1)
    lgbm.fit(X_train, y_train)
    proba_all = lgbm.predict_proba(embeddings)[:, 1]
    print("  LightGBM обучен, вероятности посчитаны для всех генов.")

    # таблица результатов
    result = pd.DataFrame({"gene_id": nodes,
                           "training_class": outcomes,
                           "disease_probability": proba_all})
    result = result.sort_values("disease_probability", ascending=False)
    scored_csv = out("genes_scored.csv")
    result.to_csv(scored_csv, index=False)
    print(f"  Таблица генов по убыванию вероятности -> {scored_csv}")
    print("  ТОП-10 генов-кандидатов (по мнению LightGBM):")
    top = result[result["training_class"] == -1].head(10)
    for _, r in top.iterrows():
        print(f"      {r['gene_id']:>10}   p = {r['disease_probability']:.3f}")

    # ------------------------------------------------------------------
    step("ШАГ 4/6. Снижение размерности t-SNE (для всех 3 картинок)")
    print("  Это самый долгий шаг (несколько минут для большой сети)...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, max_iter=1000)
    xy = tsne.fit_transform(embeddings)
    print("  t-SNE готов (2D координаты получены).")

    # общая сетка для «зон» SVM и LogReg
    x_min, x_max = xy[:, 0].min() - 5, xy[:, 0].max() + 5
    y_min, y_max = xy[:, 1].min() - 5, xy[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    xy_train, y_train_2d = xy[train_mask], outcomes[train_mask]

    def draw_train_points():
        """Общие точки для всех графиков: seed (чёрные) и дальние (зелёные)."""
        plt.scatter(xy[outcomes == 1, 0], xy[outcomes == 1, 1], color="black",
                    marker="D", s=32, label="seed-гены инсульта (1)",
                    edgecolors="white", linewidths=0.6, zorder=3)
        plt.scatter(xy[outcomes == 0, 0], xy[outcomes == 0, 1], color="limegreen",
                    marker="^", s=36, label="дальний контроль (0)",
                    edgecolors="black", linewidths=0.6, zorder=3)

    # ------------------------------------------------------------------
    step("ШАГ 5/6. Три визуализации — три подхода")

    # (A) LightGBM heatmap
    plt.figure(figsize=(12, 9))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=proba_all, cmap="coolwarm",
                     alpha=0.6, s=15, edgecolors="none")
    plt.colorbar(sc).set_label("Вероятность (LightGBM)", fontsize=11)
    draw_train_points()
    plt.title("Подход A — LightGBM: тепловая карта вероятности «ген инсульта»", fontsize=13)
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3); plt.tight_layout()
    plt.savefig(out("viz1_lightgbm_heatmap.png"), dpi=200, bbox_inches="tight"); plt.close()
    print(f"  (A) LightGBM     -> {out('viz1_lightgbm_heatmap.png')}")

    # (B) SVM zones
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)
    svm.fit(xy_train, y_train_2d)
    Z = svm.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.figure(figsize=(12, 9))
    cf = plt.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.35)
    plt.colorbar(cf).set_label("Уверенность SVM", fontsize=11)
    plt.contour(xx, yy, Z, levels=[0.5], colors="darkblue", linewidths=2, linestyles="--")
    plt.scatter(xy[outcomes == -1, 0], xy[outcomes == -1, 1], color="grey",
                alpha=0.2, s=10, label="кандидаты (-1)")
    draw_train_points()
    plt.title("Подход B — SVM (rbf): нелинейные разделяющие зоны", fontsize=13)
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.2); plt.tight_layout()
    plt.savefig(out("viz2_svm_zones.png"), dpi=200, bbox_inches="tight"); plt.close()
    print(f"  (B) SVM          -> {out('viz2_svm_zones.png')}")

    # (C) LogReg zones
    lr = LogisticRegression(random_state=RANDOM_STATE)
    lr.fit(xy_train, y_train_2d)
    Z = lr.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.figure(figsize=(12, 9))
    cf = plt.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.35)
    plt.colorbar(cf).set_label("Вероятность (LogReg)", fontsize=11)
    plt.contour(xx, yy, Z, levels=[0.5], colors="darkgreen", linewidths=2, linestyles="-.")
    plt.scatter(xy[outcomes == -1, 0], xy[outcomes == -1, 1], color="grey",
                alpha=0.2, s=10, label="кандидаты (-1)")
    draw_train_points()
    plt.title("Подход C — Logistic Regression: линейная граница", fontsize=13)
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.2); plt.tight_layout()
    plt.savefig(out("viz3_logreg_zones.png"), dpi=200, bbox_inches="tight"); plt.close()
    print(f"  (C) LogReg       -> {out('viz3_logreg_zones.png')}")

    # ------------------------------------------------------------------
    step("ШАГ 6/6. Готово")
    print("  Все результаты в папке:", args.output_dir)
    print("    genes_scored.csv          — все гены по убыванию вероятности")
    print("    viz1_lightgbm_heatmap.png — подход A")
    print("    viz2_svm_zones.png        — подход B")
    print("    viz3_logreg_zones.png     — подход C")


if __name__ == "__main__":
    main()
