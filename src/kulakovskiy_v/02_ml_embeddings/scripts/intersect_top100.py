#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
EN: Strict intersection of several top-100 gene lists, plus a membership breakdown (in how many lists each gene appears).

intersect_top100.py
===========================================================================
Пересечение нескольких топ-100 списков генов.

Два сценария использования:
  А) один алгоритм, разные размерности — насколько устойчив результат к
     размерности эмбеддинга:
       python3 intersect_top100.py results/top100_svm_8d.txt results/top100_svm_16d.txt results/top100_svm_32d.txt -o results/intersect_svm.txt
  Б) одна размерность, разные алгоритмы — гены, на которых сошлись все методы:
       python3 intersect_top100.py results/top100_logreg_8d.txt results/top100_svm_8d.txt results/top100_tree_8d.txt -o results/intersect_8d.txt

На вход — любые txt-файлы, где 1-я колонка = gene_id (формат вывода predict_top100.py).
Печатает строгое пересечение (гены во ВСЕХ списках) и таблицу «в скольки списках встретился».
"""

import argparse
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(description="Пересечение нескольких топ-100 списков генов")
    p.add_argument("files", nargs="+", help="2+ txt-файла с топами (1-я колонка = gene_id)")
    p.add_argument("-o", "--output", required=True, help="куда сохранить строгое пересечение")
    return p.parse_args()


def load_genes(path):
    genes = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                genes.add(line.split()[0])   # 1-я колонка
    return genes


def main():
    args = parse_args()
    if len(args.files) < 2:
        raise SystemExit("Нужно минимум 2 файла для пересечения.")

    sets = {f: load_genes(f) for f in args.files}
    print(f"=== Пересечение {len(sets)} списков ===")
    for f, s in sets.items():
        print(f"  {f}: {len(s)} генов")

    # счётчик: в скольки списках встретился каждый ген
    counter = Counter()
    for s in sets.values():
        counter.update(s)

    n = len(sets)
    strict = sorted(g for g, c in counter.items() if c == n)

    print(f"\nВ ЛЮБОМ списке всего уникальных генов: {len(counter)}")
    print(f"В ОТ 2 списках и более: {sum(1 for c in counter.values() if c >= 2)}")
    print(f"Во ВСЕХ {n} списках (строгое пересечение): {len(strict)}")

    with open(args.output, "w") as fh:
        for g in strict:
            fh.write(g + "\n")
    print(f"Строгое пересечение сохранено -> {args.output}")

    # заодно раскладка по числу совпадений
    print("\nСколько генов встретилось в N списках:")
    dist = Counter(counter.values())
    for k in sorted(dist, reverse=True):
        print(f"    в {k} из {n} списках: {dist[k]} генов")


if __name__ == "__main__":
    main()
