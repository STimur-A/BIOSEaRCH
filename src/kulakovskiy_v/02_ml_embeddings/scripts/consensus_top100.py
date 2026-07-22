#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
EN: Consensus top-100: rank genes by how many independent top-100 lists they appear in (then by mean probability).

consensus_top100.py
===========================================================================
Итоговый КОНСЕНСУСНЫЙ топ-100 генов по всем спискам сразу.

Идея: чем в большем числе независимых топ-100 ген встретился, тем он
надёжнее. Ген, попавший в топ и у логрегрессии, и у SVM, и у дерева, да ещё
на разных размерностях эмбеддинга — гораздо убедительнее, чем ген, всплывший
в одном списке.

Ранжирование: сначала по числу списков (n_lists), при равенстве — по средней
вероятности (mean_prob).

ЗАПУСК (все 15 файлов сразу):
  python3 consensus_top100.py results/top100_*.txt -o results/consensus_top100.tsv

На вход — файлы формата predict_top100.py: gene_id <TAB> probability.
"""

import argparse
import os
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Консенсусный топ-100 по всем спискам")
    p.add_argument("files", nargs="+", help="файлы топ-100 (gene_id, probability)")
    p.add_argument("-o", "--output", required=True, help="куда сохранить консенсусную таблицу (TSV)")
    p.add_argument("-n", "--top", type=int, default=100, help="сколько строк оставить (по умолчанию 100)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    hits = defaultdict(list)        # gene_id -> [вероятности]
    sources = defaultdict(list)     # gene_id -> [имена списков]

    for path in args.files:
        tag = os.path.basename(path).replace("top100_", "").replace(".txt", "")
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                gene = parts[0]
                prob = float(parts[1]) if len(parts) > 1 else float("nan")
                hits[gene].append(prob)
                sources[gene].append(tag)

    n_files = len(args.files)
    print(f"=== Консенсус по {n_files} спискам ===")
    print(f"Уникальных генов, встретившихся хотя бы раз: {len(hits)}")

    rows = []
    for gene, probs in hits.items():
        rows.append((gene, len(probs), sum(probs) / len(probs), ",".join(sorted(sources[gene]))))
    # сортируем: сначала по числу списков, потом по средней вероятности
    rows.sort(key=lambda r: (-r[1], -r[2]))

    with open(args.output, "w") as fh:
        fh.write("gene_id\tn_lists\tmean_prob\tsources\n")
        for r in rows[: args.top]:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]:.4f}\t{r[3]}\n")

    print(f"Консенсусный топ-{args.top} сохранён -> {args.output}")

    # раскладка
    from collections import Counter
    dist = Counter(r[1] for r in rows)
    print("\nСколько генов в скольки списках:")
    for k in sorted(dist, reverse=True):
        print(f"    в {k:>2} из {n_files} списках: {dist[k]} генов")

    print(f"\nТОП-15 консенсуса:")
    print(f"{'ген':>10} {'списков':>8} {'ср.вер.':>9}")
    for r in rows[:15]:
        print(f"{r[0]:>10} {r[1]:>8} {r[2]:>9.4f}")


if __name__ == "__main__":
    main()
