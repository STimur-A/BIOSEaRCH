#!/bin/bash
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

source ~/.venvs/network/bin/activate
cd ~/work/ML/embed
for d in 16 32 64 128; do
  for m in logreg svm tree; do
    echo "### START ${m} ${d}d $(date +%H:%M:%S)"
    python3 predict_top100.py -e data/emb${d}.csv -g /data/genes/human-orthologs-02.txt --model $m -o results/top100_${m}_${d}d.txt 2>&1 | grep -E 'сохранён|ОШИБКА'
  done
done
echo '### ALL DONE'
