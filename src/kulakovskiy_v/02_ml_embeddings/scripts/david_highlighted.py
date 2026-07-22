#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
EN: DAVID enrichment barplot with biologically meaningful GO terms highlighted and technical localisation terms greyed out.

david_highlighted.py
===========================================================================
Версия DAVID-барплота для слайда защиты: биологически значимые термины
(апоптоз, воспаление, окислительный стресс, убиквитин, аутофагия) выделены
насыщенным цветом, техническая локализация (cytoplasm/nucleus/...) — блёклым
серым, чтобы взгляд зрителя сразу шёл на содержательную часть.

ЗАПУСК:
  python3 david_highlighted.py -i МЕТРИКИ/david_svm32_enrichment_full.csv -o МЕТРИКИ/david_svm32_barplot_highlighted.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# термины, которые несут содержательную биологию для темы ишемии
KEY_TERMS = [
    "apoptotic process",
    "cellular response to oxidative stress",
    "positive regulation of canonical NF-kappaB signal",
    "ubiquitin-dependent protein catabolic process",
    "protein K63-linked ubiquitination",
    "ubiquitin protein ligase binding",
    "Autophagy - animal",
]

p = argparse.ArgumentParser()
p.add_argument("-i", "--input", default="МЕТРИКИ/david_svm32_enrichment_full.csv")
p.add_argument("-o", "--output", default="МЕТРИКИ/david_svm32_barplot_highlighted.png")
args = p.parse_args()

t = pd.read_csv(args.input)
top = t.sort_values("P-Value").head(20).iloc[::-1].reset_index(drop=True)
top["logp"] = -np.log10(top["P-Value"])
top["is_key"] = top["Term"].apply(lambda x: any(k.lower() in x.lower() for k in KEY_TERMS))

KEY_COLOR = "#D6604D"      # насыщенный терракотовый — содержательная биология
OTHER_COLOR = "#C9C9C9"    # блёклый серый — техническая локализация/общие термины

colors = [KEY_COLOR if k else OTHER_COLOR for k in top["is_key"]]

fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.barh(range(len(top)), top["logp"], color=colors, alpha=0.95,
               edgecolor=["black" if k else "none" for k in top["is_key"]], linewidth=1.2)

labels = [f"{r.Term[:48]}" for r in top.itertuples()]
ax.set_yticks(range(len(top)))
ax.set_yticklabels(labels, fontsize=9)
# выделяем жирным подписи ключевых терминов
for tick, key in zip(ax.get_yticklabels(), top["is_key"]):
    if key:
        tick.set_fontweight("bold")
        tick.set_color("#B03A2E")

ax.set_xlabel("-log10(P-value)", fontsize=11)
ax.set_title("DAVID: топ-20 обогащённых терминов\nSVM-модель, 32D эмбеддинг, топ-100 предсказанных генов",
             fontsize=12)

for i, r in enumerate(top.itertuples()):
    ax.text(top["logp"].iloc[i] + 0.3, i, f"n={int(r.Count)}", va="center", fontsize=8)

handles = [plt.Rectangle((0, 0), 1, 1, color=KEY_COLOR, ec="black"),
          plt.Rectangle((0, 0), 1, 1, color=OTHER_COLOR)]
ax.legend(handles, ["Биологически значимые\n(апоптоз, стресс, воспаление, убиквитин)",
                    "Локализация / общие термины\n(cytoplasm, protein binding...)"],
         loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(args.output, dpi=170, bbox_inches="tight")
print(f"Сохранено -> {args.output}")
