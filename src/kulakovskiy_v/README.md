# BIOSEaRCH

**Candidate-gene discovery for ischemic stroke through PPI-network analysis and
machine learning.**

Russian version: [README.ru.md](README.ru.md)

This repository collects the reproducible part of a research project that looks
for candidate genes associated with multifactorial diseases (worked example:
**ischemic stroke**) by analysing a human protein–protein interaction (PPI)
network with two complementary families of methods:

1. **Direct network analysis** — "guilt by association" candidate ranking, a
   permutation (null-model) significance test, a comparison against the
   published **DIAMOnD** disease-module algorithm, and functional annotation
   through **DAVID**.
2. **Machine learning on node embeddings** — the network is embedded with
   **node2vec**, and classifiers (logistic regression, SVM, decision tree,
   LightGBM) are trained on the embedding to score every gene by how
   "stroke-like" its position in the network is.

The work was carried out for the **"Bolshie Vyzovy 2026"** science programme
(track *"Nature-like & neurotechnologies"*, Sirius Educational Center), with
the scientific support of the **NRC "Kurchatov Institute"**.

Project website: <https://github.com/STimur-A/BIOSEaRCH>

---

## Repository layout

```
BIOSEaRCH/
├── 01_network_pipeline/     Branch 1 — direct network analysis + DIAMOnD + DAVID
│   ├── README.md            step-by-step guide (English)
│   ├── FINDINGS.md          write-up of the last run (which genes, DAVID results)
│   ├── scripts/             the pipeline, numbered in run order (01…12)
│   └── external/            patched third-party DIAMOnD (not our code)
├── 02_ml_embeddings/        Branch 2 — node2vec embeddings + ML classifiers
│   ├── README.md            step-by-step guide (English)
│   ├── scripts/             embedding, prediction, comparison, plots
│   └── data/emb8.csv        small example embedding (larger ones are regenerated)
├── results/                 key result tables and figures from both branches
│   ├── network_pipeline/
│   └── ml_embeddings/
├── data/README.md           where to obtain the input networks and gene lists
├── requirements.txt         Python dependencies
├── CITATION.cff             how to cite this work
└── LICENSE                  MIT
```

Each branch has its own README with the exact commands. Start there.

---

## Quick start

```bash
# 1. environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. get the input data (see data/README.md), then run a branch:
cd 01_network_pipeline   # or 02_ml_embeddings
less README.md
```

The input PPI networks and gene lists are **not** redistributed here (they
belong to third parties and/or are large) — see [data/README.md](data/README.md)
for how to obtain them. A small example embedding (`02_ml_embeddings/data/emb8.csv`)
is included so the ML branch can be tried without regenerating it.

---

## The two branches at a glance

| | Branch 1 — network | Branch 2 — ML on embeddings |
|---|---|---|
| **Idea** | rank neighbours of seed genes, test significance against random sets | embed the network, learn to tell seed genes from far-away controls |
| **Core method** | guilt-by-association + permutation test | node2vec + logreg / SVM / tree / LightGBM |
| **Baseline / comparison** | DIAMOnD (Ghiassian et al., 2015) | consensus across models and embedding dimensions |
| **Annotation** | DAVID functional enrichment | DAVID on the SVM top-100 |
| **Output** | significant candidate genes, Euler/p-value plots | top-100 per model, consensus list, ROC-AUC, visualisations |

---

## Reproducibility notes

- All random operations are seeded (`seed=42`) so runs are repeatable.
- Heavy intermediate outputs (the 10 000-replicate null model, full embeddings
  for every dimension, large matrices) are **not** committed — they are
  reproduced by running the pipeline. Only compact, meaningful results are kept
  under `results/`.
- The single third-party file (`01_network_pipeline/external/DIAMOnD_with_pvalue.py`)
  is DIAMOnD by Ghiassian, Menche & Barabási with a one-line patch; it keeps its
  original attribution and terms (see the header inside the file).

---

## Author & citation

**Vyacheslav Kulakovskiy** (Кулаковский Вячеслав). See [CITATION.cff](CITATION.cff).

Other project members contribute their own scripts and results, added
separately and credited to their authors.

## License

Code in this repository is released under the [MIT License](LICENSE), except the
third-party DIAMOnD file noted above. Input datasets and any results derived
from them remain subject to the licenses of their original sources
(see [data/README.md](data/README.md)).
