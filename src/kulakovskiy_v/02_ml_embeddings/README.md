# Branch 2 — ML on PPI-network embeddings

Russian version: [README.ru.md](README.ru.md)

Pipeline: **network → node embedding (node2vec) → predict stroke candidate genes
+ visualisations + quality evaluation.** A continuation of branch 1 — there the
network was analysed directly, here machine learning works on vector
representations of genes.

Run everything from **this folder** (`02_ml_embeddings/`) with the venv active:
```bash
source ~/.venvs/network/bin/activate   # or your own venv with ../requirements.txt
```

## Folder layout

```
02_ml_embeddings/
├── scripts/
│   ├── 01_create_embedding.py        step 1: network -> node2vec embedding
│   ├── 02_predict_and_visualize.py   predict candidates + 3 ML visualisations
│   ├── 03_plot_embedding.py          (utility) t-SNE view of the embedding (2D/3D)
│   ├── predict_top100.py             top-100 candidates by a single classifier
│   ├── intersect_top100.py           strict intersection of top-100 lists
│   ├── consensus_top100.py           consensus top-100 across all lists
│   ├── compare_models.py             3 models x all dims, 5-fold CV metrics
│   ├── evaluate_svm.py               SVM quality across dims (CV + held-out test)
│   ├── plot_roc_auc.py               ROC-AUC vs dimensionality figure
│   ├── plot_consensus.py             4-panel consensus overview figure
│   ├── david_highlighted.py          DAVID barplot (key terms highlighted)
│   ├── slide_figure.py               defense-slide figure
│   ├── mask-genes-01.py              (utility) random gene masking / robustness
│   ├── run_all.sh, run_512.sh        batch runners over dimensions 16…512
├── data/
│   └── emb8.csv                      small example embedding (8-dim), tracked
└── ../results/ml_embeddings/         curated results (top100, intersections, figures, tables)
```

## ⚠️ One embedding file format

**CSV, NO header, comma-separated. Column 1 = gene_id, then the features.**
```
4780,-0.29,-0.24,-0.28,0.25,0.58,-0.75,0.006,0.40
9817,-0.66,-0.12,0.05,0.23,0.19,-0.48,-0.31,0.05
```
All scripts read/write exactly this format. (An earlier mismatch — node2vec
writing tab-separated with a header while the ML scripts expected comma-separated
without one — is why "nothing worked" at one point. It is now unified.)

## Step 1 — Create the embedding
```bash
python3 scripts/01_create_embedding.py -n /data/networks/HS-PI.clean.tsv -d 8 -o data/emb8.csv
```
`-d` = embedding dimensionality. Higher = richer features, slower. Produces
`data/emb16.csv`, `emb32.csv`, … A slow step (~10 min on the full network) —
run it in `tmux`. Only `emb8.csv` is committed; the rest are regenerated
(git-ignored).

## Step 2a — Top-100 candidates by a single classifier
```bash
python3 scripts/predict_top100.py -e data/emb8.csv -g /data/genes/human-orthologs-02.txt --model logreg -o out/top100_logreg_8d.txt
python3 scripts/predict_top100.py -e data/emb8.csv -g /data/genes/human-orthologs-02.txt --model svm    -o out/top100_svm_8d.txt
python3 scripts/predict_top100.py -e data/emb8.csv -g /data/genes/human-orthologs-02.txt --model tree   -o out/top100_tree_8d.txt
```
`logreg` = logistic regression, `svm` = support-vector machine, `tree` =
decision tree. Trained on the **full** embedding. Labelling: **1** = seed genes,
**0** = the same number of genes farthest from the seed in embedding space
(reliable "non-stroke"), **−1** = all remaining candidates; the top-100 are
taken among the candidates by predicted probability.

`run_all.sh` / `run_512.sh` batch this over dimensions 16…512 and all three
models.

### Intersect top-100 lists
```bash
# A) one model, different dimensions (robustness to dimensionality)
python3 scripts/intersect_top100.py out/top100_svm_8d.txt out/top100_svm_16d.txt out/top100_svm_32d.txt -o out/intersect_svm.txt
# B) one dimension, all three models (genes all methods agree on)
python3 scripts/intersect_top100.py out/top100_logreg_8d.txt out/top100_svm_8d.txt out/top100_tree_8d.txt -o out/intersect_8d.txt
```
Prints the strict intersection (genes in **all** lists) plus a "how many lists
each gene appears in" breakdown.

> On 8 dimensions the three algorithms barely overlap (often 0 across all three)
> — 8 dims is too few, models capture different things. Overlap grows at 16/32+.

### Consensus top-100
```bash
python3 scripts/consensus_top100.py out/top100_*.txt -o out/consensus_top100.tsv
```
Ranks genes by how many independent top-100 lists they appear in, then by mean
probability. A gene chosen by logreg, SVM *and* the tree, across several
dimensions, is far more convincing than one that surfaced in a single list.

## Step 2b — Predict + 3 visualisations
```bash
python3 scripts/02_predict_and_visualize.py -e data/emb8.csv -g /data/genes/human-orthologs-02.txt -o out
```
Labels genes (1 seed / 0 far control / −1 candidates), trains three models and
saves one figure each — **LightGBM** probability heatmap, **SVM (rbf)** nonlinear
zones, **LogReg** linear boundary — plus `genes_scored.csv` (all genes by
descending probability). Verbose step-by-step console log + top-10 candidates.

## Quality evaluation
```bash
# 3 models x all dimensions, 5-fold CV
python3 scripts/compare_models.py -d data -g /data/genes/human-orthologs-02.txt -o out/metrics
# SVM in depth (CV + held-out 20% test)
python3 scripts/evaluate_svm.py   -d data -g /data/genes/human-orthologs-02.txt -o out/svm_quality.csv
# ROC-AUC vs dimensionality
python3 scripts/plot_roc_auc.py   -i out/metrics/quality_all_models.csv -o out/roc_auc.png
```

## Figures
```bash
# t-SNE overview (interactive 3D html or static png)
python3 scripts/03_plot_embedding.py -e data/emb8.csv -o out/tsne_3d.html -g /data/genes/human-orthologs-02.txt -d 3
# 4-panel consensus overview
python3 scripts/plot_consensus.py -r out -o out/consensus_overview.png
# DAVID barplot with key terms highlighted
python3 scripts/david_highlighted.py -i out/david_svm32_enrichment_full.csv -o out/david_barplot_highlighted.png
```

## How to read the result
- **Seed genes** should cluster in the high-probability ("red") zone, the **far
  control** in the low ("blue") zone. If so, the model learned a meaningful
  distinction.
- **Candidates** landing in the high-probability zone and near the top of
  `genes_scored.csv` are genes that, by their network position, look like stroke
  genes but were not in the seed list — that is the prediction.

> Note on the seed set: like branch 1, this uses the broad `human-orthologs-02.txt`
> set. See [../01_network_pipeline/FINDINGS.md](../01_network_pipeline/FINDINGS.md)
> for what that implies about interpreting the candidates.
