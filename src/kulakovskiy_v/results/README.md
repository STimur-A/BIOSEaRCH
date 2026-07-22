# Results

Curated, compact results from both branches. Heavy intermediate outputs (the
10 000-replicate null model, full embeddings, large matrices) are **not** here —
they are reproduced by running the pipelines.

## `network_pipeline/` (branch 1)

| File | What it is |
|---|---|
| `common_nodes.txt` | seed genes present in the network |
| `candidate-genes_top100.txt` | top-100 candidates (step 2) |
| `candidate_gene_significant.txt` | the 15 significant genes (branch A, step 7) |
| `diamond_candidate-genes.txt` | 100 DIAMOnD module genes with p-values |
| `diamond_vs_mine_pvalue_comparison.png`, `diamond_vs_mine_euler.png` | branch A vs DIAMOnD comparison |
| `david_enrichment_100genes.csv`, `david_enrichment_15genes.csv`, `david_enrichment_barplot.png` | DAVID functional annotation |
| `centrality.csv`, `clusters.csv`, `gene_statistics.csv`, `network.graphml` | co-occurrence network analysis (step 8) |
| `degree_distribution*.png`, `heatmap.png`, `histogram.png`, `network.png`, `top_genes.png` | plots |

See [../01_network_pipeline/FINDINGS.md](../01_network_pipeline/FINDINGS.md) for
the interpretation.

## `ml_embeddings/` (branch 2)

| Folder | What it is |
|---|---|
| `top100/` | top-100 candidate lists per model (logreg/svm/tree) × dimension (8…512) |
| `intersections/` | strict intersections (by model, by dimension, all models at 8d) |
| `figures/` | t-SNE, LightGBM/SVM/LogReg visualisations, ROC-AUC, consensus overview, DAVID barplots, defense slide |
| `tables/` | `genes_scored.csv`, consensus top-100, per-model quality metrics, DAVID enrichment |

`ЛУЧШЕЕ.txt` = a short "best results" note kept from the working run.
