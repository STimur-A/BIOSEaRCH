# Branch 1 — Network pipeline

Russian version: [README.ru.md](README.ru.md) · Results write-up: [FINDINGS.md](FINDINGS.md)

Candidate-gene prediction by **guilt-by-association** on a PPI network, with a
**permutation significance test** (random gene sets), a comparison against the
**DIAMOnD** disease-module algorithm, and functional annotation through
**DAVID**.

All commands below are run from **this folder** (`01_network_pipeline/`). Output
files (`out*/`, `analysis/`, `pics/`) are working artefacts — they are
git-ignored and reproduced by running the steps. Input networks and gene lists
come from the shared data folders — see [../data/README.md](../data/README.md).

## Environment

```bash
python3 -m venv ~/.venvs/network
source ~/.venvs/network/bin/activate
pip install -r ../requirements.txt

# step 4 runs the null model in parallel:
sudo apt install parallel        # GNU parallel
```

## Pipeline overview

Three ways to score/confirm candidates, all starting from a shared null model
(steps 1–4):

```
1 explore_network ──> common_nodes.txt (seed genes)
                            │
2 predict_candidates ───────┤──> out1-1/candidate-genes.txt   (real result)
                            │
3 generate_random_sets ─────┤──> out2-1/rg-0.txt … rg-9999.txt (10000 random seed sets)
                            │
4 run_null_model (parallel)─┴──> out3-1/candidate-genes-0.txt … -9999.txt
        │                                 │                          │
        ▼                                 ▼                          ▼
  Branch A: frequency test      Branch B: hubs / pairwise     Branch C: DIAMOnD
  (steps 5-7)                   (steps 8-9)                    (its own algorithm)
        └─────────────── 10, 11: compare A vs C ──────────────┘
                              │
                        12: DAVID (annotation)
```

## Steps

### 1 — Explore the network and seed genes
```bash
python3 scripts/01_explore_network.py \
    -n /data/networks/HS-PI.clean.tsv \
    -g /data/genes/human-orthologs-02.txt \
    -o common_nodes.txt
```
Prints node/edge counts, intersects the gene list with the network (how many
seed genes are actually present), writes them to `common_nodes.txt`, and saves
degree-distribution plots to `pics/`.

### 2 — Predict candidates
```bash
python3 scripts/02_predict_candidates.py \
    -n /data/networks/HS-PI.clean.tsv \
    -g /data/genes/human-orthologs-02.txt \
    -o out1-1/candidate-genes.txt
```
For every neighbour of the seed genes, counts how many distinct seed genes it
connects to. The top 100 (`-t`) are the candidates. The script filters seed
genes absent from the network itself, so passing the raw gene list is fine.

### 3 — Random gene sets (null model)
```bash
python3 scripts/03_generate_random_sets.py \
    -n /data/networks/HS-PI.clean.tsv \
    -g 865 \
    -o out2-1/rg \
    -r 10000
```
`-g 865` = seed-set size (must match the real number of seed genes; check with
`wc -l common_nodes.txt`). Generates 10 000 random sets of that size.

### 4 — Run step 2 on all random sets
```bash
./scripts/04_run_null_model.sh \
    /data/networks/HS-PI.clean.tsv \
    out2-1/rg \
    out3-1 \
    10000
```
Runs step 2 (via GNU `parallel`) on each random set → `out3-1/candidate-genes-{0..9999}.txt`.
This is the null model: "which genes would look like candidates purely by
chance".

> Resolution is bounded by the number of replicates: with 10 000 replicates the
> smallest resolvable p-value is `1/10000 = 0.0001`. A gene seen 0 times means
> "p < 0.0001", not "p = 0". For p-values orders of magnitude smaller, a
> permutation approach is unsuitable — use an analytic test (as DIAMOnD does).

---

**Branch A — frequency significance test (steps 5–7)**

### 5 — Count real candidates in the null model
```bash
python3 scripts/05_count_occurrences.py \
    -i out3-1 -c out1-1/candidate-genes.txt -o out4-1/candidate_gene.txt
```

### 6 — Frequency = occurrences / replicates
```bash
python3 scripts/06_add_frequency.py \
    -i out4-1/candidate_gene.txt -o out5-1/candidate_gene.txt -r 10000
```

### 7 — Filter significant genes
```bash
python3 scripts/07_filter_significant.py \
    -i out5-1/candidate_gene.txt \
    -o out5-1/candidate_gene_significant.txt -t 0.05
```
Keeps genes with frequency `<= 0.05` (rarely arise by chance), sorted by number
of real connections. On the current data: **15 significant genes**.

---

**Branch B — hubs and correlations (steps 8–9)**

### 8 — Gene co-occurrence network
```bash
python3 scripts/08_build_gene_network.py -i out3 -o analysis -m 20 -c 0.85
```
Builds a gene × replicate binary matrix, occurrence statistics, Spearman
correlations, a correlation network, centralities (degree/betweenness/
closeness/eigenvector/PageRank) and Louvain communities. Saves `matrix.csv`,
`gene_statistics.csv`, `correlations.csv`, `centrality.csv`, `clusters.csv`,
`network.graphml`, and plots into `analysis/`.

### 9 — Pairwise significance (hypergeometric)
```bash
python3 scripts/09_pairwise_significance.py \
    -i out3-1 -o analysis/gene_summary.csv -m 20 -a 0.05
```
For each pair of frequently-occurring genes (`>= -m` replicates), tests whether
they co-occur more than expected by chance (hypergeometric test). Significant
pairs are aggregated per gene into a hub score (`n-edges`) and best p-value.

Steps 8 and 9 answer different questions about the same data (cluster/hub
structure vs. direct pairwise significance) — both are kept because both were
used.

---

**Branch C — DIAMOnD (separate algorithm)**

[DIAMOnD](https://github.com/dinaghiassian/DIAMOnD) is a standard disease-module
detection algorithm: instead of simply counting connections to the seed set
(step 2), on each iteration it greedily adds the gene with the smallest
hypergeometric connectivity p-value to the growing module. Runs under Python 2.7.

```bash
python2.7 external/DIAMOnD_with_pvalue.py \
    /data/networks/HS-PI.clean.tsv \
    common_nodes.txt \
    100 \
    out6/candidate-genes-with-pvalue.txt
```

> **Patch note.** The original `DIAMOnD.py` computes a per-step p-value but never
> writes it to the output file. `external/DIAMOnD_with_pvalue.py` is a one-line
> patch that saves it; the ranking is unchanged. The file is third-party code
> and keeps its original attribution (see its header).

### 10 — Compare significance: branch A vs DIAMOnD
```bash
python3 scripts/10_compare_pvalue_diamond.py \
    --mine out5-1/candidate_gene_significant.txt \
    --diamond out6/candidate-genes-with-pvalue.txt \
    -a 0.05 -o analysis/diamond_vs_mine_pvalue_comparison.png
```
Both lists pre-filtered to `p <= 0.05`. Left panel: paired p-value comparison
(log scale) for genes significant in **both** methods; right panel: overlap
summary. On the current data: 15 significant genes (mine), 100 (DIAMOnD), 6
shared.

### 11 — Euler diagram
```bash
python3 scripts/11_euler_diagram.py \
    --mine out5-1/candidate_gene_significant.txt \
    --diamond out6/candidate-genes-with-pvalue.txt \
    -a 0.05 -o analysis/diamond_vs_mine_euler.png
```
Same two sets as area-proportional Euler diagram. Steps 10 and 11 share the
`scripts/_gene_sets.py` helper (loading/filtering both lists).

### 12 — DAVID functional annotation
```bash
./scripts/12_run_david.sh "1386,867,3932,..." out_chart.html
```
Reproduces the DAVID website flow via `curl` + a cookie jar, parses the
enrichment table. DAVID (davidbioinformatics.nih.gov) may be blocked from some
hosts — run from a machine with access (see the comment in the script). Full
interpretation → [FINDINGS.md](FINDINGS.md).

## Outputs

`out1*`…`out6/`, `analysis/`, `pics/` are results, not code, and are
git-ignored. A curated subset of the meaningful ones is kept under
[`../results/network_pipeline/`](../results/network_pipeline/). The full DIAMOnD
clone is not vendored — only the one patched file lives in `external/`.
