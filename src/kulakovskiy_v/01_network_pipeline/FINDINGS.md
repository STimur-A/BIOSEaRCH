# Findings — what was predicted and what it means

Russian version: [FINDINGS.ru.md](FINDINGS.ru.md)

A summary of the genes predicted by branch 1, what they relate to (functional
annotation via **DAVID**), and the full chain of work.

---

## 1. The work, in brief

The task: **predict candidate genes** functionally close to a given seed set,
using a protein–protein interaction network (`HS-PI.clean.tsv`, ~17 849 genes /
~633 460 edges), and assess how statistically non-random the prediction is.

Three independent blocks:

| Block | Method | Result |
|---|---|---|
| **A. Own pipeline** | guilt-by-association (counting connections to the seed set) + permutation test over 10 000 random sets | 100 candidates → **15 significant** (chance-frequency ≤ 0.05) |
| **B. DIAMOnD** | greedy disease-module growth by hypergeometric p-value (Barabási-lab algorithm) | **100 module genes**, all with p_hyper ≤ 0.05 |
| **C. Comparison + DAVID** | intersection of the two methods + functional annotation | 6 shared genes; strong pathway/process enrichment |

The seed set — `human-orthologs-02.txt` (~890 genes) — is a **broad functional
set** (immunity, signalling, metabolism, RNA processing), **not** specific to a
single disease. So the genes found reflect the topology of the network around
that set, not a specific diagnosis. This is the key caveat (see §4).

---

## 2. Which genes were found

### 6 genes significant in BOTH methods (most robust result)

The core of the finding — genes that passed both the strict permutation test
and the DIAMOnD module:

| Entrez | Symbol | Name | Role |
|---|---|---|---|
| 3932 | **LCK** | LCK proto-oncogene, Src family tyrosine kinase | T-cell tyrosine kinase |
| 6772 | **STAT1** | signal transducer and activator of transcription 1 | transcription factor, JAK-STAT |
| 5335 | **PLCG1** | phospholipase C gamma 1 | signalling, second messengers |
| 867 | **CBL** | Cbl proto-oncogene | E3 ubiquitin ligase, regulator of receptor kinases |
| 1386 | **ATF2** | activating transcription factor 2 | transcription factor |
| 4627 | **MYH9** | myosin heavy chain 9 | non-muscle myosin, cytoskeleton |

### The other 9 of the 15 significant genes

**MAPK1** (ERK2), **TNF**, **IKBKE** (NF-κB-pathway kinase), **CD81**, plus
**CDK9**, **U2AF2**, **NDN**, **NFX1**, **NHLRC2**. A mix: inflammatory/immune
signalling (TNF, IKBKE, MAPK1) + transcription/splicing (CDK9, U2AF2, ATF2, NFX1).

### The 100-gene DIAMOnD module is one functional module

Almost the entire module is **tyrosine-kinase signalling**:

- **Src family & other tyrosine kinases:** SRC, LCK, FYN, YES1, BTK, ITK, TEC,
  ZAP70, CSK, ABL1, JAK1, JAK2, TYK2, PTK2, PTK2B, PTPN11
- **Growth-factor receptors:** EGFR, ERBB2/3/4, KIT, MET, PDGFRB, FLT3, CSF1R, EPOR
- **PI3K cascade:** PIK3CA/CB/CD, PIK3R1/R2, PDPK1, INPP5D
- **JAK-STAT / cytokines:** STAT1, STAT5A/B, SOCS1, IL2/3/5, IL2RA/B, CSF2
- **Adapters:** GRB2, GAB1/2, CBL, CBLB, CRK, CRKL, SOS1, SHC2/3, LAT
- **Actin cytoskeleton (WASP/ARP2/3):** WAS, WASL, WIPF1/2/3, ARPC1A/3/4, RAC1, PAK2
- **Immune receptors:** CD19, CD247, CD3G, FCER1A

The full 100 symbols can be regenerated in one command (not stored as a file):
```bash
curl -s -X POST 'https://mygene.info/v3/gene' \
  --data "ids=$(awk -F'\t' 'NR>1{print $2}' out6/candidate-genes-with-pvalue.txt | tr '\n' ',')&fields=symbol" \
  | python3 -c 'import sys,json; print(", ".join(sorted(g["symbol"] for g in json.load(sys.stdin) if "symbol" in g)))'
```

---

## 3. What the genes relate to — DAVID results

The **100 DIAMOnD-module genes** were run through
[DAVID](https://davidbioinformatics.nih.gov) (Functional Annotation Chart; GO
BP/MF/CC, KEGG, OMIM). Result — **656 terms, 392 significant** (Benjamini < 0.05).
Enormous enrichment.

**Figure:** `../results/network_pipeline/david_enrichment_barplot.png`
**Tables:** `../results/network_pipeline/david_enrichment_100genes.csv`,
`david_enrichment_15genes.csv`

### Top enriched pathways/processes (100 genes)

| P-value | Genes | Term (category) |
|---|---|---|
| 2.9e-32 | 27 | **ErbB signaling pathway** (KEGG) |
| 1.1e-30 | 25 | **protein tyrosine kinase activity** (GO:MF) |
| 7.4e-28 | 27 | **T cell receptor signaling pathway** (KEGG) |
| 6.6e-28 | 23 | Fc epsilon RI signaling pathway (KEGG) |
| 4.9e-24 | 27 | **JAK-STAT signaling pathway** (KEGG) |
| 7.1e-23 | 39 | Pathways in cancer (KEGG) |
| 8.5e-21 | 32 | **PI3K-Akt signaling pathway** (KEGG) |
| 4.5e-20 | 27 | Ras signaling pathway (KEGG) |
| 1.4e-20 | 26 | Focal adhesion (KEGG) |

**Takeaway:** the module is a **canonical tyrosine-kinase signal-transduction
network** — growth-factor receptors (ErbB/EGFR) + immune-receptor signalling
(T-cell / Fc / NK) + JAK-STAT + PI3K-Akt/Ras. These are the densest hubs of the
interactome and naturally surface as candidates next to any large seed set.

### The 15 significant genes — an immune/inflammatory lean

A separate DAVID run on the 15 genes (30 significant terms) gives a narrower,
**immune-inflammatory** profile: Th1/Th2/Th17 differentiation, T-cell receptor,
C-type lectin, Toll-like receptor, NK-cell cytotoxicity; infection pathways
(Hepatitis B/C, COVID-19, HPV, Yersinia); PD-L1/PD-1 checkpoint, AGE-RAGE in
diabetes. Consistent with TNF, STAT1, LCK, IKBKE, MAPK1 being among the 15.

### Diseases (OMIM)

No strong disease-specific associations — only a weak link to acute myeloid
leukemia (p≈3e-3, n=3) and ovarian/gastric cancer (not significant after
correction). Expected: the module reflects *general signalling biology*, not a
specific diagnosis.

---

## 4. How to interpret this (honestly)

- **What it really is:** the network returns as "candidates" the genes
  topologically close to the seed set. Because the seed set is large and
  functionally broad, and tyrosine-kinase signalling is the most connected part
  of the interactome, the module inevitably centres there. This is more a
  statement about **PPI-network structure** than a discovery about a disease.
- **The 6 shared genes** (LCK, STAT1, PLCG1, CBL, ATF2, MYH9) are the most
  stable result: two very different methods (simple connection counting +
  permutation vs. DIAMOnD greedy agglomeration) converged on them.
- **Scale difference:** DIAMOnD yields 100 genes (all "significant" by
  construction); the permutation test is stricter — 15 of 100. The intersection
  of 6 is what is robust to the choice of method.
- **Limitation & next step:** without a biologically meaningful, disease-specific
  seed set (e.g. the stroke genes in `disgenet-stroke-genes.txt`), conclusions
  stay about network topology. The logical next step is to re-run the whole
  pipeline with seed = stroke genes and check whether the DAVID profile changes.

---

## Result files

Curated under [`../results/network_pipeline/`](../results/network_pipeline/):

| File | What it is |
|---|---|
| `david_enrichment_barplot.png` | top-20 enriched terms |
| `david_enrichment_100genes.csv` | full DAVID table for the DIAMOnD module (656 terms) |
| `david_enrichment_15genes.csv` | DAVID table for the 15 significant genes |
| `diamond_vs_mine_pvalue_comparison.png` | p-value comparison |
| `diamond_vs_mine_euler.png` | Euler diagram of the overlap |
| `candidate_gene_significant.txt` | the 15 significant genes (branch A) |
| `diamond_candidate-genes.txt` | the 100 DIAMOnD module genes with p-values |
