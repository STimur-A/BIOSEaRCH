# Input data

The pipelines take two kinds of input: **PPI networks** (edge lists) and
**gene lists** (Entrez IDs). These files are **not** stored in this repository
— they belong to third-party databases and/or are large. This page explains
what each file is and where to obtain it.

On the project server they live in the shared folders `/data/networks/` and
`/data/genes/`, and every script defaults to those paths. On another machine,
download the sources below and point the scripts at your local copies with the
`-n` / `-g` flags.

> **Note on IDs.** Genes are identified by **Entrez Gene ID** throughout. If you
> rebuild any list from a raw source, map it to Entrez IDs first (e.g. via
> [mygene.info](https://mygene.info)).

---

## Networks (`/data/networks/`)

| File | What it is | Source |
|---|---|---|
| `HS-PI.clean.tsv` | **Main PPI network** used in every run (~17 849 genes / ~633 460 edges), a cleaned human protein–protein interaction edge list | Human interactome; cleaned to a two-column Entrez edge list |
| `9606.protein.links.high.confidence.v12.0.clean.txt` | STRING high-confidence human links (not used in the current runs) | [STRING DB](https://string-db.org/) v12.0, organism 9606 |
| `HumanNet-plus_entrez.clean.tsv` | HumanNet+ functional network (not used in the current runs) | [HumanNet](https://www.inetbio.org/humannet/) |
| `alliance-network-clean.tsv` | Alliance interaction network (not used in the current runs) | [Alliance of Genome Resources](https://www.alliancegenome.org/) |
| `DataS1_interactome_clean.tsv` | Interactome from a published paper (not used in the current runs) | supplementary data of the source publication |

**Format** (all networks): whitespace/tab-separated **edge list**, two columns
of Entrez gene IDs per line — one interaction per row. Loaded with
`networkx.read_edgelist`.

---

## Gene lists (`/data/genes/`)

| File | What it is | Source |
|---|---|---|
| `human-orthologs-02.txt` | **Seed set** used in the current runs — a broad functional ortholog set (~865–890 genes present in the network). Note: this is a *broad* set, not stroke-specific — see `01_network_pipeline/FINDINGS.md` for what that implies | curated ortholog list |
| `disgenet-stroke-genes.txt` | Genes associated with stroke | [DisGeNET](https://www.disgenet.org/) |
| `degs-entrez-01.txt`, `degs-entrez-02.txt`, `degs-cells.txt` | Differentially expressed genes (blood/cells) — not used in the current runs | expression studies (e.g. Carmona-Mora et al.) |

**Format** (all gene lists): plain text, **one Entrez gene ID per line**.

---

## Reproducing a run on your own machine

```bash
# example: branch 1, step 1
python3 01_network_pipeline/scripts/01_explore_network.py \
    -n /path/to/HS-PI.clean.tsv \
    -g /path/to/human-orthologs-02.txt \
    -o common_nodes.txt
```

## Licensing

Each dataset is subject to the license of its original source (STRING, DisGeNET,
HumanNet, Alliance of Genome Resources, and the respective publications).
Consult and comply with those terms before redistributing any data or
data-derived results. This repository does not grant any rights to them.
