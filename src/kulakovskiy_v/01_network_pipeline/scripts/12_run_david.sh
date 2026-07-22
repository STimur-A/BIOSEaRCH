#!/bin/bash
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

# Step 13 - Functional annotation of a gene list via DAVID (davidbioinformatics.nih.gov).
#
# DAVID is a stateful web tool: api.jsp validates the IDs and returns an
# auto-submitting form -> chartReport.jsp renders the enrichment table.
# We replicate that flow with curl + a cookie jar (one session).
#
# NOTE: DAVID is blocked from the GPU server (curl returns 000). Run this
# from a machine that can reach davidbioinformatics.nih.gov (e.g. a laptop),
# then parse the HTML with pandas.read_html (table index 2).
#
# Usage:
#   ./13_run_david.sh "1386,867,3932,..." out_chart.html
#
# Parse the result:
#   python3 -c "import pandas as pd; t=pd.read_html('out_chart.html')[2]; \
#     print(t[['Category','Term','Count','P-Value','Benjamini']].dropna().sort_values('P-Value').head(30))"
set -euo pipefail

IDS=$1
OUT=${2:-david_chart.html}
ANNOT="GOTERM_BP_DIRECT,GOTERM_MF_DIRECT,GOTERM_CC_DIRECT,KEGG_PATHWAY,OMIM_DISEASE"
UA="Mozilla/5.0"
COOKIES=$(mktemp)

# 1. submit list -> get validated rowids
curl -s -m 30 -c "$COOKIES" -A "$UA" \
    "https://davidbioinformatics.nih.gov/api.jsp?type=ENTREZ_GENE_ID&ids=${IDS}&tool=chartReport&annot=${ANNOT}" \
    -o /tmp/david_api.html

ROWIDS=$(grep -o 'rowids.value="[^"]*"' /tmp/david_api.html | sed 's/rowids.value="//; s/"//')
ANNOTC=$(grep -o 'annot.value="[^"]*"'  /tmp/david_api.html | sed 's/annot.value="//; s/"//')

# 2. POST rowids -> render enrichment chart
curl -s -m 60 -b "$COOKIES" -c "$COOKIES" -A "$UA" \
    --data-urlencode "rowids=${ROWIDS}" \
    --data-urlencode "annot=${ANNOTC}" \
    'https://davidbioinformatics.nih.gov/chartReport.jsp' \
    -o "$OUT"

rm -f "$COOKIES"
echo "DAVID chart saved to $OUT ($(wc -c < "$OUT") bytes)"
