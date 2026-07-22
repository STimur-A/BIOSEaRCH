#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Reduce the dimension of a node embedding with t-SNE and visualize it as a
2D or 3D scatter plot. Disease/seed genes can optionally be highlighted.

Output format is chosen by extension:
  *.html -> interactive, rotatable plotly plot (great for 3D)
  *.png  -> static matplotlib image (for slides)

Examples:
  # interactive 3D (rotate in browser)
  python3 plot-emb-02.py -e out-1/emb-new.csv -o out7/emb3d.html -g /data/genes/human-orthologs-02.txt -d 3
  # static 2D png (old behaviour)
  python3 plot-emb-02.py -e out-1/emb-new.csv -o out7/emb2d.png -g /data/genes/human-orthologs-02.txt -d 2
"""

import argparse
import os
import pandas as pd
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description="Visualize a node embedding as a 2D/3D scatter plot")
parser.add_argument('-e', '--embedding', required=True, type=str, help='path/to/embedding.csv (tab separated)')
parser.add_argument('-o', '--output', required=True, type=str, help='path/to/output (.html = interactive, .png = static)')
parser.add_argument('-g', '--genes', type=str, help='path/to/genes.txt with one gene id per line to highlight')
parser.add_argument('-d', '--dims', type=int, default=3, choices=[2, 3], help='number of dimensions to reduce to (2 or 3)')
args = parser.parse_args()

# --- Load embedding ---
data = pd.read_csv(args.embedding, header=None)
emb = data.iloc[:, 1:]
nodes = data.iloc[:, 0].astype(str).to_list()
print(f'Embedding shape: {emb.shape[0]} nodes x {emb.shape[1]} dims -> reducing to {args.dims}D')

# --- t-SNE ---
print("Applying t-SNE (this can take a while for large networks)...")
reduced = TSNE(n_components=args.dims, random_state=42).fit_transform(emb)

# --- Mark highlighted genes ---
is_gene = [False] * len(nodes)
if args.genes:
    with open(args.genes) as fh:
        genes = {line.strip() for line in fh if line.strip()}
    n = 0
    for i, node in enumerate(nodes):
        if node in genes:
            is_gene[i] = True
            n += 1
    print(f'Highlighted genes found in embedding: {n} / {len(genes)}')

df = pd.DataFrame(reduced, columns=[f'Dim {i+1}' for i in range(args.dims)])
df['node'] = nodes
df['group'] = ['gene' if g else 'other' for g in is_gene]

out_dir = os.path.dirname(args.output)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

# --- Plot ---
if args.output.endswith('.html'):
    import plotly.express as px
    # draw "other" first, "gene" on top
    df = df.sort_values('group', key=lambda s: s.map({'other': 0, 'gene': 1}))
    common = dict(color='group',
                  color_discrete_map={'other': 'lightgrey', 'gene': 'red'},
                  hover_name='node', opacity=0.8)
    if args.dims == 3:
        fig = px.scatter_3d(df, x='Dim 1', y='Dim 2', z='Dim 3', **common)
    else:
        fig = px.scatter(df, x='Dim 1', y='Dim 2', **common)
    fig.update_traces(marker=dict(size=3))
    fig.update_traces(marker=dict(size=6), selector=dict(name='gene'))
    fig.write_html(args.output)
else:
    import matplotlib.pyplot as plt
    colors = ['red' if g else 'darkgrey' for g in is_gene]
    sizes = [22 if g else 8 for g in is_gene]
    order = sorted(range(len(nodes)), key=lambda i: is_gene[i])  # genes last = on top
    fig = plt.figure(figsize=(10, 10))
    if args.dims == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[order, 0], reduced[order, 1], reduced[order, 2],
                   s=[sizes[i] for i in order], c=[colors[i] for i in order])
        ax.set_zlabel('Dim 3', fontsize=14)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(reduced[order, 0], reduced[order, 1],
                   s=[sizes[i] for i in order], c=[colors[i] for i in order])
    ax.set_xlabel('Dim 1', fontsize=14)
    ax.set_ylabel('Dim 2', fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)

print(f'{args.dims}D embedding saved to {args.output}')
