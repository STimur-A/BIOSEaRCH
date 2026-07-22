#!/usr/bin/env python3
# =============================================================================
# BIOSEaRCH — candidate-gene discovery for ischemic stroke
#   PPI-network analysis + machine learning on node2vec embeddings.
# Author:  Vyacheslav Kulakovskiy  ·  Кулаковский Вячеслав
# Project: "Bolshie Vyzovy 2026", track "Nature-like & neurotechnologies".
# License: MIT  (see LICENSE at the repository root)
# =============================================================================

"""
Create embedding of network with node2vec.
"""

import argparse
import os
from node2vec import Node2Vec
import networkx as nx
import pandas as pd

# Get command line arguments
parser = argparse.ArgumentParser(description='Create embedding of network with Node2Vec')
parser.add_argument('-n', '--network',
                    help='path/to/network',
                    required=True,
                    type=str)

parser.add_argument('-d', '--dim',
                    help='The dimensionality',
                    default=32,
                    type=int)

parser.add_argument('-o', '--output',
                    help='path/to/network',
                    required=True,
                    type=str)

args = parser.parse_args()

# Initiate variables
network_file = args.network
dim = args.dim
output_file = args.output

# Load network
G = nx.read_edgelist(network_file)

# Generate embedding
node2vec = Node2Vec(
    graph=G,
    dimensions=dim,
    walk_length=80,
    num_walks=10,
    p = 1, q = 1,
    workers=5,
    seed=42
)

# Fit the model
model = node2vec.fit(window=10,
                     min_count=1,
                     batch_words=4)

# Get nodes
nodes = list(G.nodes())

emb = pd.DataFrame(
    [model.wv[node] for node in nodes],
    index=nodes
)

out_dir = os.path.dirname(output_file)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
emb.to_csv(output_file, header=False)
print(f"Saved embedding to {output_file}")