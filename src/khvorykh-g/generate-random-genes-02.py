"""
Generate N random genes for a given network.
"""

import argparse
import networkx as nx
import numpy as np

# Get command line arguments
parser = argparse.ArgumentParser(description="Generate N random genes for a given network.")
parser.add_argument("-n", "--network", help="path/to/network", 
                    required=True, type=str)
parser.add_argument("-g", "--genes", help="path/to/genes", 
                    required=True, type=str)
parser.add_argument("-r", "--replics", help="The number of replics", 
                    default=1000, type=int)
parser.add_argument("-o", "--output", help="path/to/output to save output",
                    required=True, type=str)
args = parser.parse_args() 

network = args.network
gene_file = args.genes
replics = args.replics
output = args.output

rs = np.random.RandomState(42)

# Load network
G = nx.read_edgelist(network)

# Get list of nodes
nodes = G.nodes()
print(f'Nodes: {len(nodes)}')

# Load genes
genes = []
with open(gene_file, 'r') as fh:
    for line in fh:
        genes.append(line.strip())
print(f'Genes loaded: {len(nodes)}')

# Find genes presented in network 
genes_in_network = [g for g in genes if g in nodes]
ngenes = len(genes_in_network)
print(f'Genes in network: {ngenes}')

# Get random genes
for i in range(replics):
    random_genes = rs.choice(nodes, ngenes, replace=False)
    output_file = output + f'-{i}.txt'
    with open(output_file, "w") as fh:
        for g in random_genes:
            print(g, file=fh)

