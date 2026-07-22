"""
Remove a given fraction of random genes from disease genes. Save this fraction and 
disease genes without this fraction. Repeat this selection r times. 
"""

import argparse
import networkx as nx
import numpy as np

# Get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--network", help="path/to/network", required=True, type=str)
parser.add_argument("-g", "--genes", help="path/to/genes", required=True, type=str)
parser.add_argument("-r", "--replics", help="The number of replics", default=100, type = int) 
parser.add_argument("-f", "--fraction", help="The fraction of genes to be removed", 
                    default=0.3, type=float)
parser.add_argument("-o", "--output", help="path/to/prefix to save output files",
                    required=True, type=str)
args = parser.parse_args() 

network_file = args.network
gene_file = args.genes
replics = args.replics
frac = args.fraction
output = args.output

rs = np.random.RandomState(42)

# Load network
G = nx.read_edgelist(network_file)

# Get list of nodes
nodes = G.nodes()
print(f'Nodes: {len(nodes)}')

# Load genes
genes = []
with open(gene_file, 'r') as fh:
    for line in fh:
        genes.append(line.strip())
print(f'Genes loaded: {len(nodes)}')

# Get genes presented in network 
genes_in_network = [g for g in genes if g in nodes]
ngenes = len(genes_in_network)
print(f'Genes in network: {ngenes}')

# Loop over the number of replics and remove a fraction of random genes
for i in range(replics):
    genes_removed = rs.choice(genes_in_network, round(ngenes * frac), replace=False)
    # Save disease genes removed
    with open(output + f'-removed-{i}.txt', "w") as fh:
        for g in genes_removed:
            print(g, file=fh)
    genes_left = [g for g in genes_in_network if g not in genes_removed]
    # Save disease genes left
    with open(output + f'-left-{i}.txt', "w") as fh:
        for g in genes_left:
            print(g, file=fh)

