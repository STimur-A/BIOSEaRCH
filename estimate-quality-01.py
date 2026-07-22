"""
Estimate the quality of the method
"""

import argparse
import pandas as pd

# Get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--genes', help='path/to/filename.csv with ranked candidate genes')
parser.add_argument('--removed', help='path/to/filename.txt with removed disease genes')
parser.add_argument('--percentile', help='Percentile to select top genes', type=float,
                    default=0.99)
args = parser.parse_args()
candidate_gene_path = args.genes
removed_gene_path = args.removed
percentile = float(args.percentile)

# Load data
df1 = pd.read_csv(candidate_gene_path, header=None)
removed_genes = []
with open(removed_gene_path, 'r') as fh:
    for line in fh:
        removed_genes.append(line.strip())

# Subset top genes by percentile
top_n = round(df1.shape[0] * (1 - percentile))
candidate_genes = df1.iloc[:top_n, 0].astype(str).to_list()

# Count sensitivity and specificity
tp = len(set(candidate_genes) & set(removed_genes))
fn = sum([1 for g in removed_genes if g not in candidate_genes])
sensetivity = tp/(tp + fn)
print(round(sensetivity, 4))


