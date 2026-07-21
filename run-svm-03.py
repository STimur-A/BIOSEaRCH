"""
Apply SVM to predict disease genes.
Author: Gennady Khvorykh, info [at] inzilico.com
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import argparse
import pandas as pd
import sys

# Initiate command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data', help="path/to/file.csv with data")
parser.add_argument('--output', help="path/to/prefix to save output files")

# Get command line arguments
args = parser.parse_args()
data_file = args.data
output = args.output

# Load data
data = pd.read_csv(data_file, header=None)
print(f'Data loaded: {data.shape[0]} x {data.shape[1]}')

# Subset labeled data
idx = data.iloc[:,-1].isna()
X = data.iloc[-idx,1:-1]
y = data.iloc[-idx,-1].astype(int)
print(f'Labeled data: {X.shape[0]} x {X.shape[1]}')
print(f'Class 0: {sum(y == 0)}\nClass 1: {sum(y == 1)}')

# Subset unlabeled data
X_unlabeled = data.iloc[idx,1:-1]

# Get unlabeled nodes
unlabeled_nodes = data.iloc[idx,0].to_list()
print(f'Unlabeled nodes: {len(unlabeled_nodes)}')

# Split labeles data into test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid for the base SVC
param_grid = [
    # RBF kernel
    {
        'estimator__kernel': ['rbf'],
        'estimator__C': [0.01, 0.1, 1, 10, 100],
        'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    },
    # Linear kernel 
    {
        'estimator__kernel': ['linear'],
        'estimator__C': [0.1, 1, 10, 100]
    },
    # Polynomial kernel
    {
        'estimator__kernel': ['poly'],
        'estimator__C': [0.1, 1, 10],
        'estimator__gamma': ['scale', 0.1, 0.01],
        'estimator__degree': [2, 3]  
    }
]

# Create base SVC (without probability)
base_svc = SVC(random_state=42, verbose=False)

# Create CalibratedClassifierCV
calibrated_svc = CalibratedClassifierCV(
    estimator=base_svc, 
    method='sigmoid',
    n_jobs=5, 
    ensemble=False,
    cv=5
)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=calibrated_svc,
    param_grid=param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=5,
    verbose = 1
)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Get best model
model = grid_search.best_estimator_

# Get best parameters
best_parameters = grid_search.best_params_

# Estimate the quality
y_pred = model.predict(X_test_scaled)
cr = classification_report(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig(output + "-heatmap.png")

# Estimate the probabilities for class 1
y_pred_prob = model.predict_proba(X_test_scaled)[:,1]

auc = roc_auc_score(y_test, y_pred_prob)
with open(output + "-qc.txt", "w") as fh:
    print(f'Best parameters: {best_parameters}', file=fh)
    print(f'AUC: {auc:.3f}', file=fh)
    print(cr, file=fh)

# Plot ROC curve
plt.figure(figsize = (6,5))
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f'AUC: {auc:.3f}')
plt.plot([0,1],[0,1], 'k--', alpha=0.5, label = "Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(output + "-roc.png")

# Save best model
joblib.dump(model, output + '-model.pkl')

# Predict probability for unlabeled nodes
X_unlabeled_scaled = scaler.transform(X_unlabeled)
prob = model.predict_proba(X_unlabeled_scaled)[:,1]

# Define dataframe
df = pd.DataFrame({'genes':unlabeled_nodes, 'prob':prob})
df_sorted = df.sort_values('prob', ascending=False)

# Save probabilities to file
df_sorted.to_csv(output + '-prob.csv', index=False, header=False)