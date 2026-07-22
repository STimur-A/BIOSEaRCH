import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

os.makedirs("pics", exist_ok=True)
os.makedirs("bin-tree", exist_ok=True)
os.makedirs("out4", exist_ok=True)

embedding_files = glob.glob("out2/emb*-outcome.csv")

# Список для сбора всех лучших параметров
all_best_params = []

for emb_file in embedding_files:
    dim = emb_file.split("emb")[1].split("-outcome")[0]
    
    print(f"\nОбработка размерности {dim}...")
    
    data = pd.read_csv(emb_file, header=None)  

    X = data.iloc[:, 1:-1].values  
    y = data.iloc[:, -1].values    

    labeled_idx = ~pd.isna(y)
    unlabeled_idx = pd.isna(y)

    X_labeled = X[labeled_idx]
    y_labeled = y[labeled_idx]

    X_unlabeled = X[unlabeled_idx]
    gene_ids_unlabeled = data.iloc[unlabeled_idx, 0].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled)

    # ========== GRID SEARCH ==========
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    model = DecisionTreeClassifier(random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        model, 
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"Best params for dim {dim}: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")

    best_model = grid_search.best_estimator_

    # ========== ОБУЧЕНИЕ С ЛУЧШИМИ ПАРАМЕТРАМИ ==========
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("+++++++++++++++++")
    cr = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
    print(cr)
    print("+++++++++++++++++")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Сохраняем лучшие параметры в текстовый файл
    with open(f'bin-tree/best_params-{dim}.txt', 'w') as f:
        f.write(f"Best params: {grid_search.best_params_}\n")
        f.write(f"Best CV score: {grid_search.best_score_:.3f}\n")
        f.write(f"Test Accuracy: {accuracy:.3f}\n\n")
        f.write(cr)
        f.write("\nConfusion matrix:\n")
        f.write(str(cm))

    # Сохраняем classification_report
    with open(f'bin-tree/classification_report-{dim}.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write("+++++++++++++++++\n")
        f.write(cr)
        f.write("+++++++++++++++++\n")
        f.write(str(cm))

    # Сохраняем лучшие параметры в CSV (для всех размерностей)
    all_best_params.append({
        'dimension': dim,
        'max_depth': grid_search.best_params_['max_depth'],
        'min_samples_split': grid_search.best_params_['min_samples_split'],
        'min_samples_leaf': grid_search.best_params_['min_samples_leaf'],
        'criterion': grid_search.best_params_['criterion'],
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': accuracy
    })

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'],
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"pics/conf_matr-{dim}.png")
    plt.close()

    # Предсказания для неразмеченных данных с вероятностями
    y_unlabeled_proba = best_model.predict_proba(X_unlabeled)[:, 1]

    # Сохраняем ВСЕ гены с вероятностями (отранжированные)
    results_df = pd.DataFrame({
        'gene_id': gene_ids_unlabeled,
        'probability': y_unlabeled_proba,
        'predicted_class': best_model.predict(X_unlabeled)
    }).sort_values('probability', ascending=False)

    results_df.to_csv(f'bin-tree/unlabeled_predictions-{dim}.csv', index=False)

    # Сохраняем топ-100 с вероятностями и классом
    top100_df = results_df.head(100)
    top100_df.to_csv(f'out4/top100_candidates-{dim}.csv', index=False)
    
    print(f"Готово для размерности {dim}")

# Сохраняем все лучшие параметры в один CSV файл
params_df = pd.DataFrame(all_best_params)
params_df.to_csv('bin-tree/all_best_params.csv', index=False)
print(f"\nВсе лучшие параметры сохранены в bin-tree/all_best_params.csv")