import os
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. ОБЪЯВЛЕНИЕ ФУНКЦИЙ (Они ОБЯЗАТЕЛЬНО должны быть сверху)
# ==========================================

def load_data(network_path, genes_path):
    """
    Загружает граф интерактома и список генов заболеваний.
    """
    if not os.path.exists(network_path) or not os.path.exists(genes_path):
        raise FileNotFoundError("Проверьте правильность путей к файлам данных.")
        
    print("-> Загрузка генной сети...")
    df_net = pd.read_csv(network_path, sep='\t', header=0)
    
    col1 = df_net.columns[0]
    col2 = df_net.columns[1]
    df_net[col1] = df_net[col1].astype(str)
    df_net[col2] = df_net[col2].astype(str)
    
    G = nx.from_pandas_edgelist(df_net, source=col1, target=col2)
    print(f"   Сеть успешно загружена: {G.number_of_nodes()} узлов, {G.number_of_edges()} ребер.")
    
    print("-> Загрузка генов заболевания...")
    with open(genes_path, 'r', encoding='utf-8') as f:
        disease_genes = [line.strip() for line in f if line.strip()]
        
    print(f"   Считано генов заболевания из файла: {len(disease_genes)} шт.")
    return G, disease_genes


def generate_embeddings(graph, dimensions=8):
    """
    Создает эмбеддинги узлов сети фиксированной размерности 8.
    """
    print("-> Запуск Node2Vec (генерация случайных блужданий)...")
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=30, num_walks=150, workers=4, quiet=False)
    
    print("-> Обучение модели эмбеддингов...")
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    embeddings = {node: model.wv[node] for node in graph.nodes()}
    return embeddings


def create_feature_matrix(embeddings, positive_genes, ratio_negative=2.0):
    """
    Сборка X и y. 
    Метка 0 назначается генам, декартово расстояние которых максимально от центра масс 1.
    """
    all_genes = list(embeddings.keys())
    valid_positives = [g for g in positive_genes if g in embeddings]
    
    if not valid_positives:
        raise ValueError("Ни один из генов заболевания не пересекается с вашей сетью. Проверьте формат ID!")
        
    pos_vectors = np.array([embeddings[g] for g in valid_positives])
    mean_pos_vector = pos_vectors.mean(axis=0).reshape(1, -1)
    
    candidate_negatives = [g for g in all_genes if g not in valid_positives]
    cand_neg_vectors = np.array([embeddings[g] for g in candidate_negatives])
    
    distances = euclidean_distances(cand_neg_vectors, mean_pos_vector).flatten()
    sorted_indices = np.argsort(distances)[::-1]
    
    num_negatives = int(len(valid_positives) * ratio_negative)
    chosen_neg_indices = sorted_indices[:num_negatives]
    valid_negatives = [candidate_negatives[i] for i in chosen_neg_indices]
    
    X_list = []
    y_list = []
    
    for g in valid_positives:
        X_list.append(embeddings[g])
        y_list.append(1)
        
    for g in valid_negatives:
        X_list.append(embeddings[g])
        y_list.append(0)
        
    return np.array(X_list), np.array(y_list), valid_positives


def run_svm_prioritization(embeddings, X_train, y_train, positive_genes):
    """
    Обучает нелинейный SVM и скорит всю сеть генов.
    """
    print("-> Стандартизация пространства декартовых признаков...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("-> Обучение классификатора SVM для предсказания...")
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    all_genes = list(embeddings.keys())
    predict_genes = [g for g in all_genes if g not in positive_genes]
    
    X_predict = np.array([embeddings[g] for g in predict_genes])
    X_predict_scaled = scaler.transform(X_predict)
    
    probabilities = svm.predict_proba(X_predict_scaled)[:, 1]
    
    results = pd.DataFrame({
        'Entrez_ID': predict_genes,
        'Candidate_Probability': probabilities
    }).sort_values(by='Candidate_Probability', ascending=False)
    
    return results


# ==========================================
# 2. ИСПОЛНИТЕЛЬНЫЙ БЛОК (Строго в самом конце)
# ==========================================
if __name__ == "__main__":
    NETWORK_PATH = "/data/networks/DataS1_interactome_clean.tsv"
    GENES_PATH = "/data/genes/degs-entrez-02.txt"
    
    # 0. Загрузка данных (Теперь функция load_data точно известна Python)
    G, raw_disease_genes = load_data(NETWORK_PATH, GENES_PATH)
    
    # 1. Сжатие в эмбеддинги (8 измерений)
    gene_embeddings = generate_embeddings(G, dimensions=8)
    
    # 2. Вычисление декартова расстояния и сборка матрицы
    X, y, pos_used = create_feature_matrix(gene_embeddings, raw_disease_genes, ratio_negative=2.0)
    
    print(f"\n[Инфо] Обучающая матрица готова. Всего объектов: {X.shape}")
    print(f"[Инфо] Входные гены, найденные в сети (Класс 1): {len(pos_used)} шт.")
    
    # Резервное сохранение матриц
    np.save("X_matrix.npy", X)
    np.save("y_vector.npy", y)
    print("[Успех] Копии матриц X и y сохранены в npy-файлы.")
    
    # --- БЛОК: РАСЧЕТ SENSITIVITY ЧЕРЕЗ КРОСС-ВАЛИДАЦИЮ ---
    print("\n>>> Этап: Оценка качества модели (5-Fold Cross-Validation)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_true_all = []
    y_pred_all = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
        
        scaler_cv = StandardScaler()
        X_train_scaled = scaler_cv.fit_transform(X_train_cv)
        X_test_scaled = scaler_cv.transform(X_test_cv)
        
        svm_cv = SVC(kernel='rbf', class_weight='balanced', random_state=42)
        svm_cv.fit(X_train_scaled, y_train_cv)
        
        preds = svm_cv.predict(X_test_scaled)
        y_true_all.extend(y_test_cv)
        y_pred_all.extend(preds)
    
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("-" * 50)
    print(f" МЕТРИКИ КАЧЕСТВА МОДЕЛИ:")
    print(f" True Positives (TP): {tp}")
    print(f" False Negatives (FN): {fn}")
    print(f" Sensitivity (Чувствительность, TP / (TP + FN)): {sensitivity:.4f} ({sensitivity * 100:.2f}%)")
    print("-" * 50)
    
    # 3. SVM ранжирование генов
    print("\n>>> Этап: Финальное обучение SVM и ранжирование кандидатов...")
    candidates_df = run_svm_prioritization(gene_embeddings, X, y, pos_used)
    
    # 4. Срез ТОП-1% лучших кандидатов
    total_candidates = len(candidates_df)
    top_1_percent_count = max(1, int(np.ceil(total_candidates * 0.01)))
    top_candidates_df = candidates_df.head(top_1_percent_count)
    
    print(f"\n[Инфо] Всего генов-кандидатов для оценки: {total_candidates} шт.")
    print(f"[Инфо] Размер целевой выборки (1% лучших): {top_1_percent_count} шт.")
    
    print("\n>>> ПЕРВЫЕ СТРОКИ ИЗ ИТОГОВОГО ТОП-1% ФАЙЛА:")
    print(top_candidates_df.head(20).to_string(index=False))
    
    # Сохранение строго отфильтрованных 1% результатов в файл
    output_file = "top_1_percent_gene_candidates.tsv"
    top_candidates_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n[Успех] Файл с 1% лучших кандидатов сохранен в: {output_file}")



