from rdflib import Graph
from pathlib import Path
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def extract_local_name(uri):
    if "#" in uri:
        return uri.split("#")[-1]
    elif "/" in uri:
        return uri.split("/")[-1]
    return uri

def load_triplets(g):
    triplets = []
    for subj, pred, obj in g:
        triplets.append((
            extract_local_name(str(subj)),
            extract_local_name(str(pred)),
            extract_local_name(str(obj))))

    triplets_df = pd.DataFrame(triplets, columns=["subject", "predicate", "object"])
    triplets_df.dropna(inplace=True)
    triplets_df.to_csv("triplets.csv", index=False)


g = Graph()
g.parse("Ontology_dnd.owl", format="xml")
if not Path("triplets.csv").is_file():
    load_triplets(g)



#
# train_threshold = 50
# train_df = triplets_df[triplets_df['subject'].str.len() < train_threshold]
# test_df = triplets_df[triplets_df['subject'].str.len() >= train_threshold]
#
# all_nodes = set(triplets_df["subject"]).union(triplets_df["object"])
# assert set(train_df["subject"]).union(train_df["object"]).issubset(all_nodes), "Not all nodes are in train."
# assert set(test_df["subject"]).union(test_df["object"]).issubset(all_nodes), "Not all nodes are in test."
#
# knowledge_graph = []
# for _, row in triplets_df.iterrows():
#     knowledge_graph.append((row["subject"], row["predicate"], row["object"]))
#
# # Объединение триплетов в общий список
# knowledge_graph = list(set(knowledge_graph))
# # print(knowledge_graph)
#
# triplets_text = [[s, p, o] for s, p, o in knowledge_graph]
#
# # Обучение Word2Vec
# model = Word2Vec(sentences=triplets_text, vector_size=100, window=5, min_count=1, workers=4, sg=1)
#
# # Векторы сущностей
# entity_vectors = {entity: model.wv[entity] for entity in model.wv.index_to_key}
#
#
# def evaluate_model(test_triplets, model, all_entities):
#     ranks = []
#     for s, p, o in test_triplets:
#         # Формирование негативных триплетов
#         negative_triplets = [(s, p, neg) for neg in all_entities if neg != o]
#         candidates = [(s, p, o)] + negative_triplets
#
#         # Оценка вероятности
#         scores = [(t, model.wv.similarity(t[0], t[2])) for t in candidates]
#         scores.sort(key=lambda x: x[1], reverse=True)
#
#         # Найти позицию правильного триплета
#         rank = [t[0] for t in scores].index((s, p, o)) + 1
#         ranks.append(rank)
#
#     mr = np.mean(ranks)
#     mrr = np.mean([1.0 / r for r in ranks])
#     hits_at_10 = np.mean([1 if r <= 10 else 0 for r in ranks])
#
#     return mr, mrr, hits_at_10
#
#
# all_entities = list(set(triplets_df['subject']).union(triplets_df['object']))
# test_triplets = list(test_df.itertuples(index=False, name=None))
#
# mr, mrr, hits_at_10 = evaluate_model(test_triplets, model, all_entities)
# print(f"MR: {mr}, MRR: {mrr}, Hits@10: {hits_at_10}")
#
# # Преобразование векторов для кластеризации
# vectors = np.array(list(entity_vectors.values()))
# pca = PCA(n_components=2)
# reduced_vectors = pca.fit_transform(vectors)
#
# # Кластеризация
# kmeans = KMeans(n_clusters=5)
# clusters = kmeans.fit_predict(reduced_vectors)
#
# # Визуализация
# plt.figure(figsize=(10, 8))
# plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap="viridis", s=50)
# plt.title("Clustering Results")
# plt.show()
#
#
# features = np.array([entity_vectors[e] for e in train_df['subject']])
# labels = train_df['predicate']
#
# # Разделение на обучающую и тестовую выборку
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#
# # Классификация
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)
#
# # Оценка модели
# y_pred = clf.predict(X_test)
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
#
# # Использование обученной модели для генерации ссылок
# def predict_links(triplet, model, all_entities):
#     s, p, o = triplet
#     candidates = [(s, p, e) for e in all_entities]
#     scores = [(t, model.wv.similarity(t[0], t[2])) for t in candidates]
#     scores.sort(key=lambda x: x[1], reverse=True)
#     return scores
#
# # Пример предсказания
# sample_triplet = test_triplets[0]
# predicted_links = predict_links(sample_triplet, model, all_entities)
# print(predicted_links[:5])  # Топ-5 предсказанных ссылок
