# Run this file after running app.py


# This file is for visualization. We can add it to site later.
# In the terminal the Cluster → Top TF-IDF Terms and LSI Dimensions with Top Terms ara printed.
# We can try reducing the number of clusters or LSI dimensions for testing.
# We can make it show the names of the dimensions (π.χ. "Οικονομία", "Εξωτερική Πολιτική", "Παιδεία")
# or show the list with top keywords (normal) per cluster.

# Of course, we need to create the route needed in Flask !!!!

import numpy as np
import pickle
from sklearn.decomposition import PCA
import plotly.express as px
from scipy.sparse import load_npz
import sqlite3

TFIDF_FILE = "tfidf_matrix.npz"
DOC_IDS_FILE = "doc_ids.npy"
LSI_OUTPUT_FILE = "lsi_projected_docs.npz"
CLUSTERS_FILE = "final_clustering_results.pkl"
TOP_TERMS = 10


def load_data():
    tfidf = load_npz(TFIDF_FILE).toarray()
    doc_ids = np.load(DOC_IDS_FILE)
    projected = np.load(LSI_OUTPUT_FILE)["data"]
    with open(CLUSTERS_FILE, "rb") as f:
        clusters = pickle.load(f)
    return tfidf, doc_ids, projected, clusters


def visualize_2d(projected, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(projected)

    fig = px.scatter(x=reduced[:, 0], y=reduced[:, 1], color=labels.astype(str),
                     title="2D LSI Projection (PCA)", labels={"color": "Cluster"})
    fig.show()


def visualize_3d(projected, labels):
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(projected)

    fig = px.scatter_3d(x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
                        color=labels.astype(str), title="3D LSI Projection (PCA)")
    fig.show()


def extract_cluster_themes(tfidf_matrix, doc_ids, clusters, top_terms=TOP_TERMS):
    reverse_map = np.load(DOC_IDS_FILE)
    conn = sqlite3.connect("parliament.db")
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT keyword FROM speech_keywords")
    terms = [row[0] for row in cursor.fetchall()]

    print("\nCluster → Top TF-IDF Terms:")
    for cluster_id, doc_list in clusters.items():
        if not doc_list:
            continue
        rows = [np.where(reverse_map == doc_id)[0][0] for doc_id in doc_list if doc_id in reverse_map]
        cluster_vec = tfidf_matrix[rows].mean(axis=0)
        top_indices = np.argsort(cluster_vec)[-top_terms:][::-1]
        top_keywords = [terms[i] for i in top_indices]
        print(f"Cluster {cluster_id}: {', '.join(top_keywords)}")

    conn.close()


def name_lsi_dimensions():
    tfidf = load_npz(TFIDF_FILE).toarray()
    pca = PCA(n_components=100)
    pca.fit(tfidf)

    cursor = sqlite3.connect("parliament.db").cursor()
    cursor.execute("SELECT DISTINCT keyword FROM speech_keywords")
    terms = [row[0] for row in cursor.fetchall()]

    print("\nLSI Dimensions with Top Terms:")
    for dim in range(5):  # Show first 5 dimensions
        comp = pca.components_[dim]
        top_term_ids = np.argsort(comp)[-TOP_TERMS:][::-1]
        top_terms = [terms[i] for i in top_term_ids]
        print(f"Dimension {dim}: {', '.join(top_terms)}")


if __name__ == "__main__":
    tfidf, doc_ids, projected, clusters = load_data()

    # Flatten cluster map to labels
    labels = np.zeros(len(doc_ids), dtype=int)
    doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    for cluster_id, doc_list in clusters.items():
        for doc_id in doc_list:
            if doc_id in doc_id_to_idx:
                labels[doc_id_to_idx[doc_id]] = cluster_id

    visualize_2d(projected, labels)
    visualize_3d(projected, labels)
    extract_cluster_themes(tfidf, doc_ids, clusters)
    name_lsi_dimensions()
