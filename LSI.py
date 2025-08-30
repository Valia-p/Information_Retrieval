import os
import numpy as np
import sqlite3
import pickle
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.linalg import svd
from sklearn.cluster import KMeans

# File paths
DB_PATH = "parliament.db"
TFIDF_FILE = "tfidf_matrix.npz" # Αραιός πίνακας [ομιλίες × λέξεις] με TF-IDF score
DOC_IDS_FILE = "doc_ids.npy" # Λίστα με doc_id για κάθε γραμμή του πίνακα
LSI_OUTPUT_FILE = "lsi_projected_docs.npz" # Νέος πίνακας [ομιλίες × 100 διαστάσεις] μετά το SVD
CLUSTERS_FILE = "final_clustering_results.pkl" # Λεξικό {cluster_id: [doc_ids]} μετά το KMeans


# Parameters
K = 100           # Number of LSI dimensions
CLUSTERS = 100    # Number of clusters


def build_tfidf_matrix():
    """Builds a sparse TF-IDF matrix from the speech_keywords table in SQLite."""
    print("Building TF-IDF matrix...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all unique keywords
    cursor.execute("SELECT DISTINCT keyword FROM speech_keywords")
    all_keywords = sorted([row[0] for row in cursor.fetchall()])
    keyword_to_index = {kw: idx for idx, kw in enumerate(all_keywords)}

    # Get all unique speech IDs
    cursor.execute("SELECT DISTINCT speech_id FROM speech_keywords")
    all_doc_ids = sorted([row[0] for row in cursor.fetchall()])
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(all_doc_ids)}

    print(f"Keywords: {len(all_keywords)} | Documents: {len(all_doc_ids)}")

    # Prepare data for sparse matrix
    data, rows, cols = [], [], []
    cursor.execute("SELECT speech_id, keyword, score FROM speech_keywords")
    for speech_id, keyword, score in cursor.fetchall():
        if keyword in keyword_to_index and speech_id in doc_id_to_index:
            row = doc_id_to_index[speech_id]
            col = keyword_to_index[keyword]
            data.append(score)
            rows.append(row)
            cols.append(col)

    conn.close()

    # Build the sparse matrix and save
    matrix = csr_matrix((data, (rows, cols)),
                        shape=(len(all_doc_ids), len(all_keywords)),
                        dtype=np.float32)
    save_npz(TFIDF_FILE, matrix)
    np.save(DOC_IDS_FILE, np.array(all_doc_ids))
    print(f"Saved TF-IDF matrix → '{TFIDF_FILE}' and doc IDs → '{DOC_IDS_FILE}'")


def perform_lsi():
    """Performs LSI by applying SVD on the TF-IDF matrix."""
    if not os.path.exists(TFIDF_FILE):
        build_tfidf_matrix()

    print("Loading TF-IDF matrix for LSI...")
    tfidf = load_npz(TFIDF_FILE).toarray()  # convert sparse to dense
    print(f"TF-IDF shape: {tfidf.shape}")

    print(f"Performing SVD (keeping top {K} dimensions)...")
    U, S, Vt = svd(tfidf, full_matrices=False)
    U_k = U[:, :K]
    S_k = S[:K]

    projected_docs = U_k * S_k  # shape: (num_docs, K)

    np.savez_compressed(LSI_OUTPUT_FILE, data=projected_docs)
    print(f"Saved LSI-projected documents to '{LSI_OUTPUT_FILE}'")


def clustering_lsi_docs():
    """Performs clustering on LSI-projected documents and saves clusters."""
    if not os.path.exists(LSI_OUTPUT_FILE):
        perform_lsi()

    print("Loading LSI vectors for clustering...")
    data = np.load(LSI_OUTPUT_FILE)["data"]

    print(f"Clustering {data.shape[0]} documents into {CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)

    doc_ids = np.load(DOC_IDS_FILE)

    # Build cluster dictionary
    clusters = {i: [] for i in range(CLUSTERS)}
    for doc_id, label in zip(doc_ids, labels):
        clusters[label].append(doc_id)

    with open(CLUSTERS_FILE, "wb") as f:
        pickle.dump(clusters, f)

    print(f"Saved clustering results to '{CLUSTERS_FILE}'")

    # for testing
    for cid, docs in clusters.items():
        print(f"Cluster {cid} → {len(docs)} documents")
