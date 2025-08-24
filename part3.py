import pandas as pd
import math

from flask import jsonify

from tf_idf import load_inverse_index_and_docs
from query_processing import remove_accents


def compute_tf_idf_per_speech(inverse_index, df):
    """
    Computing TF-IDF per speech (doc)
    doc_id -> {word: tf-idf, ...}
    """
    num_docs = len(df)
    doc_vectors = {}

    for doc_id in range(num_docs):
        words = str(df.loc[doc_id, "cleaned_speech"]).split()
        word_counts = {}

        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        word_vectors = {}
        for word, tf in word_counts.items():
            if word not in word_counts:
                continue
            tf_weight = 1 + math.log(tf)
            idf = math.log(1 + num_docs / len(inverse_index[word]))
            word_vectors[word] = tf_weight * idf

        doc_vectors[doc_id] = word_vectors

    return doc_vectors


def create_member_catalogue(df):
    """
    Creation of inverse index catalogue that maps members to a list of speeches (docs) of theirs.
    member -> {doc1, doc2, ...}
    """
    member_speeches = {}

    for idx, row in df.iterrows():
        doc_id = row.get("document_id")
        # member = row.get("member_name", "").strip()
        member = str(row.get("member_name", "") or "").strip()

        if member not in member_speeches:
            member_speeches[member] = []
        member_speeches[member].append(doc_id)

    return member_speeches


def compute_tf_idf_per_member(member_speeches, vectors, inverse_index):
    """
    member -> {tf_idf1, tf_idf2, ...}
    """
    # keeping only the words (keys) of the inverted index catalogue
    # as a vocabulary with all the words that appear in the speeches.
    index = sorted(inverse_index.keys())

    member_vectors = {}

    for member, doc_ids in member_speeches.items():
        member_vec = {}
        for doc_id in doc_ids:
            vector = vectors.get(doc_id, {})

            for word, val in vector.items():
                if word not in member_vec:
                    member_vec[word] = 0.0
                member_vec[word] += val

        num_docs = len(doc_ids)
        if num_docs > 0:
            for word in member_vec:
                member_vec[word] /= num_docs

        member_vector = [member_vec.get(word, 0.0) for word in index]
        member_vectors[member] = member_vector

    return member_vectors


def compute_member_similarity(member_vectors, query, topk):
    """
    We apply cosine similarity between each member's vector
    and the wanted member (query).
    """
    if query not in member_vectors:
        print("not in members")
        return {}

    mem_vec = member_vectors[query]
    similarities = {}

    for member, vec in member_vectors.items():
        if member == query:
            continue
        dot = sum(a * b for a, b in zip(mem_vec, vec))
        norm1 = math.sqrt(sum(a * a for a in mem_vec))
        norm2 = math.sqrt(sum(a * a for a in vec))

        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = dot / (norm1 * norm2)

        similarities[member] = similarity

    top = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:topk]
    return [{"member": member, "score": score} for member, score in top]


def run_all_part3_tasks(data, inverse_index, df):
    member_speeches = create_member_catalogue(df)
    vectors = compute_tf_idf_per_speech(inverse_index, df)
    member_vectors = compute_tf_idf_per_member(member_speeches, vectors, inverse_index)
    topk = compute_member_similarity(member_vectors, data, 5)

    return topk

# topk = run_all_part3_tasks("αλευρας νικολαου ιωαννης")

# Δουλεύει με έτοιμο query όπως το παρακάτω. Μένει να το κάνουμε να παίρνει query από την ιστοσελίδα.
# query = "αλευρας νικολαου ιωαννης"
# cleaned_query = remove_accents(query.strip().lower())
