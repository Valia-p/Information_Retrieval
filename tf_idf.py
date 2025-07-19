import math
import pickle
import pandas as pd
import sqlite3


def load_inverse_index_and_docs():
    # Load inverse index from pickle
    with open("inverse_index.pkl", "rb") as f:
        inverse_index = pickle.load(f)

    # Load dataframe from SQLite
    conn = sqlite3.connect("parliament.db")
    df = pd.read_sql_query("""
        SELECT s.doc_id, s.cleaned_speech, s.sitting_date, s.year,
               m.full_name AS member_name,
               p.name AS political_party
        FROM speeches s
        JOIN members m ON s.member_id = m.id
        JOIN parties p ON s.party_id = p.id
        ORDER BY s.doc_id
    """, conn)
    conn.close()

    df = df.reset_index(drop=True)

    # Map doc_id (from inverted index) to df index
    doc_id_to_index = {row["doc_id"]: idx for idx, row in df.iterrows()}
    index_to_doc_id = {v: k for k, v in doc_id_to_index.items()}

    return inverse_index, df, doc_id_to_index, index_to_doc_id


# def compute_tf_idf_similarity(query_tokens):
#     inverse_index, df, doc_id_to_index, _ = load_inverse_index_and_docs()
#     num_docs = len(df)
#
#     # Βρες όλα τα doc_ids που περιέχουν τουλάχιστον μία λέξη του query
#     candidate_docs = set()
#     for word in query_tokens:
#         if word in inverse_index:
#             candidate_docs.update(inverse_index[word].keys())
#
#     scores = {}
#     doc_lengths = {}
#
#     for word in query_tokens:
#         if word not in inverse_index:
#             continue
#
#         docs = inverse_index[word]
#         idf = math.log(1 + num_docs / len(docs))
#
#         for doc_id, tf in docs.items():
#             if doc_id not in candidate_docs:
#                 continue
#
#             tf_weight = 1 + math.log(tf)
#             scores[doc_id] = scores.get(doc_id, 0.0) + tf_weight * idf
#
#     # Υπολογισμός κανονικοποιημένων scores (cosine similarity)
#     for doc_id in scores.keys():
#         idx = doc_id_to_index.get(doc_id)
#         if idx is None:
#             continue
#
#         doc_text = df.loc[idx, "cleaned_speech"]
#         doc_tokens = doc_text.split()
#         length_squared = 0.0
#
#         # Υπολογισμός μέτρου του διανύσματος TF-IDF για κάθε doc
#         seen = set()
#         for word in doc_tokens:
#             if word in seen or word not in inverse_index:
#                 continue
#             seen.add(word)
#
#             docs = inverse_index[word]
#             if doc_id not in docs:
#                 continue
#
#             tf = docs[doc_id]
#             tf_weight = 1 + math.log(tf)
#             idf = math.log(1 + num_docs / len(docs))
#             length_squared += (tf_weight * idf) ** 2
#
#         if length_squared > 0:
#             scores[doc_id] /= math.sqrt(length_squared)
#         else:
#             scores[doc_id] = 0.0
#
#     return scores  # {doc_id: score}


def compute_tf_idf_keywords_subset(inverse_index, df, doc_ids, top_n=10, return_scores=False):
    num_docs_total = len(df)
    word_scores = {}

    # Gather all words in doc_ids
    all_words = set()
    for doc_id in doc_ids:
        speech_text = str(df.loc[doc_id, "cleaned_speech"])
        all_words.update(speech_text.split())

    # Only for those words compute tf_idf
    for word in all_words:
        if word not in inverse_index:
            continue

        doc_freqs = inverse_index[word]
        idf = math.log(1 + num_docs_total / len(doc_freqs))

        for doc_id in doc_ids:
            if doc_id not in doc_freqs:
                continue
            tf = doc_freqs[doc_id]
            tf_weight = 1 + math.log(tf)
            score = tf_weight * idf
            word_scores[word] = word_scores.get(word, 0) + score

    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    if return_scores:
        return sorted_words[:top_n]
    else:
        return [word for word, _ in sorted_words[:top_n]]
