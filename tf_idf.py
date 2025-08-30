import math
import pickle
import pandas as pd
import sqlite3


def load_inverse_index_and_docs():
    """
        Load the inverted index and the speeches dataframe.

        Returns:
            inverse_index (dict):
                word -> { doc_id: term_frequency, ... }

            df (pd.DataFrame):
                speeches joined with member and party info,
                columns: [doc_id, cleaned_speech, sitting_date, year, member_name, political_party]

            doc_id_to_index (dict):
                mapping from doc_id (used in DB/inverse_index) -> dataframe row index

            index_to_doc_id (dict):
                reverse mapping from dataframe index -> doc_id

        Steps:
            - Load inverse index from pickle ("inverse_index.pkl").
            - Load speeches + metadata from SQLite.
            - Reindex DataFrame and build mapping dicts for consistency.
    """
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


def compute_tf_idf_keywords_subset(inverse_index, df, doc_ids, top_n=10, return_scores=False):
    """
        Compute top TF-IDF keywords for a given set of documents.

        Args:
            inverse_index (dict):
                word -> { doc_id: term_frequency, ... }
            df (pd.DataFrame):
                speeches dataframe (must include 'cleaned_speech')
            doc_ids (list[int]):
                list of document indices (row indices in df)
            top_n (int):
                number of top keywords to return
            return_scores (bool):
                if True, return list of (word, score);
                if False, return list of words only

        Returns:
            list: top_n words or (word, score) pairs

        Notes:
            - Uses standard TF-IDF:
                TF = 1 + log(term_frequency)
                IDF = log(1 + N / df(word))
            - Aggregates scores over all doc_ids in the subset.
    """
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
