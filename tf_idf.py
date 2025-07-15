import math
import pickle
import pandas as pd


def load_inverse_index_and_docs():
    with open("inverse_index.pkl", "rb") as f:
        inverse_index = pickle.load(f)
    df = pd.read_csv("cleaned_data.csv")
    return inverse_index, df


def compute_tf_idf_similarity(query_tokens):
    inverse_index, df = load_inverse_index_and_docs()
    num_docs = len(df)

    scores = [0.0] * num_docs
    doc_lengths = [0.0] * num_docs

    for word in query_tokens:
        if word not in inverse_index:
            continue
        # for each word in query, find in the docs that include it
        docs = inverse_index[word]
        idf = math.log(1 + num_docs / len(docs))

        for doc_id, tf in docs.items():
            tf_weight = 1 + math.log(tf)
            scores[doc_id] += tf_weight * idf

    for doc_id in range(num_docs):
        if scores[doc_id] == 0:
            continue

        words = str(df.loc[doc_id, "cleaned_speech"]).split()
        for word in words:
            if word in inverse_index and doc_id in inverse_index[word]:
                tf = inverse_index[word][doc_id]
                tf_weight = 1 + math.log(tf)
                idf = math.log(1 + num_docs / len(inverse_index[word]))
                doc_lengths[doc_id] += (tf_weight * idf) ** 2

        if doc_lengths[doc_id] > 0:
            scores[doc_id] /= math.sqrt(doc_lengths[doc_id])
        else:
            scores[doc_id] = 0.0

    return scores


# Use in part2.py
def compute_tf_idf_keywords_subset(inverse_index, df, doc_ids, top_n=10):
    num_docs_total = len(df)
    word_scores = {}

    for word in inverse_index:
        docs = inverse_index[word]
        for doc_id in doc_ids:
            if doc_id not in docs:
                continue
            tf = docs[doc_id]
            tf_weight = 1 + math.log(tf)
            idf = math.log(1 + num_docs_total / len(docs))
            if word not in word_scores:
                word_scores[word] = 0.0
            word_scores[word] += tf_weight * idf

    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]

