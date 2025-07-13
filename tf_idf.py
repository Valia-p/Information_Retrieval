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
