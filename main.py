from data_cleaning import process_dataset
from inverted_index import create_inverse_index_catalogue
import os
from tf_idf import compute_tf_idf_similarity

if __name__ == '__main__':
    if not os.path.isfile("cleaned_data.csv"):
        process_dataset()

    create_inverse_index_catalogue()

    # testing tf-idf
    query_tokens = ['κυρι', 'προεδρ', 'νομοσχεδ', 'κυβερνης']
    scores = compute_tf_idf_similarity(query_tokens)

    # Get top-3
    top_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]

    for doc_id, score in top_docs:
        print(f"Document {doc_id} - Score: {score:.4f}")