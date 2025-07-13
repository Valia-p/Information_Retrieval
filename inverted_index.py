import pandas as pd
import pickle


def get_number_of_docs():
    df = pd.read_csv("cleaned_data.csv")
    return len(df)


def create_inverse_index_catalogue():
    """
    The inverse index catalogue maps words to a list of documents containing the word and their term frequency.
    word → {document_id: term frequency,...}
    """
    df = pd.read_csv("cleaned_data.csv")
    inverse_index = {}

    for idx, row in df.iterrows():
        doc_id = row.get("document_id")
        speech = str(row.get("cleaned_speech", "")).strip()

        words = speech.split()

        for word in words:
            if word not in inverse_index:
                inverse_index[word] = {doc_id: 1}
            else:
                if doc_id in inverse_index[word]:
                    inverse_index[word][doc_id] += 1
                else:
                    inverse_index[word][doc_id] = 1

        # for testing
        if idx % 10 == 0 and idx > 0:
            print(f"Processed {idx} speeches...")
            print(inverse_index)

    # save in pickle to use for tf-idf
    with open("inverse_index.pkl", "wb") as f:
        pickle.dump(inverse_index, f)

    print(f"Inverted index created with {len(inverse_index)} unique words.")
    print("Saved to inverse_index.pkl.")
