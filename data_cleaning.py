import pandas as pd
import spacy
import re
from spacy.lang.el.stop_words import STOP_WORDS
from greek_stemmer import GreekStemmer

# testing
stemmer = GreekStemmer()
print(stemmer.stem("ΕΥΧΑΡΙΣΤΗΣΑΤΕ"))
print(stemmer.stem("ΕΛΛΑΔΑ"))

FILEPATH = "Greek_Parliament_Proceedings_1989_2020_DataSample.csv"
OUTPUT_FILE = "cleaned_data.csv"

try:
    nlp = spacy.load("el_core_news_sm")
except:
    raise RuntimeError("Download: python -m spacy el_core_news_sm")


def remove_unwanted_pattern(word: str) -> str:
    """
    Cleans a word by removing symbols, stopwords, and invalid tokens.

    Parameters:
        word (str): The original token.

    Returns:
        str: A cleaned word, or empty string if it's not valid.

    Example:
        remove_unwanted_pattern("!Το@") -> "το"
        remove_unwanted_pattern("123") -> ""
    """

    # Compile a regex to remove punctuation, symbols, numbers, etc.
    unwanted_pattern = re.compile(r'[0-9@#$%^&*()\-_=+\[\]{};:\'",.<>/?\\|`~!]')

    # Apply pattern and remove tabs
    cleaned_word = re.sub(unwanted_pattern, '', word)
    cleaned_word = re.sub(r'\t', '', cleaned_word).strip().lower()

    if cleaned_word in STOP_WORDS or len(cleaned_word) <= 1 or not cleaned_word.isalpha():
        return ""

    return cleaned_word


def stem_word(word: str, pos: str) -> str:
    """
    Stem a word based on its part of speech tag.

    Parameters:
        word (str): The cleaned input word.
        pos (str): The POS tag from spaCy (NOUN, VERB, etc.).

    Returns:
        str: Stemmed word or empty string if stemming fails.

    Example:
    stem_word("αγοράζει", "VERB") -> "αγοραζ"
    stem_word("πρόεδρος", "NOUN") -> "προεδρ"
    """

    pos_map = {
        "NOUN": "NNM",
        "VERB": "VB",
        "ADJ": "JJM",
        "ADV": "JJM",
        "PROPN": "PRP"
    }
    stem_pos = pos_map.get(pos, "NNM")  # default to noun

    try:
        # Pass word in uppercase to match stemmer requirements
        return stemmer.stem_word(word.upper(), stem_pos).lower()
    except:
        return ""


def clean_text(text: str, dictionary: dict) -> str:
    """
    Example:
        clean_text("Ευχαριστώ κύριε Πρόεδρε!", {}) -> "ευχαριστ προεδρ"
    """
    # Run the text through the spaCy NLP pipeline (tokenization, POS tagging, etc.)
    doc = nlp(text.replace('\xa0', ' '))  # Replace non-breaking spaces with normal spaces
    result = []

    for token in doc:
        raw = token.text
        # If we've already processed this word, reuse it from the dictionary
        if raw in dictionary:
            result.append(dictionary[raw])
            continue

        # Remove unwanted characters, stopwords, and invalid tokens
        cleaned = remove_unwanted_pattern(raw)
        if cleaned == "":
            continue

        # Try to stem the word based on its part of speech
        stemmed = stem_word(cleaned, token.pos_)
        # If stemming fails, fall back to lemmatization
        if stemmed == "":
            stemmed = token.lemma_.lower()

        dictionary[raw] = stemmed
        result.append(stemmed)
    # for testing
    print(">>>", result)

    return " ".join(result)


def process_dataset():
    df = pd.read_csv(FILEPATH)
    df = df.dropna(subset=["speech"])  # delete empty speeches
    df = df.reset_index(drop=True)

    df["document_id"] = df.index  # need for tf-idf

    dictionary = {}  # dictionary for stemming
    cleaned_speeches = []

    for speech in df["speech"]:
        cleaned = clean_text(speech, dictionary)
        cleaned_speeches.append(cleaned)

    df["cleaned_speech"] = cleaned_speeches
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned dataset to: {OUTPUT_FILE}")
