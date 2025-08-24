import pandas as pd
import spacy
import re
from spacy.lang.el.stop_words import STOP_WORDS
from greek_stemmer import GreekStemmer
import unicodedata


#stemmer = GreekStemmer()

FILEPATH = "Greek_Parliament_Proceedings_1989_2020.csv"
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


def remove_accents(word: str) -> str:
    # Remove accents
    return ''.join(
        c for c in unicodedata.normalize('NFD', word)
        if unicodedata.category(c) != 'Mn'
    )


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

    **** POS map does not work. Fix in process if needed. ****
    """
    try:
        word_clean = remove_accents(word).upper()

        pos_map = {
            "NOUN": "NNM",
            "VERB": "VB",
            "ADJ": "JJM",
            "ADV": "JJM",
            "PROPN": "PRP"
        }
        stem_pos = pos_map.get(pos, "NNM")  # default to noun

        return stemmer.stem(word_clean).lower()
    except Exception:
        return ""


def clean_text(text: str, dictionary: dict) -> str:
    """
    Example:
        clean_text("Ευχαριστώ κύριε Πρόεδρε!", {}) -> "ευχαριστ προεδρ"
    """
    # Run the text through the spaCy NLP pipeline (tokenization, POS tagging, etc.)
    doc = nlp(text.replace('\xa0', ' '))
    result = []

    for token in doc:
        raw = token.text
        if raw in dictionary:
            result.append(dictionary[raw])
            continue

        cleaned = remove_unwanted_pattern(raw)
        if cleaned == "":
            continue

        stemmed = stem_word(cleaned, token.pos_)
        if stemmed == "":
            stemmed = token.lemma_.lower()

        dictionary[raw] = stemmed
        result.append(stemmed)

    cleaned_text = " ".join(result)

    # for testing
    if cleaned_text.strip():
        print(">>>", cleaned_text)

    return cleaned_text


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
    df = df[df["cleaned_speech"].str.strip() != ""]

    df = df.reset_index(drop=True)
    df["document_id"] = df.index

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned dataset to: {OUTPUT_FILE}")
