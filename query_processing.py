import re
import unicodedata
from greek_stemmer import GreekStemmer
import spacy
from spacy.lang.el.stop_words import STOP_WORDS

stemmer = GreekStemmer()

try:
    nlp = spacy.load("el_core_news_sm")
except:
    raise RuntimeError("Download: python -m spacy el_core_news_sm")


def remove_accents(word: str) -> str:
    """
        Strip accents/diacritics from a Greek word using Unicode decomposition.

        Example:
            'πρόεδρος' -> 'προεδρος'
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', word)
        if unicodedata.category(c) != 'Mn'
    )


def clean_query_word(word: str) -> str:
    """
        Clean a token for query processing:
          - remove digits/punctuation/symbols
          - lowercase
          - remove tabs
          - filter out stopwords, too-short tokens, and non-alpha tokens

        Returns:
            normalized word string, or empty string if it should be discarded.

        Example:
            clean_query_word("Το!") -> "το"
            clean_query_word("123") -> ""
    """
    unwanted_pattern = re.compile(r'[0-9@#$%^&*()\-_=+\[\]{};:\'",.<>/?\\|`~!]')
    cleaned_word = re.sub(unwanted_pattern, '', word)
    cleaned_word = re.sub(r'\t', '', cleaned_word).strip().lower()

    if cleaned_word in STOP_WORDS or len(cleaned_word) <= 1 or not cleaned_word.isalpha():
        return ""
    return cleaned_word


def stem_query_word(word: str, pos: str) -> str:
    """
        Attempt to stem a query word using GreekStemmer.

        Args:
            word: cleaned lowercase token
            pos: spaCy POS tag (NOUN, VERB, ADJ, ADV, PROPN, etc.)

        Returns:
            stemmed token in lowercase, or empty string on failure.

        Note:
            - Currently, `stemmer = GreekStemmer()` is commented out above.
            - If not instantiated, this will fall into the exception handler
              and return "" (caller then falls back to lemma).
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
        stem_pos = pos_map.get(pos, "NNM")
        return stemmer.stem(word_clean).lower()
    except Exception:
        return ""


def process_query(text: str) -> list:
    """
        Full query preprocessing pipeline:
          - tokenize input text with spaCy
          - clean each token
          - try to stem, else fallback to lemma
          - return list of normalized tokens

        Args:
            text: raw query string

        Returns:
            List[str]: processed token list

        Example:
            process_query("Ο Πρόεδρος αγοράζει βιβλία") -> ["προεδρ", "αγοραζ", "βιβλι"]
    """
    doc = nlp(text.replace('\xa0', ' '))
    tokens = []

    for token in doc:
        raw = token.text
        cleaned = clean_query_word(raw)
        if not cleaned:
            continue

        stemmed = stem_query_word(cleaned, token.pos_)
        if not stemmed:
            stemmed = token.lemma_.lower()
        tokens.append(stemmed)

    return tokens
