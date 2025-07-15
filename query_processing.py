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
    return ''.join(
        c for c in unicodedata.normalize('NFD', word)
        if unicodedata.category(c) != 'Mn'
    )


def clean_query_word(word: str) -> str:
    unwanted_pattern = re.compile(r'[0-9@#$%^&*()\-_=+\[\]{};:\'",.<>/?\\|`~!]')
    cleaned_word = re.sub(unwanted_pattern, '', word)
    cleaned_word = re.sub(r'\t', '', cleaned_word).strip().lower()

    if cleaned_word in STOP_WORDS or len(cleaned_word) <= 1 or not cleaned_word.isalpha():
        return ""
    return cleaned_word


def stem_query_word(word: str, pos: str) -> str:
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
