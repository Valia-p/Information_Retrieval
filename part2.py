import os
import json
import pandas as pd
import unidecode
from tf_idf import load_inverse_index_and_docs, compute_tf_idf_keywords_subset


def save_results_to_json(data, filename):
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize(text):
    return unidecode.unidecode(text.strip().lower())


# Calculates top-N keywords for each speech (document)
def compute_keywords_per_speech(inverse_index, df, top_n=5):
    result = {}
    for doc_id in range(len(df)):
        keywords = compute_tf_idf_keywords_subset(inverse_index, df, [doc_id], top_n)
        result[str(doc_id)] = keywords
    save_results_to_json(result, "keywords_per_speech.json")


# Calculates top-N keywords for each member of parliament
def compute_keywords_per_speaker(inverse_index, df, top_n=10):
    result = {}
    speakers = df["member_name"].dropna().unique()
    for speaker in speakers:
        doc_ids = df[df["member_name"] == speaker].index.tolist()
        if doc_ids:
            keywords = compute_tf_idf_keywords_subset(inverse_index, df, doc_ids, top_n)
            result[speaker] = keywords
    save_results_to_json(result, "keywords_per_speaker.json")


# Calculates top-N keywords for each political party
def compute_keywords_per_party(inverse_index, df, top_n=10):
    result = {}
    parties = df["political_party"].dropna().unique()
    for party in parties:
        doc_ids = df[df["political_party"] == party].index.tolist()
        if doc_ids:
            keywords = compute_tf_idf_keywords_subset(inverse_index, df, doc_ids, top_n)
            result[party] = keywords
    save_results_to_json(result, "keywords_per_party.json")


# Calculates keywords by year for a specific entity (member of parliament or party)
# 'entity_column' is either 'member_name' or 'political_party'
# 'entity_name' is the normalized name to match
def compute_keywords_by_year_for_entity(inverse_index, df, entity_column, entity_name, top_n=10):
    df["year"] = pd.to_datetime(df["sitting_date"], dayfirst=True).dt.year

    entity_norm = normalize(entity_name)
    result = {}

    for year in sorted(df["year"].unique()):
        docs_year = df[df["year"] == year]
        filtered = docs_year[docs_year[entity_column].fillna("").apply(normalize) == entity_norm]
        doc_ids = filtered.index.tolist()
        if doc_ids:
            keywords = compute_tf_idf_keywords_subset(inverse_index, df, doc_ids, top_n)
            result[str(year)] = keywords

    filename = f"keywords_per_year_{entity_column}_{entity_norm.replace(' ', '_')}.json"
    save_results_to_json(result, filename)


# Calculates keyword evolution by year across all speeches
def keyword_evolution_by_year(inverse_index, df, top_n=10):
    df["year"] = pd.to_datetime(df["sitting_date"], dayfirst=True).dt.year

    result = {}

    for year in sorted(df["year"].unique()):
        doc_ids = df[df["year"] == year].index.tolist()
        if doc_ids:
            keywords = compute_tf_idf_keywords_subset(inverse_index, df, doc_ids, top_n)
            result[str(year)] = keywords

    save_results_to_json(result, "keywords_evolution_per_year.json")


# some example for testing
def demo_σκρεκας_nd(inverse_index, df):
    compute_keywords_by_year_for_entity(inverse_index, df, "member_name", "σκρεκας θεοδωρου κωνσταντινος", top_n=10)
    compute_keywords_by_year_for_entity(inverse_index, df, "political_party", "νεα δημοκρατια", top_n=10)


def run_all_part2_tasks():
    inverse_index, df = load_inverse_index_and_docs()

    compute_keywords_per_speech(inverse_index, df)
    compute_keywords_per_speaker(inverse_index, df)
    compute_keywords_per_party(inverse_index, df)
    keyword_evolution_by_year(inverse_index, df)
    demo_σκρεκας_nd(inverse_index, df)
