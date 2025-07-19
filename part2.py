import pandas as pd
import unidecode
import sqlite3
from tf_idf import load_inverse_index_and_docs, compute_tf_idf_keywords_subset


def normalize(text):
    return unidecode.unidecode(text.strip().lower())


def store_speech_keywords_to_db(conn, speech_keywords: dict):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM speech_keywords")
    for doc_id, keywords in speech_keywords.items():
        for word, score in keywords:
            cursor.execute("""
                INSERT INTO speech_keywords (speech_id, keyword, score)
                VALUES (?, ?, ?)
            """, (doc_id, word, score))
    conn.commit()


def store_member_keywords_by_year(conn, member_keywords: dict):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM member_keywords_by_year")
    for (member_id, year), keywords in member_keywords.items():
        for word, score in keywords:
            cursor.execute("""
                INSERT INTO member_keywords_by_year (member_id, year, keyword, score)
                VALUES (?, ?, ?, ?)
            """, (member_id, year, word, score))
    conn.commit()


def store_party_keywords_by_year(conn, party_keywords: dict):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM party_keywords_by_year")
    for (party_id, year), keywords in party_keywords.items():
        for word, score in keywords:
            cursor.execute("""
                INSERT INTO party_keywords_by_year (party_id, year, keyword, score)
                VALUES (?, ?, ?, ?)
            """, (party_id, year, word, score))
    conn.commit()


def run_all_part2_tasks():
    inverse_index, df, _, _ = load_inverse_index_and_docs()
    df["year"] = pd.to_datetime(df["sitting_date"], dayfirst=True).dt.year

    # Compute keywords per speech
    print("[...] Computing keywords per speech")
    speech_keywords = {}
    for doc_id in range(len(df)):
        keywords = compute_tf_idf_keywords_subset(inverse_index, df, [doc_id], top_n=5, return_scores=True)
        speech_keywords[doc_id] = keywords

    # Compute member keywords per year
    print("[...] Computing keywords per member by year")
    member_keywords = {}
    for (member, year), group in df.groupby(["member_name", "year"]):
        doc_ids = group.index.tolist()
        if not doc_ids:
            continue
        keywords = compute_tf_idf_keywords_subset(inverse_index, df, doc_ids, top_n=10, return_scores=True)

        with sqlite3.connect("parliament.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM members WHERE full_name = ?", (member,))
            result = cursor.fetchone()
            if result:
                member_id = result[0]
                member_keywords[(member_id, year)] = keywords

    # Compute party keywords per year
    print("[...] Computing keywords per party by year")
    party_keywords = {}
    for (party, year), group in df.groupby(["political_party", "year"]):
        doc_ids = group.index.tolist()
        if not doc_ids:
            continue
        keywords = compute_tf_idf_keywords_subset(inverse_index, df, doc_ids, top_n=10, return_scores=True)

        with sqlite3.connect("parliament.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM parties WHERE name = ?", (party,))
            result = cursor.fetchone()
            if result:
                party_id = result[0]
                party_keywords[(party_id, year)] = keywords

    # Save all to DB
    with sqlite3.connect("parliament.db") as conn:
        store_speech_keywords_to_db(conn, speech_keywords)
        store_member_keywords_by_year(conn, member_keywords)
        store_party_keywords_by_year(conn, party_keywords)

    print("[✓] Keywords stored in database.")


def find_entity_id_by_name(conn, table, field, target_name):
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, {field} FROM {table}")
    for entity_id, name in cursor.fetchall():
        if normalize(name) == normalize(target_name):
            return entity_id
    return None

