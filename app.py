from flask import Flask, render_template, request, jsonify
from tf_idf import load_inverse_index_and_docs
from query_processing import process_query
from data_cleaning import process_dataset
from inverted_index import create_inverse_index_catalogue
from part2 import run_all_part2_tasks, find_entity_id_by_name
from create_database import create_schema, populate_data, is_part2_already_computed
from LSI import build_tfidf_matrix, perform_lsi, clustering_lsi_docs
import sqlite3
import os
from part3 import run_all_part3_tasks

DB_NAME = "parliament.db"
CSV_FILE = "cleaned_data.csv"
TFIDF_FILE = "tfidf_matrix.npz"
DOC_IDS_FILE = "doc_ids.npy"
LSI_OUTPUT_FILE = "lsi_projected_docs.npz"
CLUSTERS_FILE = "final_clustering_results.pkl"

# Create cleaned_data.csv if it does not exist
if not os.path.isfile(CSV_FILE):
    print("Creating cleaned_data.csv")
    process_dataset()

# Create inverse_index.pkl if it does not exist
if not os.path.isfile("inverse_index.pkl"):
    print("Creating inverse index catalogue")
    create_inverse_index_catalogue()

# Create db schema if it does not exist
if not os.path.isfile(DB_NAME):
    print(f"Creating SQLite database '{DB_NAME}'")
    conn = sqlite3.connect(DB_NAME)
    create_schema(conn)
    populate_data(conn, CSV_FILE)
    conn.close()

# Compute TF-IDF if it does not exist
if not is_part2_already_computed():
    try:
        print("Computing and storing TF-IDF keyword analysis...")
        run_all_part2_tasks()
    except Exception as e:
        print(f"Error while computing part2 tasks: {e}")
else:
    print("Keyword analysis already exists. Skipping part2 processing.")

# Files needed for LSI
if not os.path.exists(TFIDF_FILE) or not os.path.exists(DOC_IDS_FILE):
    build_tfidf_matrix()
else:
    print("TF-IDF already exists. Skipping...")

if not os.path.exists(LSI_OUTPUT_FILE):
    perform_lsi()
else:
    print("LSI projection already exists. Skipping...")

if not os.path.exists(CLUSTERS_FILE):
    clustering_lsi_docs()
else:
    print("Clustering already exists. Skipping...")

# Flask App
app = Flask(__name__)

# Load data and index
inverse_index, df, _, _ = load_inverse_index_and_docs()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify([])

    # Process user query (normalize, lowercase, stem, etc.)
    tokens = process_query(query)

    # Get TF-IDF scores for speeches that contain any of the keywords
    from collections import defaultdict
    scores = defaultdict(float)

    conn = sqlite3.connect("parliament.db")
    cursor = conn.cursor()

    for token in tokens:
        cursor.execute("""
            SELECT speech_id, score
            FROM speech_keywords
            WHERE keyword = ?
        """, (token,))
        for speech_id, score in cursor.fetchall():
            scores[speech_id] += score  # sum score if multiple keywords match

    if not scores:
        conn.close()
        return jsonify([])

    # Get top 5 speeches by score
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    doc_ids = [doc_id for doc_id, _ in top_docs]
    scores_dict = dict(top_docs)

    placeholders = ",".join("?" for _ in doc_ids)
    cursor.execute(f"""
        SELECT s.doc_id, s.speech, s.sitting_date,
               m.full_name, p.name
        FROM speeches s
        JOIN members m ON s.member_id = m.id
        JOIN parties p ON s.party_id = p.id
        WHERE s.doc_id IN ({placeholders})
    """, tuple(doc_ids))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for doc_id, speech, date, member, party in rows:
        excerpt = speech[:300] + "..." if len(speech) > 300 else speech
        results.append({
            "doc_id": doc_id,
            "score": round(scores_dict.get(doc_id, 0.0), 4),
            "speech": excerpt,
            "member": member,
            "party": party,
            "date": date
        })

    results.sort(key=lambda x: -x["score"])

    return jsonify(results)

@app.route("/keywords/by_year", methods=["POST"])
def keywords_by_year():
    data = request.get_json(force=True, silent=True) or {}
    entity_type = (data.get("type") or "").strip().lower()   # "overall" | "member" | "party"
    entity_name = (data.get("name") or "").strip()

    if not entity_type:
        return jsonify({"error": "Missing parameters"}), 400

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        if entity_type == "overall":
            # Συνολικά keywords ανά έτος από όλες τις ομιλίες (άθροιση score)
            cursor.execute("""
                SELECT s.year, sk.keyword, SUM(sk.score) AS score
                FROM speech_keywords sk
                JOIN speeches s ON s.id = sk.speech_id
                GROUP BY s.year, sk.keyword
                ORDER BY s.year ASC, score DESC
            """)
            rows = cursor.fetchall()
            conn.close()
            result = {}
            for year, keyword, score in rows:
                y = str(year if not isinstance(year, bytes) else int.from_bytes(year, "little"))
                result.setdefault(y, []).append({"keyword": keyword, "score": round(float(score), 4)})
            return jsonify(result)

        # --- member / party ---
        if entity_type == "member":
            table = "member_keywords_by_year"
            id_field = "member_id"
            lookup_table = "members"
            lookup_field = "full_name"
        elif entity_type == "party":
            table = "party_keywords_by_year"
            id_field = "party_id"
            lookup_table = "parties"
            lookup_field = "name"
        else:
            return jsonify({"error": "Invalid entity type"}), 400

        if not entity_name:
            conn.close()
            return jsonify({"error": "Missing parameters"}), 400

        entity_id = find_entity_id_by_name(conn, lookup_table, lookup_field, entity_name)
        if entity_id is None:
            conn.close()
            return jsonify({"error": f"{entity_type.title()} not found"}), 404

        cursor.execute(f"""
            SELECT year, keyword, score
            FROM {table}
            WHERE {id_field} = ?
            ORDER BY year ASC, score DESC
        """, (entity_id,))
        rows = cursor.fetchall()
        conn.close()

        result = {}
        for year, keyword, score in rows:
            y = str(year if not isinstance(year, bytes) else int.from_bytes(year, "little"))
            result.setdefault(y, []).append({"keyword": keyword, "score": round(float(score), 4)})
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route("/entities", methods=["GET"])
def list_entities():
    etype = (request.args.get("type") or "").strip().lower()
    if etype not in {"overall", "member", "party"}:
        return jsonify({"error": "Invalid type"}), 400

    if etype == "overall":
        return jsonify({"items": []})

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        if etype == "member":
            cursor.execute("SELECT full_name FROM members ORDER BY full_name COLLATE NOCASE LIMIT 20")
        else:  # party
            cursor.execute("SELECT name FROM parties ORDER BY name COLLATE NOCASE LIMIT 20")

        rows = [r[0] for r in cursor.fetchall()]
        conn.close()

        return jsonify({"items": rows})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Database error: {e}"}), 500

@app.route("/similarity", methods=["POST"])
def similarity():
    data = request.get_json()
    entity_name = data.get("member", "").strip()

    if not entity_name:
        return jsonify({"error": "Missing parameters"}), 400

    topk = run_all_part3_tasks(entity_name, inverse_index, df)

    return jsonify(topk)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
