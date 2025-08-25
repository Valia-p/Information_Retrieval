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
    data = request.get_json(force=True) or {}
    raw_query = (data.get("query") or "").strip()
    date_range = (data.get("dateRange") or "all").strip().lower()   # "all" ή "YYYY-YYYY"
    party_name = (data.get("party") or "all").strip()
    mp_name    = (data.get("mp") or "all").strip()

    if not raw_query:
        return jsonify([])

    tokens = [t for t in process_query(raw_query) if t]
    if not tokens:
        return jsonify([])

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Φίλτρα που εφαρμόζονται στον πίνακα speeches
    where_sql = []
    params_base = []

    # dateRange "YYYY-YYYY"
    if date_range != "all":
        try:
            y1, y2 = date_range.split("-")
            y1, y2 = int(y1), int(y2)
            where_sql.append("s.year BETWEEN ? AND ?")
            params_base.extend([y1, y2])
        except Exception:
            pass

    # party
    if party_name and party_name.lower() != "all":
        where_sql.append("p.name = ?")
        params_base.append(party_name)

    # member
    if mp_name and mp_name.lower() != "all":
        where_sql.append("m.full_name = ?")
        params_base.append(mp_name)

    where_clause = ""
    if where_sql:
        where_clause = " AND " + " AND ".join(where_sql)

    from collections import defaultdict
    scores = defaultdict(float)

    for tok in tokens:
        sql = f"""
        SELECT sk.speech_id, sk.score
        FROM speech_keywords sk
        JOIN speeches s ON s.id = sk.speech_id
        JOIN members  m ON m.id = s.member_id
        JOIN parties  p ON p.id = s.party_id
        WHERE (sk.keyword = ? OR sk.keyword LIKE ?){where_clause}
        """
        params = [tok, f"{tok}%"] + params_base

        for sid, sc in cursor.execute(sql, params):
            scores[sid] += float(sc)

    if not scores:
        conn.close()
        return jsonify([])

    # Top-N αποτελέσματα που θα εμφανιστούν
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    speech_ids = [sid for sid, _ in top_docs]
    scores_dict = dict(top_docs)

    placeholders = ",".join("?" for _ in speech_ids)
    cursor.execute(f"""
        SELECT s.id, s.doc_id, s.speech, s.sitting_date, m.full_name, p.name
        FROM speeches s
        JOIN members m ON s.member_id = m.id
        JOIN parties p ON s.party_id = p.id
        WHERE s.id IN ({placeholders})
    """, tuple(speech_ids))
    rows = cursor.fetchall()
    conn.close()

    # Σύνθεση απάντησης
    results = []
    for sid, doc_id, speech, date, member, party in rows:
        excerpt = speech[:600] + "..." if len(speech) > 600 else speech
        results.append({
            "doc_id": doc_id,
            "score": round(scores_dict.get(sid, 0.0), 4),
            "speech": excerpt,
            "member": member,
            "party": party,
            "date": date
        })

    results.sort(key=lambda x: -x["score"])
    return jsonify(results)


@app.route("/entities", methods=["GET"])
def list_entities():
    etype = (request.args.get("type") or "").strip().lower()
    if etype not in {"overall", "member", "party"}:
        return jsonify({"error": "Invalid type"}), 400
    if etype == "overall":
        return jsonify({"items": []})

    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        if etype == "member":
            cur.execute("SELECT full_name FROM members ORDER BY full_name COLLATE NOCASE")
        else:
            cur.execute("SELECT name FROM parties ORDER BY name COLLATE NOCASE")
        items = [r[0] for r in cur.fetchall()]
        conn.close()
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"error": f"Database error: {e}"}), 500

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
