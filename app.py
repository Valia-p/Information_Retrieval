from flask import Flask, render_template, request, jsonify
from tf_idf import load_inverse_index_and_docs
from query_processing import process_query
from data_cleaning import process_dataset
from inverted_index import create_inverse_index_catalogue
from part2 import run_all_part2_tasks, find_entity_id_by_name
from part3 import compute_and_store_all_pairs
from create_database import create_schema, populate_data, is_part2_already_computed, is_part3_already_computed
from LSI import build_tfidf_matrix, perform_lsi, clustering_lsi_docs
import sqlite3
import os
import unicodedata
import numpy as _np, pickle as _pickle
from scipy.sparse import load_npz as _load_npz

# --- File paths for persistent artifacts ---
DB_NAME = "parliament.db"
CSV_FILE = "cleaned_data.csv"
TFIDF_FILE = "tfidf_matrix.npz"
DOC_IDS_FILE = "doc_ids.npy"
LSI_OUTPUT_FILE = "lsi_projected_docs.npz"
CLUSTERS_FILE = "final_clustering_results.pkl"

# --- One-time data preparation steps ---
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

# Compute pairwise member similarities if not already done
if not is_part3_already_computed():
    try:
        print("Computing member–member similarities...")
        # thresholds settings:
        # min_score=0.05
        # topk_per_member=None για complete coverage
        stored = compute_and_store_all_pairs(min_score=0.05, topk_per_member=None)
        print(f"Stored/updated {stored} pairs in 'member_similarity_pairs'.")
    except Exception as e:
        print(f"Error while computing member similarities: {e}")
else:
    print("Member–member similarities already exist. Skipping part3 computations.")

# --- Files needed for LSI pipeline ---
# TF-IDF matrix
if not os.path.exists(TFIDF_FILE) or not os.path.exists(DOC_IDS_FILE):
    build_tfidf_matrix()
else:
    print("TF-IDF already exists. Skipping...")

# LSI projection (dimensionality reduction)
if not os.path.exists(LSI_OUTPUT_FILE):
    perform_lsi()
else:
    print("LSI projection already exists. Skipping...")

# Clustering results
if not os.path.exists(CLUSTERS_FILE):
    clustering_lsi_docs()
else:
    print("Clustering already exists. Skipping...")

# --- Helper functions for name normalization ---
def _strip_accents(s):
    """Remove accents from a string (normalize to ASCII)."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def _norm(s):
    """Lowercase + strip + accent-stripped form of a string."""
    return _strip_accents((s or "").strip().lower())

def find_member_id_fuzzy(conn, name: str):
    """Try to resolve a member name to ID (exact, case-insensitive, or fuzzy match)."""
    if not name: return None
    cur = conn.cursor()
    # exact
    cur.execute("SELECT id FROM members WHERE full_name = ?", (name.strip(),))
    r = cur.fetchone()
    if r: return r[0]
    # case-insensitive
    cur.execute("SELECT id FROM members WHERE full_name LIKE ? COLLATE NOCASE", (name.strip(),))
    r = cur.fetchone()
    if r: return r[0]
    # normalized
    nq = _norm(name)
    cur.execute("SELECT id, full_name FROM members")
    for mid, nm in cur.fetchall():
        nn = _norm(nm)
        if nn == nq or nn.startswith(nq) or nq in nn:
            return mid
    return None

# Flask App
app = Flask(__name__)

# Load data and index
inverse_index, df, _, _ = load_inverse_index_and_docs()

@app.route("/")
def index():
    """Main index page."""
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    """
        Search endpoint: performs keyword-based retrieval over speeches
        with optional filters (date range, party, member).
        Returns top-N matching speeches with snippets.
    """
    data = request.get_json(force=True) or {}
    raw_query = (data.get("query") or "").strip()
    date_range = (data.get("dateRange") or "all").strip().lower()   # "all" ή "YYYY-YYYY"
    party_name = (data.get("party") or "all").strip()
    mp_name    = (data.get("mp") or "all").strip()

    if not raw_query:
        return jsonify([])

    # Tokenize and normalize query
    tokens = [t for t in process_query(raw_query) if t]
    if not tokens:
        return jsonify([])

    # Build dynamic WHERE clause based on filters
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Φίλτρα που εφαρμόζονται στον πίνακα speeches
    where_sql = []
    params_base = []

    if date_range != "all":
        try:
            y1, y2 = date_range.split("-")
            y1, y2 = int(y1), int(y2)
            where_sql.append("s.year BETWEEN ? AND ?")
            params_base.extend([y1, y2])
        except Exception:
            pass

    # party filter
    if party_name and party_name.lower() != "all":
        where_sql.append("p.name = ?")
        params_base.append(party_name)

    # member filter
    if mp_name and mp_name.lower() != "all":
        where_sql.append("m.full_name = ?")
        params_base.append(mp_name)

    where_clause = ""
    if where_sql:
        where_clause = " AND " + " AND ".join(where_sql)

    # --- Scoring speeches ---
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

    # Take top-10 speeches by score
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

    # Build response
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
    """
        List entities (members or parties) to support dropdowns in the UI.
        GET /entities?type=member|party&full=1
    """

    etype = (request.args.get("type") or "").strip().lower()
    full  = (request.args.get("full") or "0").lower() in {"1","true","yes"}
    if etype not in {"overall","member","party"}:
        return jsonify({"error":"Invalid type"}), 400

    try:
        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        if etype == "overall":
            items = []
        elif etype == "member":
            if full:
                cur.execute("SELECT id, full_name FROM members ORDER BY full_name COLLATE NOCASE")
                items = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
            else:
                cur.execute("SELECT full_name FROM members ORDER BY full_name COLLATE NOCASE")
                items = [r[0] for r in cur.fetchall()]
        else:  # party
            if full:
                cur.execute("SELECT id, name FROM parties ORDER BY name COLLATE NOCASE")
                items = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
            else:
                cur.execute("SELECT name FROM parties ORDER BY name COLLATE NOCASE")
                items = [r[0] for r in cur.fetchall()]
        conn.close()
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"error": f"Database error: {e}"}), 500


@app.route("/keywords/by_year", methods=["POST"])
def keywords_by_year():
    """
        Return keywords aggregated per year for a given entity type.
        entity_type = "overall" | "member" | "party"
        entity_name = optional (required for member/party)

        - For "overall": sums keyword scores across all speeches by year.
        - For "member": looks up member_id and returns yearly keywords for that MP.
        - For "party": looks up party_id and returns yearly keywords for that party.
    """

    data = request.get_json(force=True, silent=True) or {}
    entity_type = (data.get("type") or "").strip().lower()   # "overall" | "member" | "party"
    entity_name = (data.get("name") or "").strip()

    if not entity_type:
        return jsonify({"error": "Missing parameters"}), 400

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        if entity_type == "overall":
            # Aggregate keyword scores across ALL speeches, grouped by year
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

        # Resolve name to ID
        entity_id = find_entity_id_by_name(conn, lookup_table, lookup_field, entity_name)
        if entity_id is None:
            conn.close()
            return jsonify({"error": f"{entity_type.title()} not found"}), 404

        # Pull yearly keywords for this entity
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


@app.route("/similarity/member", methods=["GET"])
def similarity_member_endpoint():
    """
        Given a member ID or name, return the most similar other members
        based on precomputed member–member similarities (cosine similarity).
        Params:
          - id   (optional): member_id
          - name (optional): full_name
          - k    (optional): top-k neighbors (default=10)
    """

    member_id = (request.args.get("id") or "").strip()
    name      = (request.args.get("name") or "").strip()
    topk      = int(request.args.get("k") or 10)

    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()

    # Find member_id
    if member_id:
        try:
            mid = int(member_id)
        except ValueError:
            conn.close(); return jsonify({"error":"Invalid member id."}), 400
        cur.execute("SELECT full_name FROM members WHERE id = ?", (mid,))
        r = cur.fetchone()
        if not r:
            conn.close(); return jsonify({"error": f"Member id {mid} not found."}), 404
        display_name = r[0]
    else:
        if not name:
            conn.close(); return jsonify({"error":"Provide member id or name."}), 400
        mid = find_member_id_fuzzy(conn, name)
        if not mid:
            conn.close(); return jsonify({"error": f"Member '{name}' not found."}), 404
        cur.execute("SELECT full_name FROM members WHERE id = ?", (mid,))
        display_name = cur.fetchone()[0]

    # Query top-k neighbors for this member
    cur.execute("""
        SELECT CASE WHEN member1_id=? THEN member2_id ELSE member1_id END AS other_id, score
        FROM member_similarity_pairs
        WHERE member1_id=? OR member2_id=?
        ORDER BY score DESC
        LIMIT ?
    """, (mid, mid, mid, topk))
    rows = cur.fetchall()

    if not rows:
        conn.close()
        return jsonify({"name": display_name, "neighbors": []})

    # Map neighbor IDs to names
    other_ids = [r[0] for r in rows]
    placeholders = ",".join("?" for _ in other_ids)
    cur.execute(f"SELECT id, full_name FROM members WHERE id IN ({placeholders})", other_ids)
    name_map = {rid: nm for rid, nm in cur.fetchall()}
    conn.close()

    neighbors = [{"member": name_map.get(oid, str(oid)), "score": float(sc)} for oid, sc in rows]
    return jsonify({"name": display_name, "neighbors": neighbors})


@app.route("/themes/overview", methods=["GET"])
def themes_overview():
    """
        Return cluster sizes and top TF-IDF keywords per cluster (uses precomputed files).
        Does NOT recompute LSI — it only reads:
          - tfidf_matrix.npz
          - doc_ids.npy
          - final_clustering_results.pkl
          - speech_keywords terms from DB
    """
    try:
        import numpy as _np, pickle as _pickle
        from scipy.sparse import load_npz as _load_npz

        # Load artifacts
        tfidf = _load_npz(TFIDF_FILE).toarray()
        doc_ids = _np.load(DOC_IDS_FILE)
        with open(CLUSTERS_FILE, "rb") as f:
            clusters = _pickle.load(f)

        # Get vocabulary order matching TF-IDF columns
        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("SELECT DISTINCT keyword FROM speech_keywords")
        terms = [row[0] for row in cur.fetchall()]
        conn.close()

        # Build reverse map from speech_id to row index in tfidf
        row_index = {int(sid): i for i, sid in enumerate(doc_ids.tolist())}

        def top_keywords_for_docs(doc_list, topn=8):
            rows = [row_index[sid] for sid in doc_list if int(sid) in row_index]
            if not rows:
                return []
            # Mean TF-IDF vector over cluster
            vec = tfidf[rows].mean(axis=0)
            top_idx = _np.argsort(vec)[-topn:][::-1]
            return [terms[i] for i in top_idx]

        overview = []
        for cid, docs in clusters.items():
            overview.append({
                "cluster_id": int(cid),
                "size": int(len(docs)),
                "top_keywords": top_keywords_for_docs([int(d) for d in docs], topn=8)
            })

        # sort by size desc
        overview.sort(key=lambda x: -x["size"])
        return jsonify({"clusters": overview})

    except Exception as e:
        return jsonify({"error": f"Failed to load themes: {e}"}), 500


@app.route("/themes/cluster")
def themes_cluster():
    """
        Return detailed information about a single cluster:
          - size, top keywords
          - party distribution
          - top members
          - representative speech (closest to centroid)
          - cohesion (avg. similarity to centroid)
          - sample speeches
          - date range and average speech length
    """

    cluster_id = request.args.get("id", type=int)
    if cluster_id is None:
        return jsonify({"error": "Missing id"}), 400

    try:
        # --- Load artifacts consistent with /themes/overview ---
        tfidf = _load_npz(TFIDF_FILE).toarray()          # shape: [docs x terms]
        doc_ids = _np.load(DOC_IDS_FILE)                 # speech_ids aligned with tfidf rows
        with open(CLUSTERS_FILE, "rb") as f:
            clusters = _pickle.load(f)                   # {cluster_id: [speech_ids]}

        if cluster_id not in clusters:
            return jsonify({"error": f"Cluster {cluster_id} not found"}), 404

        # Build row index: speech_id to tfidf row
        row_index = {int(sid): i for i, sid in enumerate(doc_ids.tolist())}

        # Load vocabulary (sorted for alignment with TF-IDF columns)
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT keyword FROM speech_keywords")
        terms = sorted([r[0] for r in cur.fetchall()])

        # --- Collect speeches of this cluster from DB ---
        ids = [int(d) for d in clusters[cluster_id]]
        placeholders = ",".join("?" for _ in ids) or "NULL"

        # --- Stats: parties, members, dates, avg length ---
        party_counts = {}
        member_counts = {}
        date_min = None
        date_max = None
        avg_chars = None

        rows_db = []
        if ids:
            cur.execute(f"""
                SELECT s.id, s.doc_id, s.speech, s.sitting_date, m.full_name, p.name
                FROM speeches s
                JOIN members m ON s.member_id = m.id
                JOIN parties p ON s.party_id = p.id
                WHERE s.id IN ({placeholders})
            """, tuple(ids))
            rows_db = cur.fetchall()

            total_chars, n_chars = 0, 0
            for sid, docid, speech, date, member, party in rows_db:
                if party:  party_counts[party] = party_counts.get(party, 0) + 1
                if member: member_counts[member] = member_counts.get(member, 0) + 1
                if date:
                    date_min = min(date_min, date) if date_min else date
                    date_max = max(date_max, date) if date_max else date
                if speech:
                    total_chars += len(speech); n_chars += 1
            avg_chars = (total_chars / n_chars) if n_chars else None

        # --- Top members ---
        member_top = [
            {"member": k, "count": v}
            for k, v in sorted(member_counts.items(), key=lambda x: -x[1])[:8]
        ]

        # --- Top keywords (mean TF-IDF over cluster) ---
        rows = [row_index[sid] for sid in ids if sid in row_index]
        if rows:
            vec = tfidf[rows].mean(axis=0)
            top_idx = _np.argsort(vec)[-12:][::-1]
            top_keywords = [terms[i] for i in top_idx]
        else:
            top_keywords = []

        # --- Representative doc (closest to centroid) & cohesion ---
        repr_doc = None
        avg_centroid_sim = None
        if rows:
            centroid = tfidf[rows].mean(axis=0)
            c_norm = _np.linalg.norm(centroid) or 1.0
            sims = []
            best_sim = -1.0; best_id = None
            for sid in ids:
                if sid not in row_index: continue
                v = tfidf[row_index[sid]]
                denom = (_np.linalg.norm(v) or 1.0) * c_norm
                sim = float(_np.dot(v, centroid) / denom)
                sims.append(sim)
                if sim > best_sim:
                    best_sim = sim; best_id = sid
            avg_centroid_sim = float(_np.mean(sims)) if sims else None

            if best_id is not None:
                cur.execute("""
                    SELECT s.id, s.doc_id, s.speech, s.sitting_date, m.full_name, p.name
                    FROM speeches s
                    JOIN members m ON s.member_id = m.id
                    JOIN parties p ON s.party_id = p.id
                    WHERE s.id = ?
                """, (best_id,))
                row = cur.fetchone()
                if row:
                    sid, docid, speech, date, member, party = row
                    repr_doc = {
                        "id": int(sid),
                        "doc_id": int(docid) if docid is not None else None,
                        "member": member,
                        "party": party,
                        "date": date,
                        "excerpt": (speech[:600] + ("..." if speech and len(speech) > 600 else "")) if speech else None
                    }

        # --- Sample speeches table (up to 10) ---
        samples = []
        if ids:
            cur.execute(f"""
                SELECT s.id, s.doc_id, s.speech, s.sitting_date, m.full_name, p.name
                FROM speeches s
                JOIN members m ON s.member_id = m.id
                JOIN parties p ON s.party_id = p.id
                WHERE s.id IN ({",".join("?" for _ in ids[:10])})
                ORDER BY s.id
            """, tuple(ids[:10]))
            for sid, docid, speech, date, member, party in cur.fetchall():
                excerpt = (speech[:400] + ("..." if speech and len(speech) > 400 else "")) if speech else ""
                samples.append({
                    "id": int(sid),
                    "doc_id": int(docid) if docid is not None else None,
                    "member": member,
                    "party": party,
                    "date": date,
                    "excerpt": excerpt
                })

        conn.close()
        return jsonify({
            "cluster_id": int(cluster_id),
            "size": len(ids),
            "top_keywords": top_keywords,
            "samples": samples,
            "party_counts": party_counts,
            "member_top": member_top,
            "date_min": date_min,
            "date_max": date_max,
            "avg_chars": avg_chars,
            "repr": repr_doc,
            "avg_centroid_sim": avg_centroid_sim
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load cluster {cluster_id}: {e}"}), 500


@app.route("/themes/embedding2d", methods=["GET"])
def themes_embedding2d():
    """
        Return a 2D PCA embedding of LSI-projected speeches,
        downsampled per cluster for visualization in the frontend.
        Output format:
          { "series": [ { "cluster_id": cid, "points": [[x,y], ...] }, ... ] }
    """

    try:
        # Load LSI-projected docs and cluster assignments
        data = _np.load(LSI_OUTPUT_FILE)["data"]           # (N_docs, K)
        doc_ids = _np.load(DOC_IDS_FILE)                   # (N_docs, )
        with open(CLUSTERS_FILE, "rb") as f:
            clusters = _pickle.load(f)                     # {cluster_id: [speech_ids]}

        # Map: speech_id to row index
        row_index = {int(sid): i for i, sid in enumerate(doc_ids.tolist())}

        # PCA via SVD to take first 2 components
        X = data.astype("float32")
        Xc = X - X.mean(axis=0, keepdims=True)
        _, _, Vt = _np.linalg.svd(Xc, full_matrices=False)
        W2 = Vt[:2].T
        coords = Xc @ W2                                  # (N_docs, 2)

        # Downsample to max N points per cluster
        MAX_PER_CLUSTER = int(request.args.get("per_cluster", 60))
        series = []
        for cid, doc_list in clusters.items():
            # keep only the ones that are in the row_index
            idxs = [row_index[int(d)] for d in doc_list if int(d) in row_index]
            if not idxs:
                series.append({"cluster_id": int(cid), "points": []})
                continue

            # Sampling
            if len(idxs) > MAX_PER_CLUSTER:
                rng = _np.random.default_rng(42 + int(cid))
                idxs = rng.choice(idxs, size=MAX_PER_CLUSTER, replace=False).tolist()

            pts = coords[idxs, :2].astype(float).tolist()  # [[x,y], ...]
            series.append({"cluster_id": int(cid), "points": pts})

        return jsonify({"series": series})

    except Exception as e:
        return jsonify({"error": f"Failed to build embedding: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
