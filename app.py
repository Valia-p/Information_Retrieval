from flask import Flask, render_template, request, jsonify
from tf_idf import compute_tf_idf_similarity, load_inverse_index_and_docs
from query_processing import process_query
from data_cleaning import process_dataset
from inverted_index import create_inverse_index_catalogue
from part2 import run_all_part2_tasks, compute_keywords_by_year_for_entity, normalize
import json
import os

app = Flask(__name__)

# Ensure all necessary files are created
if not os.path.isfile("cleaned_data.csv"):
    process_dataset()

if not os.path.isfile("inverse_index.pkl"):
    create_inverse_index_catalogue()

if not os.path.exists("results") or not os.listdir("results"):
    run_all_part2_tasks()

# Load data once
inverse_index, df = load_inverse_index_and_docs()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")
    tokens = process_query(query)

    scores = compute_tf_idf_similarity(tokens)
    top_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]

    results = []
    for doc_id, score in top_docs:
        if score == 0:
            continue
        speech = df.loc[doc_id, "cleaned_speech"][:300] + "..."
        results.append({
            "doc_id": doc_id,
            "score": round(score, 4),
            "speech": speech,
            "member": df.loc[doc_id, "member_name"],
            "party": df.loc[doc_id, "political_party"],
            "date": df.loc[doc_id, "sitting_date"]
        })

    return jsonify(results)


@app.route("/keywords/by_year", methods=["POST"])
def keywords_by_year():
    data = request.get_json()
    entity_type = data.get("type")  # member or party
    entity_name = data.get("name")

    if not entity_type or not entity_name:
        return jsonify({"error": "Missing parameters"}), 400

    entity_column = "member_name" if entity_type == "member" else "political_party"
    norm_name = normalize(entity_name).replace(" ", "_")

    filename = f"keywords_per_year_{entity_column}_{norm_name}.json"
    filepath = os.path.join("results", filename)

    # Try to compute if not exist
    if not os.path.exists(filepath):
        try:
            compute_keywords_by_year_for_entity(inverse_index, df, entity_column, entity_name, top_n=10)
        except Exception as e:
            return jsonify({"error": f"Computation failed: {str(e)}"}), 500

        if not os.path.exists(filepath):
            return jsonify({"error": "No data found"}), 404

    with open(filepath, "r", encoding="utf-8") as f:
        result = json.load(f)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
