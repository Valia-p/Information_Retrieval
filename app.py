from flask import Flask, render_template, request, jsonify
from tf_idf import compute_tf_idf_similarity, load_inverse_index_and_docs
from query_processing import process_query  # ✅ import stemming pipeline

app = Flask(__name__)

# Load once at startup
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


if __name__ == "__main__":
    app.run(debug=True)
