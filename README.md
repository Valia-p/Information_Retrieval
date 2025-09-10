# Greek Parliament Speeches (1989–2020) — IR Project

A web-based information retrieval app for exploring 1.28M speeches from the Hellenic Parliament (1989–2020).  
It offers search, keyword trends over time, member similarity, thematic clustering with LSI, outlier inspection, and **topic drift** (how themes shift year-to-year for a member or party).

---

## Features

- **Full-text search** across speeches with metadata filters (year, member, party).
- **Keywords over time** (TF–IDF): overall / per member / per party.
- **Member similarity**: k nearest “thematic neighbors” via cosine similarity.
- **Thematic analysis (LSI + KMeans)**:
  - Cluster overview (sizes, top keywords),
  - 2D embedding map (PCA of LSI),
  - Cluster details (party/member distribution, representative speech, samples),
  - **Outliers** inside each cluster (lowest cosine to centroid).
- **Topic Drift analysis** (LSI centroids by year): `drift(t)=1−cosine(centroid_t, centroid_{t−1})` for a member or party.

---

## Tech Stack

- **Backend:** Python, Flask, SQLite  
- **NLP/IR:** scikit-learn (TF–IDF, TruncatedSVD/LSI, KMeans), Greek stemmer, spaCy (el_core_news_sm)  
- **Frontend:** HTML/JS, Chart.js  
- **Data:** iMEdD-Lab Greek Parliament Proceedings (1989–2020)  

---

## Installation

**Prerequisites**
- Python 3.9+
- pip
- Install dependencies:
  ```bash
  pip install flask scikit-learn spacy greek-stemmer numpy scipy pandas
  python -m spacy download el_core_news_sm

## Data

The dataset comes from the official website of the Hellenic Parliament and covers all plenary session speeches from **July 1989 to July 2020**.  
It was collected and published by [iMEdD-Lab / Greek_Parliament_Proceedings](https://github.com/iMEdD-Lab/Greek_Parliament_Proceedings).

- **Total speeches:** 1,280,918  
- **Size:** ~2.3 GB  
- **Source files:** 5,355 plenary transcripts  
- **Time span:** 31 years (1989–2020)  

The original dataset is provided as a UTF-8 CSV with the following key columns:

- `speech` — full text of the speech  
- `member_name` — name of the MP  
- `political_party` — party of the MP  
- `sitting_date` — date of the session  
- `document_id` — unique identifier for the speech  

For development, a smaller subset can be used, but all preprocessing and analysis methods are designed to scale to the full dataset.  
Intermediate artifacts (e.g., `cleaned_data.csv`, `inverse_index.pkl`, `tfidf_matrix.npz`) and the normalized SQLite database (`parliament.db`) are built automatically on first run.

---

## Run

On **first run**, the pipeline will:
  - Clean and normalize the speeches (`cleaned_data.csv`)
  - Build inverted index (`inverse_index.pkl`)
  - Create SQLite DB (`parliament.db`)
  - Compute TF–IDF keywords (per speech / member-year / party-year)
  - Compute member similarity pairs
  - Compute LSI projections (`lsi_projected_docs.npz`)
  - Run clustering (`final_clustering_results.pkl`)

- Subsequent runs load the precomputed artifacts, so startup is fast.  

Run the app with:

```bash
python app.py
Open in browser at: http://127.0.0.1:5000/
```
---
## Methodology Notes

All heavy computations are performed **offline once** and stored as reusable artifacts.  
This design ensures fast responses during app usage and reproducible results.

**Pipeline overview:**
1. **Preprocessing**:  
   - Clean raw texts (remove stopwords, accents, punctuation, stemming/lemmatization).  
   - Save normalized speeches → `cleaned_data.csv`.  

2. **Indexing**:  
   - Build inverted index for fast retrieval → `inverse_index.pkl`.  
   - Map document IDs → `doc_ids.npy`.  

3. **Database**:  
   - Create normalized SQLite schema (`parliament.db`).  
   - Store speeches, members, parties, keywords, similarities.  

4. **TF–IDF Keywords**:  
   - Extract top keywords per speech, member-year, and party-year.  
   - Store in DB (`speech_keywords`, `member_keywords_by_year`, `party_keywords_by_year`).  

5. **Member Similarity**:  
   - Compute pairwise cosine similarities between MPs.  
   - Store in DB (`member_similarity_pairs`).  

6. **Latent Semantic Indexing (LSI)**:  
   - Apply Truncated SVD on TF–IDF matrix → `lsi_projected_docs.npz`.  
   - Lower-dimensional representation for semantic comparisons.  

7. **Clustering**:  
   - Run KMeans on LSI vectors → `final_clustering_results.pkl`.  
   - Enables thematic grouping, representative speeches, and outlier detection.  

8. **Topic Drift**:  
   - Compute yearly centroids in LSI space per MP/party.  
   - Drift metric = `1 − cosine(centroid_t, centroid_{t−1})`.  
   - Served dynamically via `/extras/topic_drift`.  

**Design principles:**
- **Precompute heavy steps** → app endpoints remain light.  
- **Artifact-based pipeline** → results are consistent and reloadable.  
- **Separation of concerns** → offline batch processing vs. online interactive queries.  
