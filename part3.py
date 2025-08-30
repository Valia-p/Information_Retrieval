# part3.py
import sqlite3
import numpy as np
from typing import List, Tuple, Optional
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

DB_NAME = "parliament.db"

def _ensure_similarity_table(conn: sqlite3.Connection) -> None:
    """Create the member_similarity_pairs table if it does not already exist."""
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS member_similarity_pairs (
        member1_id INTEGER NOT NULL,
        member2_id INTEGER NOT NULL,
        score REAL NOT NULL,
        PRIMARY KEY (member1_id, member2_id),
        FOREIGN KEY (member1_id) REFERENCES members(id),
        FOREIGN KEY (member2_id) REFERENCES members(id)
    );
    """)
    conn.commit()


def _fetch_member_keyword_matrix(conn: sqlite3.Connection) -> Tuple[List[int], csr_matrix]:
    """
        Build a member × keyword matrix.
        Each member vector = AVERAGE of L2-normalized speech vectors.
         This avoids bias towards members with many or long speeches.

        Returns:
            (member_ids, X) where:
                member_ids = list of member IDs in row order
                X = csr_matrix (members × vocabulary)
    """
    cur = conn.cursor()

    # Vocabulary of keywords
    cur.execute("SELECT DISTINCT keyword FROM speech_keywords ORDER BY keyword")
    vocab = [r[0] for r in cur.fetchall()]
    if not vocab:
        return [], csr_matrix((0, 0), dtype=np.float32)
    vindex = {w: i for i, w in enumerate(vocab)}

    # --- Members who have speeches with keywords ---
    cur.execute("""
        SELECT DISTINCT s.member_id
        FROM speeches s
        JOIN speech_keywords sk ON sk.speech_id = s.id
        ORDER BY s.member_id
    """)
    member_ids = [r[0] for r in cur.fetchall()]
    if not member_ids:
        return [], csr_matrix((0, len(vocab)), dtype=np.float32)
    mpos = {mid: i for i, mid in enumerate(member_ids)}


    # Sparse matrix entries
    rows, cols, data = [], [], []
    # Counter: number of speeches per member
    speech_counts = {mid: 0 for mid in member_ids}

    # --- Loop over all speeches and build normalized vectors ---
    cur.execute("SELECT id, member_id FROM speeches")
    speeches = cur.fetchall()  # [(speech_id, member_id), ...]

    for speech_id, member_id in speeches:
        cur.execute("SELECT keyword, score FROM speech_keywords WHERE speech_id = ?", (speech_id,))
        kws = cur.fetchall()
        if not kws or member_id not in mpos:
            continue

        idxs, vals = [], []
        for kw, sc in kws:
            j = vindex.get(kw)
            if j is None:
                continue
            idxs.append(j)
            vals.append(float(sc))
        if not idxs:
            continue

        # Build dense vector for this speech
        v = np.zeros(len(vocab), dtype=np.float32)
        v[np.array(idxs, dtype=int)] = np.array(vals, dtype=np.float32)

        # L2 normalize the speech vector
        norm = np.linalg.norm(v)
        if norm == 0.0:
            continue
        v /= norm

        # Add to sparse matrix entries
        i = mpos[member_id]
        nz = np.nonzero(v)[0]
        for j in nz:
            rows.append(i)
            cols.append(int(j))
            data.append(float(v[j]))

        speech_counts[member_id] += 1

    # Build sparse matrix of summed speech vectors
    X_sum = csr_matrix((data, (rows, cols)), shape=(len(member_ids), len(vocab)), dtype=np.float32)

    # Convert to average per member (divide by # speeches)
    X_avg = X_sum.tocsr(copy=True)
    for mid, cnt in speech_counts.items():
        i = mpos[mid]
        if cnt > 0:
            start, end = X_avg.indptr[i], X_avg.indptr[i + 1]
            if end > start:
                X_avg.data[start:end] *= (1.0 / float(cnt))

    return member_ids, X_avg


def compute_and_store_all_pairs(min_score: float = 0.0, topk_per_member: Optional[int] = None) -> int:
    """
        Compute cosine similarity for ALL pairs of members, store them in DB.
        - min_score: ignore pairs with score < min_score
        - topk_per_member: if given, keep only top-k neighbors for each member
        Returns:
            number of pairs written/updated in DB
    """
    conn = sqlite3.connect(DB_NAME)
    try:
        _ensure_similarity_table(conn)
        cur = conn.cursor()

        member_ids, X = _fetch_member_keyword_matrix(conn)
        n = len(member_ids)
        if n < 2:
            return 0

        # Normalize all member vectors (row-wise L2 norm)
        Xn = normalize(X, norm='l2', axis=1, copy=True)

        written = 0
        for i in range(n):
            # Compute cosine similarity with all others
            sims = (Xn[i] @ Xn.T).toarray().ravel()
            sims[i] = 0.0  # exclude self-similarity

            # Candidate neighbors
            if topk_per_member is not None and topk_per_member < n - 1:
                # Partial argpartition for efficiency
                idx = np.argpartition(-sims, topk_per_member)[:topk_per_member]
                idx = idx[np.argsort(-sims[idx])]
            else:
                idx = np.argsort(-sims)   # descending order

            m1 = member_ids[i]
            for j in idx:
                if j <= i:
                    continue  # only store one copy (j>i)
                score = float(sims[j])
                if score < min_score:
                    continue
                m2 = member_ids[j]
                # Insert or replace pair similarity
                cur.execute("""
                    INSERT OR REPLACE INTO member_similarity_pairs (member1_id, member2_id, score)
                    VALUES (?, ?, ?)
                """, (m1, m2, score))
                written += 1

        conn.commit()
        return written
    finally:
        conn.close()


# def similar_to_member(name: str, topk: int = 10):
#     """
#         Given a member's full_name, return top-k most similar members.
#         Uses fresh computation from the keyword matrix (not only the DB).
#         Returns:
#             List of (other_member_name, score)
#     """
#     conn = sqlite3.connect(DB_NAME)
#     try:
#         cur = conn.cursor()
#         cur.execute("SELECT id FROM members WHERE full_name = ?", (name,))
#         r = cur.fetchone()
#         if not r:
#             return []
#         target_id = r[0]
#
#         # Build keyword matrix again
#         member_ids, X = _fetch_member_keyword_matrix(conn)
#         if target_id not in member_ids:
#             return []
#
#         # Locate row of the target member
#         pos = {mid: i for i, mid in enumerate(member_ids)}
#         i = pos[target_id]
#
#         # Compute cosine similarities
#         Xn = normalize(X, norm='l2', axis=1, copy=True)
#         sims = (Xn[i] @ Xn.T).toarray().ravel()
#         sims[i] = 0.0
#
#         # Select top-k indices
#         if topk >= len(member_ids) - 1:
#             idx = np.argsort(-sims)
#         else:
#             idx = np.argpartition(-sims, topk)[:topk]
#             idx = idx[np.argsort(-sims[idx])]
#
#         # Map IDs to names
#         cand_ids = [member_ids[j] for j in idx]
#         placeholders = ",".join("?" for _ in cand_ids)
#         cur.execute(f"SELECT id, full_name FROM members WHERE id IN ({placeholders})", tuple(cand_ids))
#         name_map = {r[0]: r[1] for r in cur.fetchall()}
#
#         return [(name_map.get(member_ids[j], str(member_ids[j])), float(sims[j])) for j in idx]
#     finally:
#         conn.close()
