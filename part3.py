# part3.py
import sqlite3
import numpy as np
from typing import List, Tuple, Optional
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

DB_NAME = "parliament.db"

def _ensure_similarity_table(conn: sqlite3.Connection) -> None:
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
    Χτίζει X (members × keywords) ως ΜΕΣΟ ΟΡΟ των L2-κανονικοποιημένων διανυσμάτων ομιλιών
    από τον πίνακα speech_keywords. Έτσι ΔΕΝ ευνοούνται μέλη με πολλές/μακριές ομιλίες.
    Επιστρέφει (member_ids, X) όπου X είναι csr_matrix.
    """
    cur = conn.cursor()

    # Λεξιλόγιο
    cur.execute("SELECT DISTINCT keyword FROM speech_keywords ORDER BY keyword")
    vocab = [r[0] for r in cur.fetchall()]
    if not vocab:
        return [], csr_matrix((0, 0), dtype=np.float32)
    vindex = {w: i for i, w in enumerate(vocab)}

    # Μέλη που έχουν ομιλίες με keywords
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

    # Άθροισμα μονάδων-ομιλιών ανά μέλος
    rows, cols, data = [], [], []
    speech_counts = {mid: 0 for mid in member_ids}

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

        v = np.zeros(len(vocab), dtype=np.float32)
        v[np.array(idxs, dtype=int)] = np.array(vals, dtype=np.float32)

        # L2 κανονικοποίηση της ομιλίας
        norm = np.linalg.norm(v)
        if norm == 0.0:
            continue
        v /= norm

        i = mpos[member_id]
        nz = np.nonzero(v)[0]
        for j in nz:
            rows.append(i)
            cols.append(int(j))
            data.append(float(v[j]))

        speech_counts[member_id] += 1

    X_sum = csr_matrix((data, (rows, cols)), shape=(len(member_ids), len(vocab)), dtype=np.float32)

    # Μέσος όρος ανά μέλος
    X_avg = X_sum.tocsr(copy=True)
    for mid, cnt in speech_counts.items():
        i = mpos[mid]
        if cnt > 0:
            start, end = X_avg.indptr[i], X_avg.indptr[i + 1]
            if end > start:
                X_avg.data[start:end] *= (1.0 / float(cnt))

    return member_ids, X_avg


def compute_and_store_all_pairs(min_score: float = 0.0,
                                topk_per_member: Optional[int] = None) -> int:
    """
    Υπολογίζει cosine για ΟΛΑ τα ζεύγη μελών και τα αποθηκεύει ΜΙΑ φορά.
    - min_score: αγνόησε ζεύγη με score < min_score
    - topk_per_member: αν δοθεί, για κάθε μέλος κρατά μόνο τους top-k πιο όμοιους
    Επιστρέφει πόσα ζεύγη γράφτηκαν/ανανεώθηκαν.
    """
    conn = sqlite3.connect(DB_NAME)
    try:
        _ensure_similarity_table(conn)
        cur = conn.cursor()

        member_ids, X = _fetch_member_keyword_matrix(conn)
        n = len(member_ids)
        if n < 2:
            return 0

        Xn = normalize(X, norm='l2', axis=1, copy=True)

        written = 0
        for i in range(n):
            sims = (Xn[i] @ Xn.T).toarray().ravel()
            sims[i] = 0.0  # exclude self

            # επιλογή υποψηφίων
            if topk_per_member is not None and topk_per_member < n - 1:
                idx = np.argpartition(-sims, topk_per_member)[:topk_per_member]
                idx = idx[np.argsort(-sims[idx])]
            else:
                idx = np.argsort(-sims)  # φθίνουσα

            m1 = member_ids[i]
            for j in idx:
                if j <= i:
                    continue  # γράφουμε μόνο για j>i → μία εγγραφή/ζεύγος
                score = float(sims[j])
                if score < min_score:
                    continue
                m2 = member_ids[j]
                cur.execute("""
                    INSERT OR REPLACE INTO member_similarity_pairs (member1_id, member2_id, score)
                    VALUES (?, ?, ?)
                """, (m1, m2, score))
                written += 1

        conn.commit()
        return written
    finally:
        conn.close()


def is_part3_already_computed() -> bool:
    """
    Επιστρέφει True αν ο πίνακας member_similarity_pairs υπάρχει ΚΑΙ έχει τουλάχιστον 1 εγγραφή.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='member_similarity_pairs'")
        exists = (cur.fetchone()[0] == 1)
        if not exists:
            conn.close()
            return False
        cur.execute("SELECT COUNT(*) FROM member_similarity_pairs")
        done = (cur.fetchone()[0] > 0)
        conn.close()
        return done
    except Exception:
        return False


def similar_to_member(name: str, topk: int = 10):
    """Επιστρέφει [(άλλος_βουλευτής, score), ...] για χρήση στο UI."""
    conn = sqlite3.connect(DB_NAME)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM members WHERE full_name = ?", (name,))
        r = cur.fetchone()
        if not r:
            return []
        target_id = r[0]

        member_ids, X = _fetch_member_keyword_matrix(conn)
        if target_id not in member_ids:
            return []

        pos = {mid: i for i, mid in enumerate(member_ids)}
        i = pos[target_id]
        Xn = normalize(X, norm='l2', axis=1, copy=True)
        sims = (Xn[i] @ Xn.T).toarray().ravel()
        sims[i] = 0.0

        if topk >= len(member_ids) - 1:
            idx = np.argsort(-sims)
        else:
            idx = np.argpartition(-sims, topk)[:topk]
            idx = idx[np.argsort(-sims[idx])]

        cand_ids = [member_ids[j] for j in idx]
        placeholders = ",".join("?" for _ in cand_ids)
        cur.execute(f"SELECT id, full_name FROM members WHERE id IN ({placeholders})", tuple(cand_ids))
        name_map = {r[0]: r[1] for r in cur.fetchall()}

        return [(name_map.get(member_ids[j], str(member_ids[j])), float(sims[j])) for j in idx]
    finally:
        conn.close()
