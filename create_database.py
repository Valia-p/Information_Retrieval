import pandas as pd
import sqlite3

DB_NAME = "parliament.db"
CSV_FILE = "cleaned_data.csv"


def create_schema(conn):
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS members (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT UNIQUE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS parties (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS speeches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER NOT NULL,
        member_id INTEGER NOT NULL,
        party_id INTEGER NOT NULL,
        sitting_date TEXT NOT NULL,
        year INTEGER NOT NULL,
        speech TEXT NOT NULL,
        cleaned_speech TEXT NOT NULL,
        FOREIGN KEY (member_id) REFERENCES members(id),
        FOREIGN KEY (party_id) REFERENCES parties(id)
    );

    CREATE TABLE IF NOT EXISTS speech_keywords (
        speech_id INTEGER,
        keyword TEXT,
        score REAL,
        PRIMARY KEY (speech_id, keyword),
        FOREIGN KEY (speech_id) REFERENCES speeches(id)
    );

    CREATE TABLE IF NOT EXISTS member_keywords_by_year (
        member_id INTEGER,
        year INTEGER,
        keyword TEXT,
        score REAL,
        PRIMARY KEY (member_id, year, keyword),
        FOREIGN KEY (member_id) REFERENCES members(id)
    );

    CREATE TABLE IF NOT EXISTS party_keywords_by_year (
        party_id INTEGER,
        year INTEGER,
        keyword TEXT,
        score REAL,
        PRIMARY KEY (party_id, year, keyword),
        FOREIGN KEY (party_id) REFERENCES parties(id)
    );
    
    CREATE TABLE IF NOT EXISTS member_similarity_pairs (
        member1_id INTEGER NOT NULL,
        member2_id INTEGER NOT NULL,
        score REAL NOT NULL,
        PRIMARY KEY (member1_id, member2_id),  -- ΜΟΝΑΔΙΚΟ ΖΕΥΓΟΣ
        FOREIGN KEY (member1_id) REFERENCES members(id),
        FOREIGN KEY (member2_id) REFERENCES members(id)
    );
    """)
    conn.commit()


def insert_or_get_id(cursor, table, field, value):
    """
       Utility: Insert a value into a table if it does not exist,
       and return its primary key id.
    """
    cursor.execute(f"SELECT id FROM {table} WHERE {field} = ?", (value,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute(f"INSERT INTO {table} ({field}) VALUES (?)", (value,))
    return cursor.lastrowid


def populate_data(conn, csv_path):
    """
        Populate the database from a cleaned CSV dataset.
        Steps:
          - Read CSV
          - Drop rows with missing essential fields
          - Parse sitting_date to datetime
          - Add derived 'year' column
          - Insert members, parties, speeches into DB
    """
    df = pd.read_csv(csv_path)

    # Drop rows with missing speech, cleaned speech, member, party, or date
    df = df.dropna(subset=["speech", "cleaned_speech", "member_name", "political_party", "sitting_date"])
    # Parse dates
    df["sitting_date"] = pd.to_datetime(df["sitting_date"], errors="coerce")
    df = df.dropna(subset=["sitting_date"])
    # Add year column
    df["year"] = df["sitting_date"].dt.year

    cursor = conn.cursor()

    # Insert speeches and link them to members and parties
    for _, row in df.iterrows():
        member_id = insert_or_get_id(cursor, "members", "full_name", row["member_name"])
        party_id = insert_or_get_id(cursor, "parties", "name", row["political_party"])

        cursor.execute("""
            INSERT INTO speeches (doc_id, member_id, party_id, sitting_date, year, speech, cleaned_speech)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row["document_id"]),
            member_id,
            party_id,
            row["sitting_date"].strftime("%Y-%m-%d"),
            int(row["year"]),
            row["speech"],
            row["cleaned_speech"]
        ))

    conn.commit()
    print("Data insertion successful")


def is_part2_already_computed():
    """
        Check whether Part2 preprocessing (keywords extraction) has already been computed.
        Returns:
            True if the following tables contain data:
              - speech_keywords
              - member_keywords_by_year
              - party_keywords_by_year
            False otherwise.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM speech_keywords")
        if cursor.fetchone()[0] == 0:
            return False

        cursor.execute("SELECT COUNT(*) FROM member_keywords_by_year")
        if cursor.fetchone()[0] == 0:
            return False

        cursor.execute("SELECT COUNT(*) FROM party_keywords_by_year")
        if cursor.fetchone()[0] == 0:
            return False

        conn.close()
        return True

    except Exception as e:
        print(f"Error checking keyword tables: {e}")
        return False

def is_part3_already_computed() -> bool:
    """
       Check if member_similarity_pairs table exists AND has at least one row.
       Returns:
           True if similarities are already computed, else False
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
