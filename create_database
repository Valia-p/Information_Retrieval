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
    """)
    conn.commit()


def insert_or_get_id(cursor, table, field, value):
    cursor.execute(f"SELECT id FROM {table} WHERE {field} = ?", (value,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute(f"INSERT INTO {table} ({field}) VALUES (?)", (value,))
    return cursor.lastrowid


def populate_data(conn, csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["speech", "cleaned_speech", "member_name", "political_party", "sitting_date"])
    df["sitting_date"] = pd.to_datetime(df["sitting_date"], errors="coerce")
    df = df.dropna(subset=["sitting_date"])
    df["year"] = df["sitting_date"].dt.year

    cursor = conn.cursor()

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
        print(f"[!] Error checking keyword tables: {e}")
        return False
