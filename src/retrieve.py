import os
import pandas as pd
import sqlite3
import mysql.connector
from dotenv import load_dotenv


load_dotenv()

# Database connection configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

DB_PATH = os.getenv("DB_PATH", "divine_comedy.db")

def fetch_author_ids_from_db(authors):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        query = f"""
        SELECT a.name, a.id
        FROM author a
        WHERE a.name IN ({','.join(['%s'] * len(authors))})
        """
        df = pd.read_sql(query, conn, params=authors)
    finally:
        if conn:
            conn.close()

    # Ensure correct order by mapping using original authors list
    return {name: int(df.loc[df["name"] == name, "id"].values[0]) if name in df["name"].values else None for name in authors}


def fetch_author_ids_from_db_sqlite(authors):
    """Fetches author IDs from the SQLite database for the given author names."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT a.name, a.id
        FROM author a
        WHERE a.name IN ({})
        """.format(','.join(['?'] * len(authors)))
        df = pd.read_sql_query(query, conn, params=authors)
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    authors = ["musa", "singleton", "dante", "kirkpatrick"]
    out = fetch_author_ids_from_db(authors)
    print(out)