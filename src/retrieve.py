import os
import pandas as pd
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



if __name__ == "__main__":
    authors = ["musa", "singleton", "dante", "kirkpatrick"]
    out = fetch_author_ids_from_db(authors)
    print(out)