import mysql.connector
import os
import sqlite3
import pandas as pd
import numpy as np
import mysql.connector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import config

load_dotenv()
# Database connection configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}


def clean_dataframe(df):
    """
    Cleans specified columns in a DataFrame by removing commas, dots, and newlines.

    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    columns (list): List of column names to clean.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    columns = [col for col in df.columns.to_list() if col != 'verse']
    df_cleaned = df.copy()
    df_cleaned[columns] = (
        df_cleaned[columns]
        .replace({",": "", "\.": "", "\n": " "}, regex=True)  # Remove unwanted characters
        .apply(lambda x: x.str.lower())  # Convert to lowercase
    )
    return df_cleaned


def get_non_zero_keys(dictionary):
    """Return list of keys with values greater than 0."""
    return [key for key, value in dictionary.items() if value > 0]


def fetch_data_from_db_mysql(authors, types):
    """Fetches data from the MySQL database table `divine_comedy` filtered by author names and type strings."""

    conn = mysql.connector.connect(**DB_CONFIG)
    query = f"""
    SELECT dc.cantica_id, dc.canto, dc.start_verse, dc.end_verse, dc.text, a.id AS author_id, t.id AS type_id
    FROM divine_comedy dc
    JOIN author a ON dc.author_id = a.id
    JOIN type t ON dc.type_id = t.id
    WHERE a.name IN ({','.join(['%s'] * len(authors))})
    AND t.name IN ({','.join(['%s'] * len(types))})
    """
    df = pd.read_sql(query, conn, params=authors + types)
    conn.close()
    return df

def fetch_data_from_db_sqlite(authors, types):
    """Fetches data from the SQLite database table `divine_comedy` filtered by author names and type strings."""
    conn = sqlite3.connect(config.DB_PATH)
    query = """
    SELECT dc.cantica_id, dc.canto, dc.start_verse, dc.end_verse, dc.text, a.id AS author_id, t.id AS type_id
    FROM divine_comedy dc
    JOIN author a ON dc.author_id = a.id
    JOIN type t ON dc.type_id = t.id
    WHERE a.name IN ({})
    AND t.name IN ({})
    """.format(
        ','.join(['?'] * len(authors)),
        ','.join(['?'] * len(types))
    )
    df = pd.read_sql_query(query, conn, params=authors + types)
    conn.close()
    return df
