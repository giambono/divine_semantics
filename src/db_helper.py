import json
import sqlite3
import mysql.connector
import pandas as pd

import config

def get_db_connection():
    """Returns a database connection based on the configured DB_TYPE."""
    if config.DB_TYPE == "mysql":
        return mysql.connector.connect(**config.MYSQL_CONFIG)
    elif config.DB_TYPE == "sqlite":
        return sqlite3.connect(config.SQLITE_DB_PATH)
    else:
        raise ValueError("Unsupported database type. Use 'mysql' or 'sqlite'.")


def fetch_data_from_db(authors, types):
    """Fetches data from the configured database (MySQL or SQLite)."""
    conn = get_db_connection()
    query_mysql = """
        SELECT dc.cantica_id, dc.canto, dc.start_verse, dc.end_verse, dc.text, 
               a.id AS author_id, t.id AS type_id
        FROM divine_comedy dc
        JOIN author a ON dc.author_id = a.id
        JOIN type t ON dc.type_id = t.id
        WHERE a.name IN ({})
        AND t.name IN ({})
    """.format(
        ','.join(['%s' if config.DB_TYPE == "mysql" else '?'] * len(authors)),
        ','.join(['%s' if config.DB_TYPE == "mysql" else '?'] * len(types))
    )

    df = pd.read_sql(query_mysql, conn, params=authors + types)
    conn.close()
    return df


def fetch_cantica_data(cantica_id=None, canto=None, start_verse=None, end_verse=None):
    """Fetches 'cantica_id', 'canto', 'start_verse', and 'end_verse' from the database table `divine_comedy`,
    filtered by the given parameters. Supports both MySQL and SQLite.

    Parameters:
        cantica_id (int or None): Filter by cantica ID.
        canto (int or None): Filter by canto number.
        start_verse (int or None): Filter by start verse.
        end_verse (int or None): Filter by end verse.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered results.
    """
    conn = get_db_connection()

    # Base query
    query = """
    SELECT dc.text
    FROM divine_comedy dc
    WHERE 1=1
    """

    # Parameters list
    params = []

    # Dynamically add filters based on provided arguments
    if cantica_id is not None:
        query += " AND dc.cantica_id = {}".format('%s' if config.DB_TYPE == "mysql" else '?')
        params.append(cantica_id)
    if canto is not None:
        query += " AND dc.canto = {}".format('%s' if config.DB_TYPE == "mysql" else '?')
        params.append(canto)
    if start_verse is not None:
        query += " AND dc.start_verse = {}".format('%s' if config.DB_TYPE == "mysql" else '?')
        params.append(start_verse)
    if end_verse is not None:
        query += " AND dc.end_verse = {}".format('%s' if config.DB_TYPE == "mysql" else '?')
        params.append(end_verse)

    query += " AND dc.author_id = 1 AND dc.type_id = 1"

    # Execute the query
    df = pd.read_sql(query, conn, params=params)
    conn.close()

    return df


def fetch_author_ids_from_db(authors):
    """Fetches author IDs from the configured database for the given author names."""
    conn = get_db_connection()
    query = """
    SELECT a.name, a.id
    FROM author a
    WHERE a.name IN ({})
    """.format(','.join(['%s' if config.DB_TYPE == "mysql" else '?'] * len(authors)))

    try:
        df = pd.read_sql(query, conn, params=authors)
    finally:
        if conn:
            conn.close()

    return {name: int(df.loc[df["name"] == name, "id"].values[0]) if name in df["name"].values else None for name in authors}


def fetch_cumulative_indices(cantica_id, canto, start_verse, end_verse):
    """Fetch cumulative indices given cantica, canto, and verse."""
    conn = get_db_connection()
    query = """
    SELECT cumulative_indices FROM verse_mappings
    WHERE cantica_id = ? AND canto = ? AND start_verse = ? and end_verse = ?
    """
    result = pd.read_sql(query, conn, params=(cantica_id, canto, start_verse, end_verse))
    conn.close()

    if result.empty:
        return None

    return json.loads(result.iloc[0]["cumulative_indices"])