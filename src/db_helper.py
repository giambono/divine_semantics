import json
import sqlite3
import pandas as pd

import config

def get_db_connection():
    """Returns a database connection based on the configured DB_TYPE."""
    if config.DB_TYPE == "mysql":
        raise NotImplemented("support for MySQL databases not implemented")
        # return mysql.connector.connect(**config.MYSQL_CONFIG)
    elif config.DB_TYPE == "sqlite":
        return sqlite3.connect(config.SQLITE_DB_PATH)
    else:
        raise ValueError("Unsupported database type. Use 'mysql' or 'sqlite'.")


def retrieve_text(cantica, canto, start_verse, end_verse, author_names, type_name):
    """
    Retrieves text fragments from the database spanning a given verse range,
    supporting both SQLite and MySQL based on config.DB_TYPE.

    Parameters:
      - author_names: a single author name or a list of author names.
      - type_name: the type name (e.g., translation type).
      - cantica_name: the cantica (e.g., 'Inferno', 'Purgatorio', 'Paradiso').
      - canto: the canto number.
      - start_verse: the starting verse number of the desired range.
      - end_verse: the ending verse number of the desired range.

    Returns:
      A dictionary mapping each author's name to the concatenated text
      fragments that overlap the specified verse range.
    """
    # Ensure author_names is a list
    if isinstance(author_names, str):
        author_names = [author_names]

    cantica = cantica if isinstance(cantica, int) else cantica_name2id(cantica)

    conn = get_db_connection()
    cur = conn.cursor()

    # Set the placeholder style based on the DB type
    placeholder = '%s' if config.DB_TYPE == "mysql" else '?'
    placeholders_authors = ','.join([placeholder for _ in author_names])

    # SELECT a.name, d.text, d.start_verse

    # Build query: select rows that overlap with the desired verse range.
    # Using d.start_verse <= requested_end_verse and d.end_verse >= requested_start_verse
    # ensures any overlapping row is included.
    query = """
    SELECT c.id as cantica_id,
           c.name as cantica_name,
           d.canto as canto, 
           a.name as author_name, 
           a.id as author_id, 
           d.text as text, 
           d.start_verse as d_start_verse, 
           d.end_verse as d_end_verse
    FROM divine_comedy d
    JOIN author a ON d.author_id = a.id
    JOIN cantica c ON d.cantica_id = c.id
    JOIN type t ON d.type_id = t.id
    WHERE a.name IN ({authors})
      AND t.name = {ph}
      AND d.cantica_id = {ph}
      AND d.canto = {ph}
      AND d.start_verse <= {ph}
      AND d.end_verse >= {ph}
    ORDER BY d.start_verse ASC
    """.format(authors=placeholders_authors, ph=placeholder)

    # Parameters: authors first, then type, cantica, canto,
    # then the verse range (note the order: end_verse for d.start_verse and start_verse for d.end_verse)
    params = author_names + [type_name, cantica, canto, end_verse, start_verse]
    cur.execute(query, params)
    rows = cur.fetchall()

    column_names = [desc[0] for desc in cur.description]
    dict_rows = [dict(zip(column_names, row)) for row in rows]


    # Aggregate text fragments for each author
    result = {}
    for dict_row in dict_rows:
        author = dict_row['author_name']
        if author in result:
            result[author] += " " + dict_row['text']
        else:
            result[author] = dict_row['text']

    conn.close()
    return result, dict_rows


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


def fetch_cantica_data(cantica=None, canto=None, start_verse=None, end_verse=None):
    """Fetches 'cantica_id', 'canto', 'start_verse', and 'end_verse' from the database table `divine_comedy`,
    filtered by the given parameters. Supports both MySQL and SQLite.

    Parameters:
        cantica (int or None or str): Filter by cantica ID.
        canto (int or None): Filter by canto number.
        start_verse (int or None): Filter by start verse.
        end_verse (int or None): Filter by end verse.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered results.
    """
    conn = get_db_connection()

    cantica = cantica if isinstance(cantica, int) else cantica_name2id(cantica)

    # Base query
    query = """
    SELECT dc.text
    FROM divine_comedy dc
    WHERE 1=1
    """

    # Parameters list
    params = []

    # Dynamically add filters based on provided arguments
    if cantica is not None:
        query += " AND dc.cantica_id = {}".format('%s' if config.DB_TYPE == "mysql" else '?')
        params.append(cantica)
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

    return json.loads(json.loads(result.iloc[0]["cumulative_indices"]))



def cantica_id2name(cantica_id):
    conn = get_db_connection()
    cur = conn.cursor()

    cantica_id = [cantica_id] if isinstance(cantica_id, int) else cantica_id

    query = """
    SELECT a.name
    FROM cantica a
    WHERE a.id = {ph}
    """.format(ph='?')

    params = cantica_id
    cur.execute(query, params)
    return cur.fetchall()[0][0]


def cantica_name2id(cantica_name):
    conn = get_db_connection()
    cur = conn.cursor()

    cantica_name = [cantica_name] if isinstance(cantica_name, str) else cantica_name

    query = """
    SELECT a.id
    FROM cantica a
    WHERE a.name = {ph}
    """.format(ph='?')

    params = cantica_name
    cur.execute(query, params)
    return cur.fetchall()[0][0]


if __name__ == "__main__":


    # from datasets import load_dataset
    #
    # dataset = load_dataset("maiurilorenzo/divina-commedia")
    # dataset_en = load_dataset("giambono/commedia_en")
    #
    # # Display the first few entries
    # df = dataset["train"].to_pandas()
    # df_en = dataset_en["train"].to_pandas()
    #
    # cantica_id = 3
    # canto = 31
    # start_verse = 130
    # end_verse = 132
    #
    # out = fetch_cumulative_indices(cantica_id, canto, start_verse, end_verse)
    # print(out)
    #
    # data = fetch_cantica_data(cantica_id, canto, start_verse, end_verse)
    # print(data.text)
    #
    # print(df.iloc[out])

    # text = retrieve_text(["dante", "musa"], "TEXT", 1, 1, 4, 6)

    # print(text)

    params = {'cantica': 1, 'canto': 1, 'start_verse': 1, 'end_verse': 200}
    _, d_list = retrieve_text(**{**params, **{"author_names": ["dante"], "type_name": "TEXT"}})

    for d in d_list:
        # {'author_id': 1, 'author_name': 'dante', 'cantica_id': 1, 'cantica_name': 'Inferno', 'canto': 1, 'd_end_verse': 3, 'd_start_verse': 1, 'text': 'Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura, ché la diritta via era smarrita.'}
        {"cantica": d["cantica_name"], "canto": d["canto"], "start_verse": d["d_start_verse"], "end_verse": d["d_end_verse"], "tercet": d["text"]}
    print(df)
