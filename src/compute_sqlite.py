import os
import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

import config


import sqlite3
import pandas as pd


def fetch_cantica_data(cantica_id=None, canto=None, start_verse=None, end_verse=None):
    """Fetches 'cantica_id', 'canto', 'start_verse', and 'end_verse' from the SQLite database table `divine_comedy`,
    filtered by the given parameters.

    Parameters:
        cantica_id (int or None): Filter by cantica ID.
        canto (int or None): Filter by canto number.
        start_verse (int or None): Filter by start verse.
        end_verse (int or None): Filter by end verse.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered results.
    """
    conn = sqlite3.connect(config.DB_PATH)

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
        query += " AND dc.cantica_id = ?"
        params.append(cantica_id)
    if canto is not None:
        query += " AND dc.canto = ?"
        params.append(canto)
    if start_verse is not None:
        query += " AND dc.start_verse = ?"
        params.append(start_verse)
    if end_verse is not None:
        query += " AND dc.end_verse = ?"
        params.append(end_verse)

    query += " AND dc.author_id = 1"
    query += " AND dc.type_id = 1"

    # Execute the query
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    return df


def fetch_data_from_db(authors, types):
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


def compute_embeddings(authors, types, models=None):
    """
    Computes embeddings using different strategies from data stored in the database, filtering by author names and type strings.

    Parameters:
    authors (list): List of author names to filter the data.
    types (list): List of type names to filter the data.
    models (dict, optional): A dictionary of models to use for embeddings.

    Returns:
    pd.DataFrame: DataFrame with separate embeddings for each model.
    """
    df = fetch_data_from_db(authors, types)

    # Load default models if not provided
    if models is None:
        models = {name: SentenceTransformer(path) for name, path in config.MODELS.items()}

    # Compute embeddings
    for model_name, model in models.items():
        print(f"Computing embeddings with {model_name}...")

        # Compute embeddings per row
        df[f"embedding_{model_name}"] = df["text"].apply(
            lambda text: model.encode(text) if pd.notnull(text) else np.zeros(model.get_sentence_embedding_dimension())
        )

    return df

def weighted_avg_embedding(model_name, df, author_weights):
    """
    Compute the weighted average embedding per verse segment.

    :param df: DataFrame containing columns ['cantica_id', 'canto', 'start_verse', 'end_verse', 'author_id', 'embedding']
                output of compute_embeddings(AUTHORS, TYPES, models=MODELS)
    :param author_weights: Dictionary mapping author_id to weight (e.g., {1: 0.5, 2: 0.3, 3: 0.2}).
    :return: DataFrame with weighted average embeddings per verse segment.
    """
    df["weight"] = df["author_id"].map(author_weights)

    # Convert embedding column from list to NumPy arrays
    df[f"embedding_{model_name}"] = df[f"embedding_{model_name}"].apply(lambda x: np.array(x))

    # Compute weighted sum of embeddings per verse segment
    weighted_embeddings = df.groupby(["cantica_id", "canto", "start_verse", "end_verse"]).apply(
        lambda g: np.sum(np.stack(g[f"embedding_{model_name}"]) * g["weight"].values[:, np.newaxis], axis=0) / g["weight"].sum()
    ).reset_index(name=f"weighted_embedding_{model_name}")

    return weighted_embeddings

if __name__ == "__main__":
    from fake import FakeModel

    AUTHORS = ["musa", "durling"]
    TYPES = ["TEXT"]
    MODELS = {"fake": FakeModel()}

    df_embeddings = compute_embeddings(AUTHORS, TYPES, models=MODELS)

    author_weights_dict = {3: 0.2, 5: 0.8}  # Example author weights
    avg_weights_df = weighted_avg_embedding("fake", df_embeddings, author_weights_dict)

    print(avg_weights_df)
