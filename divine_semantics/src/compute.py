import pandas as pd
import numpy as np

from divine_semantics.src.db_helper import fetch_data_from_db


def compute_embeddings(authors, types, models):
    """
    Computes embeddings using different strategies from data stored in the database, filtering by author names and type strings.

    Parameters:
    authors (list): List of author names to filter the data.
    types (list): List of type names to filter the data.
    models (dict): A dictionary of models to use for embeddings.

    Returns:
    pd.DataFrame: DataFrame with separate embeddings for each model.
    """
    types = [types] if isinstance(types, str) else types
    df = fetch_data_from_db(authors, types)

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

    # Convert embedding column from string/list to NumPy arrays
    df[f"embedding_{model_name}"] = df[f"embedding_{model_name}"].apply(lambda x: np.array(x))

    # Compute weighted sum of embeddings per verse segment
    weighted_embeddings = df.groupby(["cantica_id", "canto", "start_verse", "end_verse"]).apply(
        lambda g: np.sum(np.stack(g[f"embedding_{model_name}"]) * g["weight"].values[:, np.newaxis], axis=0) / g["weight"].sum()
    ).reset_index(name=f"weighted_embedding_{model_name}")

    return weighted_embeddings
