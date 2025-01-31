import yaml
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def compute_ensemble_embeddings(df, columns, models=None):
    """
    Computes embeddings for each model and stores them in separate columns.

    Parameters:
    df (pd.DataFrame): The DataFrame with verse translations.
    columns (list): The text columns to embed.
    models (dict, optional): A dictionary of models to use for embeddings.
                             If not provided, the default models from the config will be used.
                             Format: {"model_name": SentenceTransformer(model_path), ...}

    Returns:
    pd.DataFrame: DataFrame with separate embeddings for each model.
    """
    # Load default models from config if none are provided
    if models is None:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        models = {name: SentenceTransformer(path) for name, path in config["models"].items()}

    # Compute embeddings for each model
    for model_name, model in models.items():
        print(f"Computing embeddings with {model_name}...")
        df[f"embedding_{model_name}"] = df.apply(
            lambda row: np.mean([model.encode(row[col]) for col in columns if pd.notnull(row[col])], axis=0),
            axis=1
        )
    return df