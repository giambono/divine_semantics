import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


import yaml
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def find_most_similar_ensemble(input_text, df, models=None):
    """
    Finds the most similar verse in the 'dante' column based on the highest similarity score
    across all individual model similarities and the ensemble similarity score.

    Parameters:
    input_text (str): The user-provided query.
    df (pd.DataFrame): The DataFrame with stored embeddings.
    models (dict, optional): A dictionary of models to use for embeddings.
                             If not provided, the default models from the config will be used.
                             Format: {"model_name": SentenceTransformer(model_path), ...}

    Returns:
    str: The most similar verse from the 'dante' column.
    """

    # Load default models from config if none are provided
    if models is None:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        models = {name: SentenceTransformer(path) for name, path in config["models"].items()}

    # Compute query embeddings with "query:" prefix
    input_embeddings = {name: model.encode(f"query: {input_text}") for name, model in models.items()}

    # Ensure embeddings are NumPy arrays
    for col in ["embedding_multilingual_e5", "embedding_minilm", "embedding_contriever"]:
        df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

    # Compute similarity for each model
    similarity_columns = []
    for model_name in models.keys():
        column_name = f"similarity_{model_name}"
        df[column_name] = cosine_similarity(
            [input_embeddings[model_name]], np.vstack(df[f"embedding_{model_name}"])
        )[0]
        similarity_columns.append(column_name)

    # Compute final ensemble similarity (average of all models)
    df["similarity_ensemble"] = df[similarity_columns].mean(axis=1)
    similarity_columns.append("similarity_ensemble")  # Add ensemble score to comparison

    # Find the row with the highest value across all similarity columns
    best_match_idx = df[similarity_columns].max(axis=1).idxmax()  # Get the highest similarity index
    most_similar_verse = df.iloc[best_match_idx]["dante"]

    return most_similar_verse
