import yaml
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

import config


def find_most_similar_ensemble_old(input_text, df, models=None):
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
        models = {name: SentenceTransformer(path) for name, path in config.MODELS.items()}

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

    # Determine which similarity column achieved the highest score for this row
    best_similarity_series = df.loc[best_match_idx, similarity_columns]
    best_similarity_column = best_similarity_series.idxmax()
    # Remove the prefix to get the model name; if ensemble wins, return 'ensemble'
    if best_similarity_column == "similarity_ensemble":
        best_model = "ensemble"
    else:
        best_model = best_similarity_column.replace("similarity_", "")

    return most_similar_verse, best_model


def find_most_similar_ensemble(input_text, df,
                               models= {"multilingual_e5": SentenceTransformer("intfloat/multilingual-e5-large")},
                               nlargest=5):
    """
    Finds the most similar verse in the 'dante' column based on the highest similarity score
    across all individual model similarities and the ensemble similarity score.
    Also returns the model (or 'ensemble') that achieved this highest similarity.

    Parameters:
    input_text (str): The user-provided query.
    df (pd.DataFrame): The DataFrame with stored embeddings.
    models (dict, optional): A dictionary of models to use for embeddings.
                             If not provided, the default models from the config will be used.
                             For "facebook/contriever", the entry will be a dict with keys "model" and "tokenizer".
                             For other models, the value is a SentenceTransformer instance.

    Returns:
    tuple: (most_similar_verse, best_model) where:
           - most_similar_verse (str): The most similar verse from the 'dante' column.
           - best_model (str): The model name that achieved the highest similarity, or "ensemble" if the ensemble score was highest.
    """
    # Load default models from config if none are provided
    if models is None:
        models = {name: SentenceTransformer(path) for name, path in config.MODELS.items()}

    # Compute query embeddings with "query:" prefix for each model
    input_embeddings = {}
    for model_name, model_obj in models.items():
        input_embeddings[model_name] = model_obj.encode(f"query: {input_text}")

    # Ensure stored embeddings are numpy arrays
    for col in [f"weighted_embedding_{key}" for key in models.keys()]:
        df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

    # Compute similarity for each model and store in new columns
    similarity_columns = []
    for model_name in models.keys():
        col_name = f"similarity_{model_name}"
        embeddings = np.vstack(df[f"weighted_embedding_{model_name}"])
        df[col_name] = cosine_similarity(
            [input_embeddings[model_name]], embeddings
        )[0]
        similarity_columns.append(col_name)

    topn = df[[col for col in df.columns if col.startswith("similarity")]].stack().nlargest(nlargest)
    topn_indices = topn.index.get_level_values(0)

    return df.loc[topn_indices]


    # # Identify the best match row (the one with the maximum similarity across any column)
    # best_match_idx = df[similarity_columns].max(axis=1).idxmax()
    # most_similar_verse = df.loc[best_match_idx, "dante"]
