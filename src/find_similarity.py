import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def find_most_similar_ensemble(input_text, df, top_n=5, models=None):
    """
    Computes similarity scores using an ensemble of models.

    Parameters:
    input_text (str): The user-provided query.
    df (pd.DataFrame): The DataFrame with stored embeddings.
    top_n (int): Number of top matches to return.
    models (dict, optional): A dictionary of models to use for embeddings.
                                 If not provided, the default models from the config will be used.
                                 Format: {"model_name": SentenceTransformer(model_path), ...}

    Returns:
    pd.DataFrame: Top N most similar verses with similarity scores.
    """

    # Load default models from config if none are provided
    if models is None:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        models = {name: SentenceTransformer(path) for name, path in config["models"].items()}

    input_embeddings = {name: model.encode(f"query: {input_text}") for name, model in models.items()}

    # Compute similarity for each model
    for model_name in models.keys():
        df[f"similarity_{model_name}"] = cosine_similarity(
            [input_embeddings[model_name]], np.vstack(df[f"embedding_{model_name}"])
        )[0]

    # Compute final ensemble similarity (average of all models)
    df["similarity_ensemble"] = df[[f"similarity_{name}" for name in models.keys()]].mean(axis=1)

    # Sort by ensemble similarity
    top_matches = df.sort_values(by="similarity_ensemble", ascending=False).head(top_n)

    return top_matches["dante"]  # [["dante", "singleton", "musa", "kirkpatrick", "durling", "similarity_ensemble"]]
