import numpy as np
import pandas as pd
import yaml
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from src.performance import evaluate_performance


@evaluate_performance
def compute_ensemble_embeddings(df, columns, models=None, weights=None):
    """
    Computes embeddings using different strategies: averaging, concatenation, weighted, and single translation embeddings.

    Parameters:
    df (pd.DataFrame): The DataFrame with verse translations.
    columns (list): The text columns to embed.
    models (dict, optional): A dictionary of models to use for embeddings.
    weights (dict, optional): Dictionary of weights for each translation column (for weighted embeddings).

    Returns:
    pd.DataFrame: DataFrame with separate embeddings for each model.
    """
    # Load default models if not provided
    if models is None:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        models = {name: SentenceTransformer(path) for name, path in config["models"].items()}

    # Compute embeddings
    for model_name, model in models.items():
        print(f"Computing embeddings with {model_name}...")

        # Compute single translation embeddings
        for col in columns:
            df[f"embedding_{model_name}_{col}"] = df[col].apply(
                lambda text: model.encode(text) if pd.notnull(text) else np.zeros(model.get_sentence_embedding_dimension())
            )

        # Compute aggregation methods
        df[f"embedding_{model_name}_avg"] = df.apply(
            lambda row: np.mean(
                [row[f"embedding_{model_name}_{col}"] for col in columns if pd.notnull(row[col])], axis=0
            ),
            axis=1
        )

        if weights:
            df[f"embedding_{model_name}_weighted"] = df.apply(
                lambda row: np.sum(
                    [
                        weights[col] * row[f"embedding_{model_name}_{col}"]
                        for col in columns if pd.notnull(row[col])
                    ],
                    axis=0,
                ),
                axis=1,
            )

    return df  # Only return the DataFrame, the decorator will handle evaluation


if __name__ == "__main__":
    from src.fake import FakeModel


    df = pd.read_pickle("/home/rfflpllcn/IdeaProjects/divine_semantics/out/ensemble_embeddings.pkl")

    df = df[['volume', 'canto', 'verse', 'dante', 'singleton', 'musa', 'kirkpatrick', 'durling']]

    # === USAGE EXAMPLE ===
    test_queries = {
        "midway_query": "midway upon the journey of our life",
        "dark_wood_query": "when i saw him in the desert"
    }

    ground_truth = {
        "midway_query": 0,  # Expected verse index
        "dark_wood_query": 21
    }

    weights = {
        "dante": 0.0,
        "singleton": 0.1,
        "musa": 0.3,
        "kirkpatrick": 0.3,
        "durling": 0.3,
    }

    df, scores = compute_ensemble_embeddings(df, ["singleton", "musa", "kirkpatrick", "durling"],
                                             models={"fake": FakeModel()},
                                             test_queries=test_queries, ground_truth=ground_truth, weights=weights)

    print("\nPerformance Scores:")
    print(scores)
