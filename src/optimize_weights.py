import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import config

from src.query import evaluate_query, process_query, run_evaluation
from src.compute_embeddings import weighted_avg_embedding_qdrant
from src.utils import load_model

# ----------------- Setup Functions ----------------- #


def get_search_space(columns=["musa", "kirkpatrick", "durling"]):
    """
    Returns the search space for the embedding weights.

    Args:
        columns: list of keys for which weights will be optimized.
    """
    return [Real(0.0, 1.0, name=col) for col in columns]

# ----------------- Optimization Functions ----------------- #

def create_loss_function(space, qdrant_client, collection_name, model, embedding_dim,
                         author_name_ids, author_ids, type_ids, test_queries, model_name):
    """
    Create and return the loss function for Bayesian optimization.

    The loss function:
        1. Normalizes the input weights.
        2. Computes the weighted average embeddings from Qdrant.
        3. Evaluates performance on test queries.
        4. Returns the negative performance.
    """
    @use_named_args(space)
    def loss(**weights):
        # Normalize weights so they sum to 1.
        total_weight = sum(weights.values())
        normalized_weights = {author_name_ids[key]: val / total_weight for key, val in weights.items()}
        print("Normalized weights:", normalized_weights)

        # Create a weighted collection name.
        collection_name_weighted = f"{collection_name}_weighted"
        qdrant_client.create_collection(
            collection_name=collection_name_weighted,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

        # Compute weighted average embeddings.
        df_embeddings = weighted_avg_embedding_qdrant(
            model_name=model_name,
            collection_name=collection_name,
            qdrant_client=qdrant_client,
            author_weights=normalized_weights,
            store_in_qdrant=True,
            collection_name_weighted=collection_name_weighted
        )

        # Evaluate performance using the provided evaluation function.
        _, performance = run_evaluation(qdrant_client, collection_name, model, author_ids, type_ids, test_queries)

        # Clean up the temporary weighted collection.
        qdrant_client.delete_collection(collection_name=collection_name_weighted)

        print(f"Normalized weights: {normalized_weights}, Performance: {performance}")
        # Return negative performance for minimization.
        return -performance

    return loss

def optimize_embedding_weights(loss, space, columns=["musa", "kirkpatrick", "durling"]):
    """
    Run Bayesian optimization and return the best embedding weights.

    Args:
        loss: the loss function to minimize.
        space: the search space for optimization.
        columns: list of keys corresponding to the weights.
    """
    result = gp_minimize(loss, space, n_calls=15, random_state=42)
    best_weights = {columns[i]: result.x[i] for i in range(len(columns))}
    return best_weights

# ----------------- Main Function ----------------- #

