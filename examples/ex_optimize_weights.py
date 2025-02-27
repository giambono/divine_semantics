import os
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient

import config
from src.optimize_weights import get_search_space, create_loss_function, optimize_embedding_weights
from src.utils import load_model, setup_environment, initialize_qdrant_client, initialize_model, load_test_queries


def get_fixed_parameters():
    """
    Returns fixed evaluation parameters.
    """
    author_name_ids = {"dante": 1, "singleton": 2, "musa": 3, "kirkpatrick": 4, "durling": 5}
    author_ids = [1, 2, 3, 4, 5]
    type_ids = 1
    return author_name_ids, author_ids, type_ids


def main():
    # Setup environment and clients.
    setup_environment()
    qdrant_client = initialize_qdrant_client()

    # Initialize model and determine embedding dimension.
    model_name = "multilingual_e5"
    model, embedding_dim = initialize_model(model_name)

    # Define collection name and load test queries.
    collection_name = "dante_multilingual_e5"
    test_queries_path = os.path.join(config.ROOT, "data/paraphrased_verses.parquet")
    test_queries = load_test_queries(test_queries_path, n=2)

    # Get fixed evaluation parameters.
    author_name_ids, author_ids, type_ids = get_fixed_parameters()

    # Setup search space and loss function.
    columns = ["musa", "kirkpatrick", "durling"]
    space = get_search_space(columns)
    loss = create_loss_function(space, qdrant_client, collection_name, model, embedding_dim,
                                author_name_ids, author_ids, type_ids, test_queries, model_name)

    # Optimize and print the best weights.
    best_weights = optimize_embedding_weights(loss, space, columns)
    print("Optimized embedding weights:", best_weights)


if __name__ == "__main__":
    main()
