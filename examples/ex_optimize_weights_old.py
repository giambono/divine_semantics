import os
import pandas as pd
import numpy as np
import concurrent.futures
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from src.query import evaluate_query, process_query, run_evaluation
from src.compute_embeddings import weighted_avg_embedding_qdrant
from src.utils import load_model
import config

load_dotenv()


# Setup Qdrant client, model, and test queries.
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
collection_name = "dante_multilingual_e5"
model_name = "multilingual_e5"
model = load_model(model_name)

sample_embedding = model.encode("test")
embedding_dim = len(sample_embedding)

# Load test queries (only the required columns)
path = os.path.join(config.ROOT, "data/paraphrased_verses.parquet")
test_queries = pd.read_parquet(path)[["transformed_text", "expected_index"]]
test_queries = test_queries.iloc[:2]

# Fixed evaluation parameters.
author_name_ids = {"dante": 1, "singleton": 2, "musa": 3, "kirkpatrick": 4, "durling": 5}
author_ids = [1, 2, 3, 4, 5]
type_ids = 1

# Define the search space for embedding weights.
# 'columns' here represent the embedding weight keys that you want to optimize.
columns = ["musa", "kirkpatrick", "durling"]  # adjust as needed
space = [Real(0.0, 1.0, name=col) for col in columns]

@use_named_args(space)
def loss(**weights):
    """
    Loss function for Bayesian optimization.
    1. Normalizes the input weights.
    2. Computes the average embeddings from the Qdrant DB using these weights.
    3. Evaluates performance (via cosine similarity or a similar metric) on test queries.
    4. Returns the negative performance (so that maximizing performance minimizes loss).
    """
    # Normalize weights so they sum to 1.
    total_weight = sum(weights.values())
    normalized_weights = {author_name_ids[key]: val / total_weight for key, val in weights.items()}
    print("normalized_weights: ", normalized_weights)

    # Compute the average embeddings based on normalized weights.
    # This function should query your Qdrant DB and return a DataFrame of embeddings.
    collection_name_weighted = f"{collection_name}_weighted"

    qdrant_client.create_collection(
        collection_name=collection_name_weighted,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    df_embeddings = weighted_avg_embedding_qdrant(model_name=model_name,
                                                  collection_name=collection_name,
                                                  qdrant_client=qdrant_client,
                                                  author_weights=normalized_weights,
                                                  store_in_qdrant=True,
                                                  collection_name_weighted=collection_name_weighted
                                                  )

    # Evaluate performance using our concurrent evaluation function.
    # (Optionally, df_embeddings could be used inside evaluate_performance if needed.)
    _, performance = run_evaluation(qdrant_client, collection_name, model, author_ids, type_ids, test_queries)

    qdrant_client.delete_collection(collection_name=collection_name_weighted)

    # For debugging, print the normalized weights and the achieved performance.
    print(f"Normalized weights: {normalized_weights}, Performance: {performance}")

    # We return the negative performance since gp_minimize minimizes the loss.
    return -performance

def optimize_embedding_weights():
    """
    Run Bayesian optimization to find the best embedding weights.
    """
    result = gp_minimize(loss, space, n_calls=15, random_state=42)
    best_weights = {columns[i]: result.x[i] for i in range(len(columns))}
    return best_weights


if __name__ == "__main__":
    best_weights = optimize_embedding_weights()
    print("Optimized embedding weights:", best_weights)
