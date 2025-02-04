import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.metrics.pairwise import cosine_similarity

from src.performance import evaluate_performance


def compute_ensemble_embeddings(df, columns, models=None, weights=None, cache=None):
    """
    Computes ensemble embeddings with caching.
    """
    if cache is None:
        cache = {}

    # Load models if not provided
    if models is None:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        models = {name: SentenceTransformer(path) for name, path in config["models"].items()}

    # Cache single translation embeddings
    for model_name, model in models.items():
        for col in columns:
            key = f"embedding_{model_name}_{col}"
            if key not in cache:
                df[key] = df[col].apply(
                    lambda text: model.encode(text) if pd.notnull(text) else np.zeros(model.get_sentence_embedding_dimension())
                )
                cache[key] = df[key]

    # Compute weighted embeddings
    for model_name in models:
        weighted_embedding_key = f"embedding_{model_name}_weighted"
        df[weighted_embedding_key] = df.apply(
            lambda row: np.sum(
                [
                    weights[col] * row[f"embedding_{model_name}_{col}"]
                    for col in columns if pd.notnull(row[col])
                ],
                axis=0,
            ),
            axis=1,
        )

    return df  # Return DataFrame with embeddings


def optimize_weights(df, columns, models, test_queries):
    """
    Bayesian Optimization for best embedding weights using cosine similarity.
    """
    cache = {}  # Store cached embeddings

    # Define weight search space (each weight between 0 and 1)
    space = [Real(0.0, 1.0, name=col) for col in columns]

    @use_named_args(space)
    def loss(**weights):
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {key: val / total_weight for key, val in weights.items()}

        # Compute embeddings using cached data
        df_embeddings = compute_ensemble_embeddings(df.copy(), columns, models=models, weights=normalized_weights, cache=cache)

        # Evaluate performance using cosine similarity
        _, performance_results = evaluate_performance(df_embeddings, models, test_queries)

        # Get the average accuracy across models
        avg_accuracy = np.mean([np.mean(list(performance.values())) for performance in performance_results.values()])

        return -avg_accuracy  # We minimize the negative accuracy to maximize accuracy

    # Run Bayesian Optimization
    result = gp_minimize(loss, space, n_calls=15, random_state=42)

    # Convert optimized weights to a dictionary
    best_weights = {columns[i]: result.x[i] for i in range(len(columns))}

    return best_weights


if __name__ == "__main__":

    df = pd.read_pickle("/home/rfflpllcn/IdeaProjects/divine_semantics/out/ensemble_embeddings.pkl")
    df = df[['volume', 'canto', 'verse', 'dante', 'singleton', 'musa', 'kirkpatrick', 'durling']]

    test_queries = pd.read_pickle("/home/rfflpllcn/IdeaProjects/divine_semantics/out/test_set.pkl")
    test_queries = test_queries[["query", "expected_index"]]
    test_queries = dict(zip(test_queries.iloc[:, 0], test_queries.iloc[:, 1]))

    models={"multilingual_e5": SentenceTransformer("intfloat/multilingual-e5-large")}

    best_weights = optimize_weights(df, ["dante", "singleton", "musa", "kirkpatrick", "durling"], models, test_queries)

    print("Best Weights Found:", best_weights)