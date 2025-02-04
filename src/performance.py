import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import wraps

# === PERFORMANCE EVALUATION DECORATOR ===
def evaluate_performance(func):
    """
    Decorator to evaluate embedding performance after computing embeddings.

    Parameters:
    - func: function that computes embeddings and returns a DataFrame.

    Returns:
    - Modified DataFrame with embeddings.
    - Dictionary of model performance metrics.
    """
    @wraps(func)
    def wrapper(df, *args, test_queries=None, ground_truth=None, **kwargs):
        df = func(df, *args, **kwargs)  # Compute embeddings using the original function

        performance_results = {}
        if test_queries and ground_truth:
            models = kwargs.get("models", {})  # Get models from function arguments
            print("\nStarting performance evaluation...")

            for model_name in models.keys():
                print(f"Evaluating {model_name}...")

                scores = {method: [] for method in ["avg", "concat", "weighted"] if f"embedding_{model_name}_{method}" in df.columns}

                for query_name, query_text in test_queries.items():
                    query_embedding = models[model_name].encode(query_text).reshape(1, -1)

                    for method in scores.keys():
                        embeddings_matrix = np.vstack(df[f"embedding_{model_name}_{method}"].values)
                        similarities = cosine_similarity(query_embedding, embeddings_matrix).flatten()
                        top_match = np.argmax(similarities)
                        expected_index = ground_truth[query_name]

                        # Simple accuracy metric (1 if correct, 0 otherwise)
                        score = 1 if top_match == expected_index else 0
                        scores[method].append(score)

                # Compute mean accuracy per method
                performance_results[model_name] = {method: np.mean(scores[method]) for method in scores}

        return df, performance_results

    return wrapper