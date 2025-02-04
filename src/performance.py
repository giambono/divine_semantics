import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import wraps


def evaluate_performance(df, models, test_queries):

    performance_results = {}
    print("\nStarting performance evaluation...")

    for model_name in models.keys():
        print(f"Evaluating {model_name}...")

        scores = {method: [] for method in [col for col in df.columns if col.startswith("embedding_")]}

        for query_text, expected_index in test_queries.items():
            query_embedding = models[model_name].encode(query_text).reshape(1, -1)

            for method in scores.keys():
                embeddings_matrix = np.vstack(df[method].values)
                similarities = cosine_similarity(query_embedding, embeddings_matrix).flatten()
                top_match = np.argmax(similarities)

                # Simple accuracy metric (1 if correct, 0 otherwise)
                score = 1 if top_match == expected_index else 0
                scores[method].append(score)

        # Compute mean accuracy per method
        performance_results[model_name] = {method: np.mean(scores[method]) for method in scores}

    return df, performance_results
