import ast
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from functools import wraps

from src.db_helper import get_db_connection


def evaluate_performance(df, models, test_queries):

    conn = get_db_connection()
    query = """
        SELECT * FROM verse_mappings
        """
    verse_mappings = pd.read_sql(query, conn)
    conn.close()


    performance_results = {}
    print("\nStarting performance evaluation...")

    model_key = list(models.keys())[0]
    scores = {method: [] for method in [col for col in df.columns if col.startswith("weighted_embedding_")]}

    correct_queries = []
    incorrect_queries = []
    for query_text, expected_index in test_queries.items():
        query_embedding = models[model_key].encode(query_text).reshape(1, -1)

        for method in scores.keys():
            embeddings_matrix = np.vstack(df[method].values)
            # print(embeddings_matrix)
            similarities = cosine_similarity(query_embedding, embeddings_matrix).flatten()
            top_match = np.argmax(similarities)

            _d = dict(zip(["cantica_id", "canto", "start_verse", "end_verse"], df.index[top_match]))

            _value = verse_mappings.iloc[top_match].cumulative_indices
            if _value.startswith('"') and _value.endswith('"'):
                _value = _value[1:-1]  # Strip outer quotes

            _indices = json.loads(_value)

            # Simple accuracy metric (1 if correct, 0 otherwise)
            # score = 1 if top_match == expected_index else 0
            score = 1 if expected_index in _indices else 0
            scores[method].append(score)

            if score == 1:
                correct_queries.append(query_text)
            else:
                incorrect_queries.append(query_text)

    # Compute mean accuracy per method
    performance_results[model_key] = {method: np.mean(scores[method]) for method in scores}

    return df, performance_results, correct_queries, incorrect_queries

