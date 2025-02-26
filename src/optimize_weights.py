import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.compute import weighted_avg_embedding
from src.db_helper import get_db_connection
from src.performance import evaluate_performance

author_name_ids = {"dante": 1, "singleton": 2, "musa": 3, "kirkpatrick": 4, "durling": 5}


def optimize_weights(df, columns, models, test_queries):
    """
    Bayesian Optimization for best embedding weights using cosine similarity.
    """

    # Define weight search space (each weight between 0 and 1)
    space = [Real(0.0, 1.0, name=col) for col in columns]

    @use_named_args(space)
    def loss(**weights):
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {author_name_ids[key]: val / total_weight for key, val in weights.items()}

        # Compute average embeddings
        model_key = list(models.keys())[0]
        df_embeddings = weighted_avg_embedding(model_key, df.copy(), normalized_weights)
        df_embeddings.set_index(["cantica_id", "canto", "start_verse", "end_verse"], inplace=True)

        # Evaluate performance using cosine similarity
        _, performance_results, correct_queries, incorrect_queries = evaluate_performance(df_embeddings, models, test_queries)

        # Get the average accuracy across models
        avg_accuracy = np.mean([np.mean(list(performance.values())) for performance in performance_results.values()])
        # print(f"Iteration Incorrect Queries: {incorrect_queries}")

        return -avg_accuracy  # We minimize the negative accuracy to maximize accuracy

    # Run Bayesian Optimization
    result = gp_minimize(loss, space, n_calls=15, random_state=42)

    # Convert optimized weights to a dictionary
    best_weights = {columns[i]: result.x[i] for i in range(len(columns))}

    return best_weights


if __name__ == "__main__":

    path = r"/home/rfflpllcn/IdeaProjects/divine_semantics/experiments/embeddings/multilingual_e5/embeddings.parquet"
    df = pd.read_parquet(path)
    df = df[(df["cantica_id"]==1) & (df["type_id"]==1) & (df["author_id"]!=1)]  # excluding dante
    # df = df[(df["cantica_id"]==1) & (df["type_id"]==1)]  # only type = text

    path = r"/home/rfflpllcn/IdeaProjects/divine_semantics/data/paraphrased_verses.parquet"
    test_queries = pd.read_parquet(path)
    # test_queries = test_queries.iloc[:10]

    test_queries = test_queries[["transformed_text", "expected_index"]]
    test_queries = dict(zip(test_queries.iloc[:, 0], test_queries.iloc[:, 1]))

    models={"multilingual_e5": SentenceTransformer("intfloat/multilingual-e5-large")}

    best_weights = optimize_weights(df, ["musa", "kirkpatrick", "durling"], models, test_queries)

    print("Best Weights Found:", best_weights)