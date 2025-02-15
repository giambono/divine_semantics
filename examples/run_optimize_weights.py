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
from src.experiment import embeddings_exist
from src.performance import evaluate_performance
from src.optimize_weights import optimize_weights

author_name_ids = {"dante": 1, "singleton": 2, "musa": 3, "kirkpatrick": 4, "durling": 5}


if __name__ == "__main__":
    import os
    import config

    model_key = "multilingual_e5"
    models = {"multilingual_e5": SentenceTransformer("intfloat/multilingual-e5-large")}

    embedding_path = os.path.join(config.EXPERIMENTS_ROOT, f"/embeddings/{model_key}/embeddings.parquet")
    df = pd.read_parquet(embedding_path)
    df = df[(df["cantica_id"] == 1) & (df["type_id"] == 1)]

    test_queries = pd.read_pickle(os.path.join(config.ROOT, "out/test_set.pkl"))
    test_queries = test_queries[["query", "expected_index"]]
    test_queries = dict(zip(test_queries.iloc[:, 0], test_queries.iloc[:, 1]))

    best_weights = optimize_weights(df, ["dante", "musa", "kirkpatrick", "durling"], models, test_queries)

    print("Best Weights Found:", best_weights)
