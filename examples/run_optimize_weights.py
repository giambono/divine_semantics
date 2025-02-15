import os
import pandas as pd
from sentence_transformers import SentenceTransformer

import config
from src.optimize_weights import optimize_weights

author_name_ids = {"dante": 1, "singleton": 2, "musa": 3, "kirkpatrick": 4, "durling": 5}


if __name__ == "__main__":
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
