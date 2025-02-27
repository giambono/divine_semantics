from dotenv import load_dotenv
from qdrant_client import QdrantClient

from src.utils import load_model

import os
import pandas as pd

from src.query import run_evaluation

import config

load_dotenv()

if __name__ == "__main__":
    # Define constants for filtering
    N = 1
    author_ids = [1, 2, 3, 4, 5]
    type_ids = 1

    # Load test queries and sample N rows
    path = os.path.join(config.ROOT, "data/paraphrased_verses.parquet")
    test_queries = pd.read_parquet(path)
    test_queries_sample = test_queries.sample(n=N)[["transformed_text", "expected_index"]]

    # Initialize Qdrant client and model
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    collection_name = "dante_multilingual_e5"
    model = load_model("multilingual_e5")

    out_collect, performance = run_evaluation(qdrant_client, collection_name, model, author_ids, type_ids,
                                              test_queries_sample)

    out_collect_df = pd.DataFrame(out_collect, columns=["query_text", "is_correct"])
    out_collect_df.to_csv("output.csv", index=False)
    out_collect_df.to_clipboard()
    print(f"True count: {performance * 100:.2f}%")
