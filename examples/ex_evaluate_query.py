from dotenv import load_dotenv
from qdrant_client import QdrantClient

from src.query import evaluate_query
from src.utils import load_model

load_dotenv()

if __name__ == "__main__":
    import os
    import pandas as pd

    import config

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    collection_name = "dante_multilingual_e5"

    model_key = "multilingual_e5"
    model = load_model(model_key)

    path = os.path.join(config.ROOT, "data/paraphrased_verses.parquet")
    test_queries = pd.read_parquet(path)

    N = 2
    test_queries_sample = test_queries.sample(n=N)[["transformed_text", "expected_index"]]

    # Iterate over the sampled rows and collect the evaluation output
    out_collect = []
    for _, row in test_queries_sample.iterrows():
        query_text = row["transformed_text"]
        expected_index = row["expected_index"]
        result = evaluate_query(
            qdrant_client,
            collection_name,
            query_text,
            expected_index,
            model,
            author_ids=[1, 2, 3, 4, 5],
            type_ids=1
        )
        out_collect.append([query_text, result])

    # Calculate performance based on the count of True results
    true_count = sum(flag for _, flag in out_collect)
    performance = true_count / len(out_collect)

    print(f"True count: {performance * 100:.2f}%")