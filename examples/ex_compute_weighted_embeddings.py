from qdrant_client import QdrantClient

from src.compute_embeddings import weighted_avg_embedding_qdrant


if __name__ == "__main__":

    qdrant_client = QdrantClient(url="http://localhost:6333")
    # Example usage:
    weight_map = {1: 0.5, 2: 0.3, 3: 0.2}
    df_weighted = weighted_avg_embedding_qdrant(
        model_name="fake_text",
        collection_name="dante_fake_text",
        qdrant_client=qdrant_client,  # Your pre-initialized QdrantClient instance
        author_weights=weight_map,
        batch_size=1000,
        store_in_qdrant=True,
        upsert_batch_size=200
    )
    print(df_weighted)
