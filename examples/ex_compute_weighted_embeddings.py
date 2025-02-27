from qdrant_client import QdrantClient

from src.compute_embeddings import weighted_avg_embedding_qdrant
from src.utils import load_model


if __name__ == "__main__":

    # collection_name = "dante_multilingual_e5"
    collection_name = "dante_fake_text"
    model_name = "fake_text"
    model = load_model(model_name)

    qdrant_client = QdrantClient(url="http://localhost:6333")
    # Example usage:
    weight_map = {3: 0.45266643204217655, 4: 0.10424393059867922, 5: 0.44308963735914425}
    # {'musa': 0.45266643204217655, 'kirkpatrick': 0.10424393059867922, 'durling': 0.44308963735914425}

    df_weighted = weighted_avg_embedding_qdrant(
        model_name=model_name,
        collection_name=collection_name,
        qdrant_client=qdrant_client,  # Your pre-initialized QdrantClient instance
        author_weights=weight_map,
        batch_size=1000,
        store_in_qdrant=True,
        upsert_batch_size=200,
        collection_name_weighted=f"{collection_name}_optim_weights"
    )
    print(df_weighted)
