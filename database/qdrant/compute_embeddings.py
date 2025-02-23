import numpy as np
import pandas as pd

from qdrant_client.models import PointStruct
from qdrant_client.models import Distance, VectorParams


def compute_embeddings_and_upsert(df, models, qdrant_client, collection_name_prefix="dante_"):
    """
    Computes embeddings using different models for data stored in the SQLite database,
    and upserts the computed embeddings directly into Qdrant collections.

    Parameters:
    - models (dict): Dictionary where keys are model names and values are the model instances.
    - qdrant_client (QdrantClient): An instance of QdrantClient.
    - collection_name_prefix (str): Prefix for Qdrant collection names.

    The function fetches all rows from the "divine_comedy" table.
    """

    # Loop over each model to compute embeddings and upsert to a corresponding Qdrant collection
    for model_name, model in models.items():
        print(f"Computing embeddings with {model_name}...")

        sample_embedding = model.encode("test")
        embedding_dim = len(sample_embedding)

        # Compute embeddings for each row's 'text' column
        df[f"embedding_{model_name}"] = df["text"].apply(
            lambda text: model.encode(text) if pd.notnull(text)
            else np.zeros(model.get_sentence_embedding_dimension())
        )

        # Define collection name for this model's embeddings
        collection_name = f"{collection_name_prefix}{model_name}"
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        # Build list of Qdrant points from the dataframe
        points = []
        for idx, row in df.iterrows():
            vector = row[f"embedding_{model_name}"]

            # Prepare payload with all other fields (dropping the embedding field)
            payload = row.drop(labels=[f"embedding_{model_name}", "text"]).to_dict()
            # Add new field "model" to the payload
            payload["model"] = model_name

            points.append(PointStruct(id=idx, vector=vector, payload=payload))

        batch_size = 100  # adjust this based on your system and network
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=batch,
            )
            print(f"Upserted batch {i // batch_size + 1} containing {len(batch)} points")