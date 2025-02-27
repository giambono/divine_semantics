import json
import numpy as np
import pandas as pd
import uuid

from qdrant_client import QdrantClient
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

    existing_collections = [c.name for c in qdrant_client.get_collections().collections]

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

        if collection_name in existing_collections:
            # raise Exception(f"Collection {collection_name} already exists.")
            print(f"Collection {collection_name} already exists. Skipping {model_name}.")
            continue

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



def weighted_avg_embedding_qdrant(
        model_name,
        collection_name,
        qdrant_client,
        author_weights,
        batch_size=1000,
        type_id=1,
        store_in_qdrant=False,
        upsert_batch_size=500,
        collection_name_weighted=None
):
    """
    Compute the weighted average embedding per verse segment from a Qdrant collection
    and optionally store the results back in the same collection.

    Parameters:
    - model_name (str): The model name to filter points (the payload's "model" field must match).
    - collection_name (str): The name of the Qdrant collection to query.
    - qdrant_client: An initialized QdrantClient instance.
    - author_weights (dict): A dictionary mapping author_id to weight (e.g., {1: 0.5, 2: 0.3, 3: 0.2}).
    - batch_size (int): Number of points to retrieve per batch.
    - type_id (int): The ID of table "type" to filter points for. Defaults to 1 (TEXT).
    - store_in_qdrant (bool): If True, stores the computed embeddings back into the collection.
    - upsert_batch_size (int): Number of points per upsert batch (to prevent timeout issues).

    Returns:
    - A Pandas DataFrame with columns:
      ['cantica_id', 'canto', 'start_verse', 'end_verse', f'weighted_embedding_{model_name}']
    """

    collection_name_weighted = f"{collection_name}_weighted" if collection_name_weighted is None else collection_name_weighted
    # Serialize global author weights for consistency
    sorted_author_weights = json.dumps(dict(sorted(author_weights.items())))

    all_points = []
    offset = None  # Qdrant scroll API uses None as the initial offset

    # Retrieve all points from the collection in batches
    while True:
        scroll_result, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=True
        )
        if not scroll_result:
            break

        all_points.extend(scroll_result)
        offset = next_offset  # Use next_offset for pagination
        if next_offset is None:
            break

    # Filter points by payload["model"] == model_name
    filtered_points = [
        point for point in all_points
        if point.payload.get("model") == model_name and
           point.payload.get("type_id") == type_id
    ]

    if not filtered_points:
        raise Exception(f"No points found in collection '{collection_name}' with model '{model_name}'.")

    # Define the weighted embedding model name
    weighted_avg_model_name = f"weighted_embedding_{model_name}"

    for point in all_points:
        if (point.payload.get("model") == weighted_avg_model_name and
                point.payload.get("applied_author_weights") == sorted_author_weights):
            raise ValueError("Weighted average embedding with the given author weights already exists.")

    # Group points by verse segment keys: (cantica_id, canto, start_verse, end_verse)
    groups = {}
    for point in filtered_points:
        payload = point.payload
        cantica_id = payload.get("cantica_id")
        canto = payload.get("canto")
        start_verse = payload.get("start_verse")
        end_verse = payload.get("end_verse")
        author_id = payload.get("author_id")

        # Look up the weight for this author; skip if not provided
        weight = author_weights.get(author_id)
        if weight is None:
            continue

        group_key = (cantica_id, canto, start_verse, end_verse)
        if group_key not in groups:
            groups[group_key] = {"embeddings": [], "weights": [], "author_ids": [], "cumulative_indices": []}
        groups[group_key]["embeddings"].append(np.array(point.vector))
        groups[group_key]["weights"].append(weight)
        groups[group_key]["author_ids"].append(author_id)
        groups[group_key]["cumulative_indices"] = payload.get("cumulative_indices")

    # Compute weighted average embedding per group
    results = []
    upsert_points = []

    for group_key, data in groups.items():
        embeddings = np.stack(data["embeddings"])
        weights = np.array(data["weights"])
        weighted_avg = np.average(embeddings, axis=0, weights=weights).tolist()

        point_payload = {
            "cantica_id": group_key[0],
            "canto": group_key[1],
            "start_verse": group_key[2],
            "end_verse": group_key[3],
            "model": weighted_avg_model_name,
            "type_id": type_id,  # Keeping type_id for clarity
            "author_id": None,   # Weighted average is not tied to a specific author
            # Storing the group-specific weights and author IDs for reference
            "applied_weights": data["weights"],
            "applied_author_ids": data["author_ids"],
            # Also store the global author weights used to compute these embeddings
            "applied_author_weights": sorted_author_weights,
            "cumulative_indices": data["cumulative_indices"]
        }

        results.append({**point_payload, weighted_avg_model_name: weighted_avg})

        # Ensure unique ID generation
        unique_id = str(uuid.uuid4())

        # Prepare upsert payload if storing in Qdrant
        if store_in_qdrant:
            upsert_points.append(
                PointStruct(
                    id=unique_id,
                    vector=weighted_avg,
                    payload=point_payload
                )
            )

        # Perform batched upserts to prevent timeout issues
        if store_in_qdrant and len(upsert_points) >= upsert_batch_size:
            qdrant_client.upsert(
                collection_name=collection_name_weighted,
                points=upsert_points
            )
            upsert_points.clear()  # Clear batch after upserting

    # Final upsert for any remaining points
    if store_in_qdrant and upsert_points:
        qdrant_client.upsert(
            collection_name=collection_name_weighted,
            points=upsert_points
        )

    return pd.DataFrame(results)



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
