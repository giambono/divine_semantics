import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

from src.utils import load_model

load_dotenv()


def evaluate_query(client, collection_name, query_text, expected_index, model, author_ids, type_ids):
    author_ids = [author_ids] if isinstance(author_ids, int) else author_ids
    type_ids = [type_ids] if isinstance(type_ids, int) else type_ids

    # Compute the query embedding from your query_text
    query_embedding = model.encode(query_text)

    search_filter = Filter(
        must=[
            FieldCondition(key="type_id", match=MatchAny(any=type_ids)),
            FieldCondition(key="author_id", match=MatchAny(any=author_ids))
        ]
    )

    # Perform the search in Qdrant
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=1,  # we only care about the top match
        query_filter=search_filter
    )

    if hits:
        # Compare the returned point's id or payload index with expected_index
        top_hit = hits[0]
        predicted_index = top_hit.payload.get("index")  # adjust based on your payload structure

        print(f"Query: {query_text}")
        print(f"Expected Index: {expected_index}, Predicted Index: {predicted_index}")
        return predicted_index == expected_index
    else:
        print("No matching verses found.")
        return False


if __name__ == "__main__":
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    collection_name = "dante_multilingual_e5"
    # collection = qdrant_client.get_collection(collection_name)

    model_key = "multilingual_e5"
    model = load_model(model_key)

    query_text = "I found me in the wood"
    expected_index = 0
    evaluate_query(qdrant_client, collection_name, query_text, expected_index, model, author_ids=[1, 2, 3, 4, 5],
                   type_ids=1)
