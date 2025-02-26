"""
sudo docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

"""
from qdrant_client import QdrantClient

from src.db_helper import retrieve_text, cantica_id2name
from src.fake import FakeModel


def query_and_retrieve(model, collection_name, qdrant_client, query_text):
    """
    Queries a Qdrant collection for a point matching the encoded query_text using the given model,
    verifies that the payload contains a 'model' field, processes the payload, and retrieves text.

    Parameters:
    - model: The model instance that provides an encode method.
    - collection_name (str): The name of the Qdrant collection.
    - qdrant_client (QdrantClient): An already initialized QdrantClient instance.
    - query_text (str): The text to query with.

    Returns:
    - The response from retrieve_text(**payload)

    Raises:
    - Exception: if the collection is not found, no points are returned, or the payload is missing a 'model' field.
    """
    # Check if the collection exists
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in existing_collections:
        raise Exception(f"Collection {collection_name} not found")

    # Query Qdrant using the encoded query text from the model
    search_points = qdrant_client.query_points(
        collection_name=collection_name,
        query=model.encode(query_text),
        with_payload=True,
        limit=1
    ).points

    if not search_points:
        raise Exception("No points found for the query")

    search_result = search_points[0]
    payload = search_result.payload

    # Verify that the payload contains the "model" field
    if "model" not in payload:
        raise Exception("Model not found in the payload")

    # Remove the "model" field from the payload
    payload.pop("model")

    # Remove unwanted fields from the payload, if present
    payload.pop("id", None)
    payload.pop("author_id", None)
    payload.pop("type_id", None)

    # Process cantica_id and add additional fields
    cantica_id = payload.pop("cantica_id", None)
    payload["cantica"] = cantica_id  # Optionally, you can convert cantica_id to a name if needed
    payload["author_names"] = "dante"
    payload["type_name"] = "TEXT"

    # Retrieve text using the updated payload
    response, response_d = retrieve_text(**payload)
    return response, response_d


if __name__ == "__main__":
    from qdrant_client import QdrantClient
    client = QdrantClient(url="http://localhost:6333")
    model = FakeModel()  # Replace with your actual model instance
    collection_name = "dante_fake_text"

    while True:
        input_text = input("Enter a verse or phrase (or type 'exit' to quit): ")
        if input_text.lower() == 'exit':
            print("Exiting the loop. Goodbye!")
            break

        result = query_and_retrieve(model, collection_name, client, query_text=input_text)
        print(result)
