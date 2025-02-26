import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)


all_points = []
offset = None  # Qdrant scroll API uses None as the initial offset

# Retrieve all points from the collection in batches
while True:
    scroll_result, next_offset = qdrant_client.scroll(
        collection_name="dante_multilingual_e5",
        limit=100,
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