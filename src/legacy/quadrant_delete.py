from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

# Retrieve list of collections
collections = client.get_collections()

# Iterate over each collection and delete it
for collection in collections.collections:
    client.delete_collection(collection_name=collection.name)
    print(f"Deleted collection: {collection.name}")