from qdrant_client import QdrantClient

from divine_semantics.src.db_helper import retrieve_text, cantica_id2name
from divine_semantics.src.fake import FakeModel

client = QdrantClient(url="http://localhost:6333")

# collections = client.get_collections()
# print(collections)

query_text = "Ok"
model = FakeModel()

search_result = client.query_points(
    collection_name="dante_fake_text",
    query=model.encode(query_text),
    with_payload=True,
    limit=1
).points[0]

payload = search_result.payload
model = payload.pop('model')
payload.pop('id')
payload.pop('author_id')
payload.pop('type_id')
cantica_id = payload.pop('cantica_id')
payload['cantica'] = cantica_id  # cantica_id2name(cantica_id)
payload['author_names'] = "dante"
payload['type_name'] = "TEXT"

response = retrieve_text(**payload)
print(response)
