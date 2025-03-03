from dotenv import load_dotenv
from qdrant_client import QdrantClient

from src.utils import load_model

import os
import pandas as pd

from src.query import run_evaluation

import config

load_dotenv()

if __name__ == "__main__":

    commedia = pd.read_csv(r"/home/rfflpllcn/IdeaProjects/divine_semantics_db/divine_semantics_db/data/commedia_modified.csv",
                           sep=";", quotechar='"')

    commedia = commedia[commedia["author"] == "dante"]

    # Initialize Qdrant client and model
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    collection_name = "dante_parafrasi_multilingual_e5"
    model = load_model("multilingual_e5")

    query_text = "che lasciai la strada"
    query_embedding = model.encode(query_text)

    hits = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=1
    )

    cum_verse_number = hits.points[0].payload['cum_verse_number']

    result = commedia[commedia["cum_verse_number"] == cum_verse_number].to_dict(orient="records")[0]
    out = f"cantica: {result['cantica']}\ncanto: {result['canto']}\nverse_number: {result['verse_number']}\nverse: {result['text']}\nuser query: {query_text}"
    print(out)

