"""
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.n_H2-Mn_tP4xlhhh5-tVN2PQLictMcv-izNxIBsOJgQ

from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://5396fea8-cac7-4233-8bb3-d9800bac734c.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.n_H2-Mn_tP4xlhhh5-tVN2PQLictMcv-izNxIBsOJgQ",
)

print(qdrant_client.get_collections())
"""
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.compute_embeddings import compute_embeddings_and_upsert
from src.db_helper import get_db_connection
from src.utils import load_model


if __name__ == "__main__":

    # Get the SQLite connection and fetch the data
    conn = get_db_connection()  # Ensure get_db_connection() is defined/imported
    df = pd.read_sql_query("SELECT * FROM divine_comedy", conn)  #.iloc[:5]

    qdrant_client = QdrantClient(url="http://localhost:6333")

    model_key = "fake_text"
    models = {model_key: load_model(model_key)}

    compute_embeddings_and_upsert(df, models, qdrant_client, collection_name_prefix="dante_")