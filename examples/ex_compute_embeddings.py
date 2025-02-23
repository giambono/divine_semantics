import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from database.qdrant.compute_embeddings import compute_embeddings_and_upsert
from divine_semantics.src.db_helper import get_db_connection
from divine_semantics.src.experiment import load_model


if __name__ == "__main__":

    # Get the SQLite connection and fetch the data
    conn = get_db_connection()  # Ensure get_db_connection() is defined/imported
    df = pd.read_sql_query("SELECT * FROM divine_comedy", conn)  #.iloc[:5]

    qdrant_client = QdrantClient(url="http://localhost:6333")

    model_key = "fake_text"
    models = {model_key: load_model(model_key)}

    compute_embeddings_and_upsert(df, models, qdrant_client, collection_name_prefix="dante_")