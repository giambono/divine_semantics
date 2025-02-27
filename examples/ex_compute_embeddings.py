"""

"""
import ast
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
    verse_mappings = pd.read_sql_query("SELECT * FROM verse_mappings", conn)

    # Merge the cumulative_indices column from verse_mappings into df
    df = df.merge(
        verse_mappings[['cantica_id', 'canto', 'start_verse', 'end_verse', 'cumulative_indices']],
        on=['cantica_id', 'canto', 'start_verse', 'end_verse'],
        how='left'
    )

    # Convert the string to a list
    df['cumulative_indices'] = df['cumulative_indices'].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )

    qdrant_client = QdrantClient(url="http://localhost:6333")

    model_key = "fake_text"
    models = {model_key: load_model(model_key)}

    compute_embeddings_and_upsert(df, models, qdrant_client, collection_name_prefix="dante_")