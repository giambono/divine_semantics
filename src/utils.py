import os
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import config
from src.fake import FakeModel



def load_model(model_key):
    """Loads the appropriate models, using FakeModel for 'fake'."""

    if model_key.lower().startswith("fake"):
        return FakeModel()

    if model_key in config.MODELS:
        return SentenceTransformer(config.MODELS[model_key])

    raise ValueError(f"Invalid model key {model_key}.")


def setup_environment():
    """Load environment variables from .env file."""
    load_dotenv()

def initialize_qdrant_client():
    """Initialize and return the Qdrant client using environment variables."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

def initialize_model(model_name):
    """
    Load the model and compute embedding dimension.

    Returns:
        model: the loaded model.
        embedding_dim: dimension of the model's embedding.
    """
    model = load_model(model_name)
    sample_embedding = model.encode("test")
    embedding_dim = len(sample_embedding)
    return model, embedding_dim

def load_test_queries(filepath, n=2):
    """
    Load and return test queries DataFrame with the required columns.

    Args:
        filepath: path to the parquet file.
        n: number of test queries to load.
    """
    df = pd.read_parquet(filepath)
    return df[["transformed_text", "expected_index"]].iloc[:n]
