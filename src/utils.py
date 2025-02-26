from sentence_transformers import SentenceTransformer


import config
from src.fake import FakeModel



def load_model(model_key):
    """Loads the appropriate models, using FakeModel for 'fake'."""

    if model_key.lower().startswith("fake"):
        return FakeModel()

    if model_key in config.MODELS:
        return SentenceTransformer(config.MODELS[model_key])

    raise ValueError(f"Invalid model key {model_key}.")
