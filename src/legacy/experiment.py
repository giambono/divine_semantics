import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import hashlib

import config
from src.compute import compute_embeddings, weighted_avg_embedding
from src.fake import FakeModel
from src.db_helper import fetch_author_ids_from_db


def ensure_folder_exists(*paths):
    """Create a directory if it does not exist and return its path."""
    path = os.path.join(config.ROOT, *paths)
    os.makedirs(path, exist_ok=True)
    return path


def create_folder_structure():
    """Creates the necessary folder structure for experiments."""
    os.makedirs(config.EXPERIMENTS_ROOT, exist_ok=True)
    folders = ["embeddings", "models", "results", "weights"]
    for folder in folders:
        ensure_folder_exists(os.path.join(config.EXPERIMENTS_ROOT, folder))
    print("Folder structure created successfully.")


def save_json(data, *path_parts):
    """Saves a dictionary as a JSON file, ensuring the directory exists."""
    file_path = os.path.join(config.ROOT, *path_parts)
    folder_path = os.path.dirname(file_path)
    os.makedirs(folder_path, exist_ok=True)  # Ensure directory exists

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_model(model_key):
    """Loads the appropriate models, using FakeModel for 'fake'."""

    if model_key.lower().startswith("fake"):
        return FakeModel() \

    if model_key in config.MODELS:
        return SentenceTransformer(config.MODELS[model_key])

    raise ValueError(f"Invalid model key {model_key}.")


def load_parquet(*path_segments):
    """Load a Parquet file given a path."""
    path = os.path.join(*path_segments)
    return pd.read_parquet(path)


def save_parquet(df, file_path):
    """Saves a DataFrame as a Parquet file."""
    folder_path = os.path.dirname(file_path)
    os.makedirs(folder_path, exist_ok=True)

    df.to_parquet(file_path, engine="pyarrow")  # Save directly to the correct file path


def get_types_hash(types_list):
    """Generate a hash from the types list to track changes."""
    return hashlib.md5(json.dumps(sorted(types_list)).encode()).hexdigest()


def get_weights_hash(weights_dict):
    """Generate a hash from the weights dictionary to track changes."""
    return hashlib.md5(json.dumps(weights_dict, sort_keys=True).encode()).hexdigest()


def get_embeddings_filename(model_name, types_hash=None):
    """Generate a filename for embeddings based on model and types hash."""
    if types_hash is None:
        return os.path.join(config.EXPERIMENTS_ROOT, "embeddings", f"{model_name}", "embeddings.parquet")
    return os.path.join(config.EXPERIMENTS_ROOT, "embeddings", f"{model_name}_{types_hash}", "embeddings.parquet")


def get_results_filename(model_key, weights_key, weights_hash=None):
    """Generate a filename for results based on weights hash."""
    if weights_hash is None:
        return os.path.join(config.EXPERIMENTS_ROOT, "results", f"{model_key}_{weights_key}", "embeddings.parquet")
    return os.path.join(config.EXPERIMENTS_ROOT, "results", f"{model_key}_{weights_key}_{weights_hash}",
                        "embeddings.parquet")


def embeddings_exist(embeddings_path):
    """Check if embeddings file exists."""
    path = os.path.join(config.ROOT, embeddings_path)
    return os.path.exists(path)


def save_embeddings(df_embeddings, embeddings_path):
    """Save embeddings to a specified path."""
    folder_path = os.path.dirname(embeddings_path)  # Get parent directory
    os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists (not the file itself)
    save_parquet(df_embeddings, embeddings_path)  # Save file


def load_embeddings(embeddings_path):
    """Load embeddings from a specified path."""
    path = os.path.join(config.ROOT, embeddings_path)
    return pd.read_parquet(path)


def process_experiment(model_config, weights_config):
    """Main function to execute the entire workflow."""
    create_folder_structure()

    # Save model configuration
    model_key = model_config["key"]
    save_json(model_config, "experiments", "models", model_key, "config.json")

    # Save weight configuration
    weights_key = weights_config["key"]
    save_json(weights_config, "experiments", "weights", f"{weights_key}.json")

    # Compute author weights
    authors_name_weights = weights_config["authors"]
    authors_names = list(authors_name_weights.keys())

    authors_name_ids = fetch_author_ids_from_db(authors_names)
    authors_id_weights = {authors_name_ids[name]: weight for name, weight in authors_name_weights.items() if
                          name in authors_name_ids}

    # Generate hashes for types and weights
    # types_hash = get_types_hash(weights_config["types"])
    # weights_hash = get_weights_hash(weights_config["authors"])

    # Get the correct embeddings filename
    embeddings_path = get_embeddings_filename(model_key)

    # Load or compute embeddings
    if embeddings_exist(embeddings_path):
        print(f"embeddings already exists for {model_key}, loading from {embeddings_path}")
        df_embeddings = load_embeddings(embeddings_path)
    else:
        # Load models and compute embeddings
        model = load_model(model_config)
        df_embeddings = compute_embeddings(authors_names, model_config["type"], models=model)

        # Save embeddings with a unique filename
        save_embeddings(df_embeddings, embeddings_path)

    # Get the correct results filename based on weights
    results_path = get_results_filename(model_key, weights_key)

    # Compute weighted average embeddings (always recompute since weights_config can change)
    avg_weights_df = weighted_avg_embedding(model_key, df_embeddings, authors_id_weights)

    # Save weighted embeddings, ensuring different weights configurations do not overwrite
    save_parquet(avg_weights_df, results_path)
