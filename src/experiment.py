import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import hashlib

import config
from src.compute import compute_embeddings, weighted_avg_embedding
from src.compute_sqlite import compute_embeddings as compute_embeddings_sqlite
from src.compute_sqlite import weighted_avg_embedding as weighted_avg_embedding_sqlite
from src.fake import FakeModel
from src.retrieve import fetch_author_ids_from_db, fetch_author_ids_from_db_sqlite


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


def load_models(model_name, model_dict):
    """Loads the appropriate models, using FakeModel for 'fake'."""
    model_dict_ = {model_name: model_dict[model_name]}
    return {k: (FakeModel() if k.lower() == "fake" else SentenceTransformer(m)) for k, m in model_dict_.items()
            }


def fetch_author_weights(authors_name_weights):
    """Fetch author IDs from database and map them to their corresponding weights."""
    authors_name_ids = fetch_author_ids_from_db(list(authors_name_weights.keys()))
    return {authors_name_ids[name]: weight for name, weight in authors_name_weights.items()}


def fetch_author_weights_sqlite(authors_name_weights):
    """Fetch author IDs from database and map them to their corresponding weights."""
    authors_name_ids = fetch_author_ids_from_db_sqlite(list(authors_name_weights.keys()))
    return {authors_name_ids[name]: weight for name, weight in authors_name_weights.items()}


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

def get_results_filename(model_name, weights_name, weights_hash=None):
    """Generate a filename for results based on weights hash."""
    if weights_hash is None:
        return os.path.join(config.EXPERIMENTS_ROOT, "results", f"{model_name}_{weights_name}", "embeddings.parquet")
    return os.path.join(config.EXPERIMENTS_ROOT, "results", f"{model_name}_{weights_name}_{weights_hash}", "embeddings.parquet")

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

def process_experiment(model_config, weights_config, is_sqlite=True):
    """Main function to execute the entire workflow."""
    create_folder_structure()

    # Save model configuration
    model_name = next(k for k in model_config if k != "types")
    save_json(model_config, "experiments", "models", model_name, "config.json")

    # Save weight configuration
    weights_name = weights_config["name"]
    save_json(weights_config, "experiments", "weights", f"{weights_name}.json")

    # Compute author weights
    authors_name_weights = weights_config["authors"]
    if is_sqlite:
        authors_id_weights = fetch_author_weights_sqlite(authors_name_weights)
    else:
        authors_id_weights = fetch_author_weights(authors_name_weights)
    authors_names = list(authors_name_weights.keys())

    # Generate hashes for types and weights
    # types_hash = get_types_hash(weights_config["types"])
    # weights_hash = get_weights_hash(weights_config["authors"])

    # Get the correct embeddings filename
    embeddings_path = get_embeddings_filename(model_name)

    # Load or compute embeddings
    if embeddings_exist(embeddings_path):
        df_embeddings = load_embeddings(embeddings_path)
    else:
        # Load models and compute embeddings
        models = load_models(model_name, model_config)
        if is_sqlite:
            df_embeddings = compute_embeddings_sqlite(models, authors_names, weights_config)
        else:
            df_embeddings = compute_embeddings(authors_names, model_config["types"], models=models)

        # Save embeddings with a unique filename
        save_embeddings(df_embeddings, embeddings_path)

    # Get the correct results filename based on weights
    results_path = get_results_filename(model_name, weights_name)

    # Compute weighted average embeddings (always recompute since weights_config can change)
    avg_weights_df = weighted_avg_embedding(model_name, df_embeddings, authors_id_weights)

    # Save weighted embeddings, ensuring different weights configurations do not overwrite
    save_parquet(avg_weights_df, results_path)


if __name__ == "__main__":
    # Example usage
    MODEL = {"fake": "",
             "types": ["TEXT"]
             }
    WEIGHTS_CONFIG = {
        "name": "weights_2",
        "authors": {"durling": 0.2, "musa": 0.8}
    }

    process_experiment(MODEL, WEIGHTS_CONFIG)

    # load embedding
    model_name = next(k for k in MODEL if k != "types")
    weights_name = WEIGHTS_CONFIG["name"]
    results_path = get_results_filename(model_name, weights_name)
    df = load_embeddings(results_path)

    df.to_clipboard()

