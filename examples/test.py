import os
import json
from sentence_transformers import SentenceTransformer

import config
from src.compute import compute_embeddings, weighted_avg_embedding
from src.fake import FakeModel
from src.retrieve import fetch_author_ids_from_db

MODEL = "model1"
WEIGHTS = "weights_1"


def create_folder_structure():
    # Define the folder structure
    folders = [
        "experiments/embeddings",
        "experiments/models",
        "experiments/results",
        "experiments/weights"
    ]

    # Create the folders if they do not exist
    for folder in folders:
        os.makedirs(os.path.join(config.ROOT, folder), exist_ok=True)

    print("Folder structure created successfully.")

# load model and weights

with open(os.path.join(config.ROOT, "experiments", "models", f"{MODEL}/config.json")) as f:
    models = json.load(f)

with open(os.path.join(config.ROOT, "experiments", "weights", f"{WEIGHTS}.json")) as f:
    weights_config = json.load(f)

authors_name_weights = weights_config["authors"]
authors_name_ids = fetch_author_ids_from_db(list(authors_name_weights.keys()))
authors_id_weights = {authors_name_ids[n]: w for n, w in authors_name_weights.items()}
authors_names = list(authors_name_weights.keys())

types = weights_config["types"]

models = {k: FakeModel() if k.lower() == "fake" else SentenceTransformer(m) for k, m in models.items()}

df_embeddings = compute_embeddings(authors_names, types, models=models)
saveto = os.path.join(config.ROOT, "experiments", "embeddings", f"{MODEL}")
os.makedirs(saveto, exist_ok=True)
df_embeddings.to_parquet(os.path.join(saveto, "embeddings.parquet"), engine="pyarrow")

avg_weights_df = weighted_avg_embedding("fake", df_embeddings, authors_id_weights)
saveto = os.path.join(config.ROOT, "experiments", "results", f"{MODEL}_{WEIGHTS}")
os.makedirs(saveto, exist_ok=True)
avg_weights_df.to_parquet(os.path.join(saveto, "weighted_embeddings.parquet"), engine="pyarrow")
