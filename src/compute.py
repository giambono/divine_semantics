import yaml
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def compute_ensemble_embeddings_old(df, columns, models=None):
    """
    Computes embeddings for each model and stores them in separate columns.

    Parameters:
    df (pd.DataFrame): The DataFrame with verse translations.
    columns (list): The text columns to embed.
    models (dict, optional): A dictionary of models to use for embeddings.
                             If not provided, the default models from the config will be used.
                             Format: {"model_name": SentenceTransformer(model_path), ...}

    Returns:
    pd.DataFrame: DataFrame with separate embeddings for each model.
    """
    # Load default models from config if none are provided
    if models is None:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        models = {name: SentenceTransformer(path) for name, path in config["models"].items()}

    # Compute embeddings for each model
    for model_name, model in models.items():
        print(f"Computing embeddings with {model_name}...")
        df[f"embedding_{model_name}"] = df.apply(
            lambda row: np.mean([model.encode(row[col]) for col in columns if pd.notnull(row[col])], axis=0),
            axis=1
        )
    return df

import numpy as np
import pandas as pd
import yaml
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

def compute_ensemble_embeddings(df, columns, models=None):
    """
    Computes embeddings for each model and stores them in separate columns.

    Parameters:
    df (pd.DataFrame): The DataFrame with verse translations.
    columns (list): The text columns to embed.
    models (dict, optional): A dictionary of models to use for embeddings.
                             If not provided, the default models from the config will be used.
                             Format: {"model_name": model_instance, ...}
                             Note: For "facebook/contriever", model_instance should be an instance of AutoModel.
                                   Its corresponding tokenizer will be loaded automatically.

    Returns:
    pd.DataFrame: DataFrame with separate embeddings for each model.
    """
    # Load default models from config if none are provided
    if models is None:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        models = {}
        for name, path in config["models"].items():
            if name == "facebook/contriever":
                # For contriever, load the AutoModel from transformers.
                models[name] = AutoModel.from_pretrained(path)
            else:
                models[name] = SentenceTransformer(path)

    # Iterate over each model and compute embeddings
    for model_name, model in models.items():
        print(f"Computing embeddings with {model_name}...")

        # Special handling for facebook/contriever which requires manual mean pooling.
        if model_name == "facebook/contriever":
            tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

            def encode_contriever(text):
                # Tokenize the text
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                # Get the model output
                outputs = model(**inputs)
                # Extract token embeddings from the last hidden state
                token_embeddings = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_dim]
                # Get the attention mask and expand its dimensions to match token_embeddings
                attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # shape: [batch_size, seq_len, 1]
                # Compute the sum of embeddings for tokens that are not padding
                sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                # Avoid division by zero
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                # Compute the mean by dividing the summed embeddings by the sum of the mask
                mean_embedding = sum_embeddings / sum_mask
                return mean_embedding.squeeze(0).detach().cpu().numpy()

            # Apply encoding for each row and compute the mean across specified columns
            df[f"embedding_{model_name}"] = df.apply(
                lambda row: np.mean(
                    [encode_contriever(row[col]) for col in columns if pd.notnull(row[col])],
                    axis=0
                ),
                axis=1
            )
        else:
            # Use SentenceTransformer's encode method for other models
            df[f"embedding_{model_name}"] = df.apply(
                lambda row: np.mean(
                    [model.encode(row[col]) for col in columns if pd.notnull(row[col])],
                    axis=0
                ),
                axis=1
            )
    return df
