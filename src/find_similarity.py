import yaml
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


def find_most_similar_ensemble_old(input_text, df, models=None):
    """
    Finds the most similar verse in the 'dante' column based on the highest similarity score
    across all individual model similarities and the ensemble similarity score.

    Parameters:
    input_text (str): The user-provided query.
    df (pd.DataFrame): The DataFrame with stored embeddings.
    models (dict, optional): A dictionary of models to use for embeddings.
                             If not provided, the default models from the config will be used.
                             Format: {"model_name": SentenceTransformer(model_path), ...}

    Returns:
    str: The most similar verse from the 'dante' column.
    """

    # Load default models from config if none are provided
    if models is None:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        models = {name: SentenceTransformer(path) for name, path in config["models"].items()}

    # Compute query embeddings with "query:" prefix
    input_embeddings = {name: model.encode(f"query: {input_text}") for name, model in models.items()}

    # Ensure embeddings are NumPy arrays
    for col in ["embedding_multilingual_e5", "embedding_minilm", "embedding_contriever"]:
        df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

    # Compute similarity for each model
    similarity_columns = []
    for model_name in models.keys():
        column_name = f"similarity_{model_name}"
        df[column_name] = cosine_similarity(
            [input_embeddings[model_name]], np.vstack(df[f"embedding_{model_name}"])
        )[0]
        similarity_columns.append(column_name)

    # Compute final ensemble similarity (average of all models)
    df["similarity_ensemble"] = df[similarity_columns].mean(axis=1)
    similarity_columns.append("similarity_ensemble")  # Add ensemble score to comparison

    # Find the row with the highest value across all similarity columns
    best_match_idx = df[similarity_columns].max(axis=1).idxmax()  # Get the highest similarity index
    most_similar_verse = df.iloc[best_match_idx]["dante"]

    # Determine which similarity column achieved the highest score for this row
    best_similarity_series = df.loc[best_match_idx, similarity_columns]
    best_similarity_column = best_similarity_series.idxmax()
    # Remove the prefix to get the model name; if ensemble wins, return 'ensemble'
    if best_similarity_column == "similarity_ensemble":
        best_model = "ensemble"
    else:
        best_model = best_similarity_column.replace("similarity_", "")

    return most_similar_verse, best_model


def find_most_similar_ensemble(input_text, df, models=None):
    """
    Finds the most similar verse in the 'dante' column based on the highest similarity score
    across all individual model similarities and the ensemble similarity score.
    Also returns the model (or 'ensemble') that achieved this highest similarity.

    Parameters:
    input_text (str): The user-provided query.
    df (pd.DataFrame): The DataFrame with stored embeddings.
    models (dict, optional): A dictionary of models to use for embeddings.
                             If not provided, the default models from the config will be used.
                             For "facebook/contriever", the entry will be a dict with keys "model" and "tokenizer".
                             For other models, the value is a SentenceTransformer instance.

    Returns:
    tuple: (most_similar_verse, best_model) where:
           - most_similar_verse (str): The most similar verse from the 'dante' column.
           - best_model (str): The model name that achieved the highest similarity, or "ensemble" if the ensemble score was highest.
    """
    # Load default models from config if none are provided
    if models is None:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        models = {}
        for name, path in config["models"].items():
            if name == "facebook/contriever":
                models[name] = {
                    "model": AutoModel.from_pretrained(path),
                    "tokenizer": AutoTokenizer.from_pretrained("facebook/contriever")
                }
            else:
                models[name] = SentenceTransformer(path)

    # Compute query embeddings with "query:" prefix for each model
    input_embeddings = {}
    for model_name, model_obj in models.items():
        if model_name == "facebook/contriever":
            tokenizer = model_obj["tokenizer"]
            model_instance = model_obj["model"]
            # Tokenize and encode using mean pooling
            inputs = tokenizer(f"query: {input_text}", return_tensors="pt", truncation=True, max_length=512)
            outputs = model_instance(**inputs)
            token_embeddings = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # shape: [batch_size, seq_len, 1]
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            input_embeddings[model_name] = mean_embedding.squeeze(0).detach().cpu().numpy()
        else:
            input_embeddings[model_name] = model_obj.encode(f"query: {input_text}")

    # Ensure stored embeddings are numpy arrays
    for col in [f"embedding_{key}" for key in models.keys()]:
        df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

    # Compute similarity for each model and store in new columns
    similarity_columns = []
    for model_name in models.keys():
        col_name = f"similarity_{model_name}"
        embeddings = np.vstack(df[f"embedding_{model_name}"])
        df[col_name] = cosine_similarity(
            [input_embeddings[model_name]], embeddings
        )[0]
        similarity_columns.append(col_name)

    # Compute ensemble similarity as the average of the individual model similarities
    df["similarity_ensemble"] = df[similarity_columns].mean(axis=1)
    similarity_columns.append("similarity_ensemble")

    # Identify the best match row (the one with the maximum similarity across any column)
    best_match_idx = df[similarity_columns].max(axis=1).idxmax()
    most_similar_verse = df.loc[best_match_idx, "dante"]

    # # Determine which similarity column had the maximum value for that row
    # best_similarity_series = df.loc[best_match_idx, similarity_columns]
    # best_similarity_column = best_similarity_series.idxmax()
    # best_model = best_similarity_column.replace("similarity_", "") if best_similarity_column != "similarity_ensemble" else "ensemble"

    # return most_similar_verse, best_model

    # Determine the top 2 models/ensemble for that row
    similarity_series = df.loc[best_match_idx, similarity_columns]
    # Sort the similarity values in descending order and take the top 2
    top_two = similarity_series.sort_values(ascending=False).iloc[:2]
    top_models = []
    for col, sim_value in top_two.items():
        model_name = col.replace("similarity_", "") if col != "similarity_ensemble" else "ensemble"
        top_models.append((model_name, sim_value))

    return most_similar_verse, top_models