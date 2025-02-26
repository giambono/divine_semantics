
import numpy as np
import pandas as pd
import yaml
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from src.experiment import load_model


def find_most_similar_ensemble(input_text, df, model_key, nlargest=5):
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
    model = load_model({"key": model_key, "model_name": ""})
    model_obj = model[model_key]

    # Compute query embeddings with "query:" prefix for each model
    input_embeddings = {}
    input_embeddings[model_key] = model_obj.encode(f"query: {input_text}")

    # Ensure stored embeddings are numpy arrays
    embedding_cols = [_col for _col in df.columns if "embedding" in _col]
    for col in embedding_cols:
        df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

    # Compute similarity for each model and store in new columns
    similarity_columns = []
    for col in embedding_cols:
        col_name = f"similarity_{col}"
        embeddings = np.vstack(df[col])
        df[col_name] = cosine_similarity(
            [input_embeddings[model_key]], embeddings
        )[0]
        similarity_columns.append(col_name)

    topn = df[[col for col in df.columns if col.startswith("similarity")]].stack().nlargest(nlargest)
    topn_indices = topn.index.get_level_values(0)

    return df.loc[topn_indices]


if __name__ == "__main__":
    from divine_semantics.src.experiment import process_experiment, get_results_filename, load_embeddings, get_embeddings_filename, load_model
    from divine_semantics.src.find_similarity import find_most_similar_ensemble
    from divine_semantics.src.db_helper import fetch_cantica_data
    from sentence_transformers import SentenceTransformer

    model_key = "fake_text"
    weights_key =  "weights_1"

    model = load_model({"key": model_key})

    # load embeddings (with columns 'embedding_{model_key}')
    embeddings_path = get_embeddings_filename(model_key)
    embeddings = load_embeddings(embeddings_path)
    embeddings = embeddings[embeddings["cantica_id"] == 1]

    # load weighted embeddings (with columns 'weighted_embedding_{model_key}')
    results_path = get_results_filename(model_key, weights_key)
    weighted_embeddings = load_embeddings(results_path)
    weighted_embeddings = weighted_embeddings[weighted_embeddings["cantica_id"] == 1]

    df = embeddings #weighted_embeddings
    df["embeddings2"] = df["embedding_fake_text"]
    while True:
        # Ask the user for input
        input_text = input("Enter a verse or phrase (or type 'exit' to quit): ")

        # Check if the user wants to exit
        if input_text.lower() == 'exit':
            print("Exiting the loop. Goodbye!")
            break

        response = find_most_similar_ensemble(input_text, df, model_key=model_key)
        print(response)
        params = response.iloc[0][['cantica_id', 'canto', 'start_verse', 'end_verse']].to_dict()

        result_df = fetch_cantica_data(**params)

        print("Response:\n", result_df)
        print()