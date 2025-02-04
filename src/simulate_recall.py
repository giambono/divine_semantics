import random
import itertools

import pandas as pd
import numpy as np
from nltk.corpus import wordnet
import nltk

from transformers import pipeline

# Download necessary NLP resources
nltk.download('wordnet')

# Load paraphrasing model (only if using a pretrained model)
paraphrase_pipe = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

# === Recall Simulation Methods ===
def drop_random_words(text, drop_prob=0.3):
    """Simulates partial recall by randomly removing words."""
    words = text.split()
    new_words = [word for word in words if random.random() > drop_prob]
    return " ".join(new_words) if new_words else words[0]  # Avoid empty output

def replace_with_synonyms(text, replace_prob=0.2):
    """Simulates recall with slight paraphrasing by replacing words with synonyms."""
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms and random.random() < replace_prob:
            new_word = synonyms[0].lemmas()[0].name().replace("_", " ")  # Choose first synonym
            new_words.append(new_word)
        else:
            new_words.append(word)
    return " ".join(new_words)

def typo_simulation(text, typo_prob=0.15):
    """Simulates minor spelling errors by randomly altering characters."""
    def introduce_typo(word):
        if len(word) > 3 and random.random() < typo_prob:
            i = random.randint(0, len(word) - 2)
            return word[:i] + word[i+1] + word[i] + word[i+2:]  # Swap adjacent letters
        return word

    words = text.split()
    new_words = [introduce_typo(word) for word in words]
    return " ".join(new_words)

def truncate_middle(text):
    """Simulates recall where the user remembers the beginning and end but forgets the middle."""
    words = text.split()
    if len(words) > 5:
        return f"{' '.join(words[:2])} ... {' '.join(words[-2:])}"
    return text

def extract_partial_text(text, fraction=0.5):
    """Extracts a fraction of the tercet to simulate recalling only part of it."""
    words = text.split()
    num_words = max(3, int(len(words) * fraction))  # At least 3 words
    start_index = random.randint(0, len(words) - num_words)
    return " ".join(words[start_index : start_index + num_words])

def generate_paraphrase(text):
    """Generates a paraphrase using a pretrained model."""
    result = paraphrase_pipe(f"paraphrase: {text}", max_length=128, do_sample=True, top_k=50)
    return result[0]['generated_text']

def simulate_user_recall(text, partial=False, use_paraphrasing=False):
    """
    Simulates how a user recalls a tercet.

    Parameters:
    - text (str): The original verse translation.
    - partial (bool): If True, simulate recall of only part of the tercet.
    - use_paraphrasing (bool): If True, use a pretrained model to generate a paraphrase.

    Returns:
    - Simulated user recall string.
    """
    transformations = [
        drop_random_words,
        replace_with_synonyms,
        typo_simulation,
        truncate_middle
    ]

    # If partial recall is enabled, extract part of the tercet first
    if partial:
        text = extract_partial_text(text, fraction=0.5)

    # If using a pretrained model for paraphrasing
    if use_paraphrasing:
        return generate_paraphrase(text)

    # Apply a random transformation
    return random.choice(transformations)(text)


# === Function to Generate Test Set with a Limit on Instances & Paraphrasing Option ===
def build_test_set(df, text_column="translation", num_queries_per_row=3, max_instances=None,
                   partial_recall=False, use_paraphrasing=False):
    """
    Given a DataFrame where each row is a tercet translation,
    generate simulated user recall queries.

    Parameters:
    - df (pd.DataFrame): The dataframe with translations.
    - text_column (str): The column containing the verse translation.
    - num_queries_per_row (int): How many recall variations to generate per row.
    - max_instances (int, optional): Maximum number of test queries to generate.
    - partial_recall (bool, optional): If True, users recall only **part of the tercet**.
    - use_paraphrasing (bool, optional): If True, use a **pretrained paraphrasing model**.

    Returns:
    - pd.DataFrame: A new DataFrame with simulated queries and expected matches.
    """
    test_data = []

    for idx, row in df.iterrows():
        original_text = row[text_column]
        for _ in range(num_queries_per_row):
            simulated_query = simulate_user_recall(original_text,
                                                   partial=partial_recall,
                                                   use_paraphrasing=use_paraphrasing)
            test_data.append({"query": simulated_query, "expected_index": idx})

        # Stop if we reach the limit
        if max_instances and len(test_data) >= max_instances:
            break

    # If the test set exceeds the max limit, randomly sample it
    if max_instances and len(test_data) > max_instances:
        test_data = random.sample(test_data, max_instances)

    return pd.DataFrame(test_data)


def run_simulation(df):
    # Define the possible values for each parameter
    text_columns = ['dante', 'singleton', 'musa', 'kirkpatrick', 'durling']
    partial_recall_options = [True, False]
    use_paraphrasing_options = [True, False]

    # Create all possible combinations
    parameter_combinations = list(itertools.product(text_columns, partial_recall_options, use_paraphrasing_options))

    # Store results in a list
    all_queries = []

    # Iterate over all parameter combinations
    for text_column, partial_recall, use_paraphrasing in parameter_combinations:
        print(f"Generating test set for: text_column={text_column}, partial_recall={partial_recall}, use_paraphrasing={use_paraphrasing}")

        # Generate the test set for the current parameter combination
        df_queries = build_test_set(df, text_column=text_column,
                                    num_queries_per_row=3, max_instances=100,
                                    partial_recall=partial_recall, use_paraphrasing=use_paraphrasing)

        # Add metadata about the input parameters
        df_queries["text_column"] = text_column
        df_queries["partial_recall"] = partial_recall
        df_queries["use_paraphrasing"] = use_paraphrasing

        # Store the results
        all_queries.append(df_queries)

    # Concatenate all results into a single DataFrame
    df_all_queries = pd.concat(all_queries, ignore_index=True)
    return df_all_queries


if __name__ == "__main__":
    import itertools
    import pandas as pd

    df = pd.read_pickle("/home/rfflpllcn/IdeaProjects/divine_semantics/out/ensemble_embeddings.pkl")
    df = df[['volume', 'canto', 'verse', 'dante', 'singleton', 'musa', 'kirkpatrick', 'durling']]

    out = run_simulation(df)
    df.to_pickle("out/test_set.pkl")

