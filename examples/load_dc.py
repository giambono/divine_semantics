import pandas as pd
from datasets import load_dataset
import config
from src.db_helper import get_db_connection

import sqlite3
import pandas as pd

def map_tercets_to_indices(df, conn):
    """
    Maps tercets from the SQLite table 'divine_comedy' to verse indices in the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing indexed verses with columns
                           ['volume', 'canto', 'tercet', 'verse_number', 'text'].
        conn (sqlite3.Connection): SQLite connection object.

    Returns:
        dict: Mapping of (cantica_id, canto, start_verse, end_verse) to lists of verse indices.
    """
    # Load tercets from the SQLite table
    query = "SELECT cantica_id, canto, start_verse, end_verse FROM divine_comedy"
    tercets = pd.read_sql_query(query, conn)

    # Create a mapping of (cantica_id, canto, start_verse, end_verse) → verse indices in df
    mappings = {}

    for _, row in tercets.iterrows():
        cantica_id, canto, start_verse, end_verse = row

        # Get corresponding verse indices from df
        matching_indices = df.loc[
            (df['volume'] == cantica_id) &
            (df['canto'] == canto) &
            (df['verse_number'].between(start_verse, end_verse))
            ].index.tolist()

        # Store mapping
        key = (cantica_id, canto, start_verse, end_verse)
        mappings[key] = matching_indices

    return mappings

dataset = load_dataset("maiurilorenzo/divina-commedia")
df = dataset["train"].to_pandas()

# Example usage (assuming df and conn are defined)
conn = get_db_connection()  # Define your database connection
tercet_mappings = map_tercets_to_indices(df, conn)

print(list(tercet_mappings.items())[:5])



