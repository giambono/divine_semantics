import re
from datasets import load_dataset


# dataset = load_dataset("maiurilorenzo/divina-commedia")  #TODO: cache it locally
# df = dataset["train"].to_pandas()
#
#
# def convert_roman_to_int(canto):
#     roman_numerals = {
#         'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
#         'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
#         'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25, 'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29, 'XXX': 30,
#         'XXXI': 31, 'XXXII': 32, 'XXXIII': 33, 'XXXIV': 34
#     }
#     match = re.match(r'Canto (\w+)', canto)
#     if match:
#         roman = match.group(1)
#         return roman_numerals.get(roman, canto)  # Return converted number if valid, else original canto
#     return canto
#
# # Transform dataframe
# df_grouped = df.groupby(["volume", "canto", "tercet"]).agg({
#     "verse_number": lambda x: f"{min(x)}-{max(x)}",
#     "text": lambda x: " ".join(x)
# }).reset_index()
#
# # Convert "canto" column to an ordinal number
# df_grouped["canto"] = df_grouped["canto"].apply(convert_roman_to_int)
#
#
# volume_order = {"Inferno": 1, "Purgatorio": 2, "Paradiso": 3}
#
# # Sort dataframe first by volume (using the custom order) and then by canto
# df_grouped["volume_order"] = df_grouped["volume"].map(volume_order)  # Map volumes to numeric order
# df_sorted = df_grouped.sort_values(by=["volume_order", "canto"]).drop(columns=["volume_order"])
#
#
# # Save to CSV
# df_sorted.to_csv("commedia_it_grouped_by_triplets.csv", index=False)

import pandas as pd

df = pd.read_csv("commedia_it_grouped_by_triplets.csv")

print()
