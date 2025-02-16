import pandas as pd
import re

# Mapping for Roman numerals in Canto
ROMAN_NUMERAL_MAP = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
    'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25, 'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29, 'XXX': 30,
    'XXXI': 31, 'XXXII': 32, 'XXXIII': 33, 'XXXIV': 34
}

# Mapping for Volume (Inferno, Purgatorio, Paradiso)
VOLUME_MAP = {
    "Inferno": 1,
    "Purgatorio": 2,
    "Paradiso": 3
}

# Function to convert "Canto X" -> 10
def convert_canto_to_int(canto_str):
    match = re.search(r'Canto\s+([IVXLCDM]+)', canto_str)  # Extract the Roman numeral
    if match:
        roman_numeral = match.group(1)
        return ROMAN_NUMERAL_MAP.get(roman_numeral, None)  # Convert to integer
    return None  # Return None if not a valid canto

path = r"/home/rfflpllcn/IdeaProjects/divine_semantics/data/paraphrased_verses.parquet"
df = pd.read_parquet(path)

# Apply transformations
df["canto"] = df["canto"].apply(convert_canto_to_int)  # Convert Canto to integer
df["volume"] = df["volume"].map(VOLUME_MAP)  # Convert Volume names to numbers

path = r"/home/rfflpllcn/IdeaProjects/divine_semantics/data/paraphrased_verses2.parquet"
df.to_parquet(path, engine="pyarrow", index=False)


