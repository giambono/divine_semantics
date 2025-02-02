import pandas as pd

# Load the ODS file
df = pd.read_excel("inferno_translations_aligned.ods", engine="odf")

# Display the first few rows
print(df.head())
