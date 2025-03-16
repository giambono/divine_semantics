import re
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


dataset = load_dataset("maiurilorenzo/divina-commedia")
# dataset = load_dataset("giambono/commedia_en")

# Display the first few entries
df = dataset["train"].to_pandas()

doc = "".join(df.iloc[0:5].text.values)


from keybert import KeyBERT

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(1, 1),  # single and two-word phrases
    # stop_words='italian',          # remove common Italian words
    # top_n=5,                       # top 5 keywords/phrases
    # use_mmr=True,                  # Maximal Marginal Relevance for diversity
    # diversity=0.6
)

print(keywords)

print(keywords)