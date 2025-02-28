import re
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


dataset = load_dataset("maiurilorenzo/divina-commedia")
# dataset = load_dataset("giambono/commedia_en")

# Display the first few entries
df = dataset["train"].to_pandas()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

# Input text you want to paraphrase
text = "Your input text to be paraphrased."

# Prepend the task instruction and add the end-of-sequence token if needed
input_text = "paraphrase: " + text + " </s>"

# Tokenize the input text
encoding = tokenizer.encode_plus(
    input_text,
    return_tensors="pt",
    max_length=256,
    padding="max_length",
    truncation=True
)
input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

# Generate paraphrased text using beam search
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=256,
    num_beams=5,
    repetition_penalty=10.0,
    length_penalty=1.0,
    early_stopping=True
)

# Decode the generated tokens to a string
paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
print("Paraphrased text:", paraphrased_text)
