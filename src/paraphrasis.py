import os
import pandas as pd
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def process_verses(df, text_column='text', volume_column='volume', canto_column='canto',
                   tercet_column='tercet', verse_number_column='verse_number', num_return_sequences=5):
    """
    Processes each verse from the input DataFrame, generating five paraphrased versions for each verse
    using a transformer model and storing them in a new DataFrame along with the original metadata.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the verses of the Divine Comedy.
        text_column (str): Column name for the verse text.
        volume_column (str): Column name for the volume (Inferno, Purgatorio, Paradiso).
        canto_column (str): Column name for the canto number.
        tercet_column (str): Column name for the tercet number.
        verse_number_column (str): Column name for the verse number.

    Returns:
        pd.DataFrame: A new DataFrame containing the original metadata and five paraphrased outputs for each verse.
    """
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_data = []

    for _, row in df.iterrows():
        verse_text = row[text_column]
        text = "paraphrase: " + verse_text + " </s>"
        encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=200,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=num_return_sequences
        )

        paraphrased_texts = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]

        for i, paraphrased_text in enumerate(paraphrased_texts, 1):
            output_data.append({
                'volume': row[volume_column],
                'canto': row[canto_column],
                'tercet': row[tercet_column],
                'verse_number': row[verse_number_column],
                'original_text': verse_text,
                'output_number': i,
                'transformed_text': paraphrased_text
            })

    return pd.DataFrame(output_data)

# dataset = load_dataset("maiurilorenzo/divina-commedia")
dataset = load_dataset("giambono/commedia_en")

df = dataset["train"].to_pandas().iloc[:10]

result_df = process_verses(df, num_return_sequences=1)
result_df.to_csv("../data/paraphrases.csv", index=False)
print(result_df.transformed_text)