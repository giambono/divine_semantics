import pandas as pd

from src.db_helper import fetch_cantica_data
from src.experiment import process_experiment, get_results_filename, load_embeddings, load_model
from src.find_similarity import find_most_similar_ensemble

if __name__ == "__main__":
    # Example usage
    # MODEL = {"fake": "",
    #          "types": ["TEXT"]
    #          }

    # 1. a model can have one type only
    # 2. given a model, we can average model's embeddings on different types
    # 3. check that authors have rows of the input type

    # MODEL = {"key": "multilingual_e5_text",
    #          "model_name": "intfloat/multilingual-e5-large",
    #          "type": "TEXT"
    #          }
    MODEL = {"key": "multilingual_e5",
             "model_name": "intfloat/multilingual-e5-large",
             "type": "TEXT"
             }
    # MODEL = {"key": "fake_text",
    #          "model_name": "fake",
    #          "type": "TEXT"
    #          }

    WEIGHTS_CONFIG = {
        "key": "weights_1",
        "authors": {
            "dante": 0.0,
            "durling": 0.1,
            "musa": 0.4,
            "kirkpatrick": 0.5
        }
    }

    # process_experiment(MODEL, WEIGHTS_CONFIG)

    # load embedding
    model_key = MODEL["key"]
    weights_key = WEIGHTS_CONFIG["key"]
    results_path = get_results_filename(model_key, weights_key)
    df = load_embeddings(results_path)
    df = df[df["cantica_id"] == 1]

    print()
    while True:
        # Ask the user for input
        input_text = input("Enter a verse or phrase (or type 'exit' to quit): ")

        # Check if the user wants to exit
        if input_text.lower() == 'exit':
            print("Exiting the loop. Goodbye!")
            break

        response = find_most_similar_ensemble(input_text, df, models=load_model(MODEL))

        params = response.iloc[0][['cantica_id', 'canto', 'start_verse', 'end_verse']].to_dict()

        result_df = fetch_cantica_data(**params)

        print("Response:\n", result_df)
        print()
