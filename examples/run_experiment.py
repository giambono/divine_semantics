import pandas as pd

from src.compute_sqlite import fetch_cantica_data
from src.experiment import process_experiment, get_results_filename, load_embeddings
from src.find_similarity import find_most_similar_ensemble

if __name__ == "__main__":
    # Example usage
    # MODEL = {"fake": "",
    #          "types": ["TEXT"]
    #          }
    IS_SQLITE = True
    MODEL = {"multilingual_e5": "intfloat/multilingual-e5-large",
             "types": ["TEXT"]
             }

    WEIGHTS_CONFIG = {
        "name": "weights_1",
        "authors": {
            "dante": 0.0,
            "durling": 0.1,
            "musa": 0.4,
            "kickpatrick": 0.5
        }
    }

    # process_experiment(MODEL, WEIGHTS_CONFIG, is_sqlite=IS_SQLITE)

    # load embedding
    model_name = next(k for k in MODEL if k != "types")
    weights_name = WEIGHTS_CONFIG["name"]
    results_path = get_results_filename(model_name, weights_name)
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

        response = find_most_similar_ensemble(input_text, df)

        params = response.iloc[0][['cantica_id', 'canto', 'start_verse', 'end_verse']].to_dict()

        result_df = fetch_cantica_data(**params)

        print("Response:\n", result_df)
        print()
