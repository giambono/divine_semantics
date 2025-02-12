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
        "name": "weights_2",
        "authors": {"dante": 0.0, "durling": 0.2, "musa": 0.8}
    }

    process_experiment(MODEL, WEIGHTS_CONFIG, is_sqlite=IS_SQLITE)

    # load embedding
    model_name = next(k for k in MODEL if k != "types")
    weights_name = WEIGHTS_CONFIG["name"]
    results_path = get_results_filename(model_name, weights_name)
    df = load_embeddings(results_path)

    while True:
        # Ask the user for input
        input_text = input("Enter a verse or phrase (or type 'exit' to quit): ")

        # Check if the user wants to exit
        if input_text.lower() == 'exit':
            print("Exiting the loop. Goodbye!")
            break

        # Process the input and generate a response
        response = find_most_similar_ensemble(input_text, df)

        # Print the response
        print("Response:", response)
        print()

        """error:
         "File "/home/rfflpllcn/IdeaProjects/divine_semantics/src/find_similarity.py", line 116, in find_most_similar_ensemble
         most_similar_verse = df.loc[best_match_idx, "dante"]
         File "/home/rfflpllcn/IdeaProjects/divine_semantics/venv/lib/python3.9/site-packages/pandas/core/indexing.py", line 1183, in __getitem__
    return self.obj._get_value(*key, takeable=self._takeable)
File "/home/rfflpllcn/IdeaProjects/divine_semantics/venv/lib/python3.9/site-packages/pandas/core/frame.py", line 4214, in _get_value
series = self._get_item_cache(col)
File "/home/rfflpllcn/IdeaProjects/divine_semantics/venv/lib/python3.9/site-packages/pandas/core/frame.py", line 4638, in _get_item_cache
loc = self.columns.get_loc(item)
File "/home/rfflpllcn/IdeaProjects/divine_semantics/venv/lib/python3.9/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
raise KeyError(key) from err
KeyError: 'dante'")
        """