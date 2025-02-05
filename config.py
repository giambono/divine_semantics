import os
import yaml


ROOT = os.path.dirname(os.path.abspath(__file__))

TRANSLATIONS_REPO = os.path.join(ROOT, "data/inferno_translations_aligned.ods")

DP_PATH = os.path.join(ROOT, "data", "divine_comedy.db")

TEXTS_FOR_SEMANTIC_SEARCH = {'dante': 0., 'singleton': 0., 'musa': 0.2, 'kirkpatrick': 0.2, 'durling': 0.6}

# #######################################################
# NLP SETTINGS
# #######################################################

CONFIG_NLP_PATH = os.path.join(ROOT, "config_nlp.yaml")

# Load YAML configuration
def load_yaml_config(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}  # Return empty dict if the file is empty
    except FileNotFoundError:
        print(f"Warning: Config file {path} not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {path}: {e}")
        return {}

# Load the NLP configuration
NLP_CONFIG = load_yaml_config(CONFIG_NLP_PATH)

# Example: Accessing model name
MODELS = NLP_CONFIG.get("models", {})

# Print debug info (optional)
if __name__ == "__main__":
    print(f"ROOT: {ROOT}")
    print(f"CONFIG_NLP_PATH: {CONFIG_NLP_PATH}")
    print(f"NLP_CONFIG: {NLP_CONFIG}")
    print(f"MULTILINGUAL_E5: {MODELS}")
