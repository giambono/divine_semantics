import os
import yaml
from dotenv import load_dotenv

load_dotenv()

DB_TYPE = "sqlite"  # Change to "mysql" when needed


# Detect if running in Google Colab
if "COLAB_GPU" in os.environ:
    ROOT = "/content/divine_semantics"
else:
    ROOT = os.path.dirname(os.path.abspath(__file__))

TRANSLATIONS_REPO = os.path.join(ROOT, "data/inferno_translations_aligned.ods")

EXPERIMENTS_ROOT = os.path.join(ROOT, "experiments")
os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)

# MySQL Config
MYSQL_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

# SQLite Config
SQLITE_DB_PATH = os.path.join(ROOT, "data", "divine_comedy.db")

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
