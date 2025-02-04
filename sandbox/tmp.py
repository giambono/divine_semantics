import yaml
import pandas as pd
import os

import sys
from pathlib import Path

from src.find_similarity import find_most_similar_ensemble

ROOT_DIR = Path(__file__).resolve().parent.parent  # Adjust if needed
sys.path.append(str(ROOT_DIR))


with open(os.path.join(ROOT_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

fpath = os.path.join(ROOT_DIR, "out/test_set.pkl")

df = pd.read_pickle(fpath)

df.to_clipboard()