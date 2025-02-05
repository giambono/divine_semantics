import yaml
import pandas as pd
import os

import config


with open(os.path.join(config.ROOT, "config_nlp.yaml"), "r") as f:
    config = yaml.safe_load(f)

fpath = os.path.join(ROOT_DIR, "out/test_set.pkl")

df = pd.read_pickle(fpath)

df.to_clipboard()