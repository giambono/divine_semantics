import yaml
import pandas as pd
import os

import config

path = r"/home/rfflpllcn/IdeaProjects/divine_semantics/experiments/embeddings/multilingual_e5/embeddings.parquet"
df = pd.read_parquet(path)

print(df)