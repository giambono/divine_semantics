import os
import pandas as pd

import config

path = os.path.join(config.ROOT, "data/paraphrased_verses.parquet")
test_queries = pd.read_parquet(path)

print()