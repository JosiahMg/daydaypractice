import config
import pandas as pd

import os

df_merge = []

for root, dirs, files in os.walk(config.ai_corpus_path):
    for file in files:
        filename = os.path.join(root, file)
        df = pd.read_csv(filename, encoding='utf-8', header=[0])
        df_merge.append(df)

df_merge = pd.concat(df_merge)

df_merge.to_csv('test.csv', header=['q', 'a'], index=None, encoding='utf-8')
