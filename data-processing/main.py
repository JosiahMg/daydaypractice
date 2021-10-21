import pandas as pd
import config
import os
from pprint import pprint

df_merge = {}

for root, dirs, files in os.walk(config.ai_corpus_path):
    for file in files:
        filenames = os.path.join(root, file)
        df = pd.read_csv(filenames, encoding='utf-8', header=[0])
        for q, a in zip(df['question'], df['answer']):
            df_merge[q] = {}
            df_merge[q]['answer'] = a
            df_merge[q]['entity'] = []

pprint(df_merge)

