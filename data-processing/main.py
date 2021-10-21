import pandas as pd
import config


df = pd.read_csv(config.tianchi_news_train_path, encoding='utf-8', nrows=10, header=[0], sep='\t')
print(df.columns)
df['label_str'] = '__label__' + df['label'].astype(str)

df[['label_str', 'text']].to_csv('test.csv', encoding='utf-8', index=False, sep='\t')
