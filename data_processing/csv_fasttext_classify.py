"""
功能说明:
config.tianchi_news_train_path
将corpus/origin/tianchi/train_set.csv 生成可以用于fasttext进行文本分的数据集

输入数据格式:
csv的数据格式为:
colums = [label:int \t text: Text]

输出数据格式:
[text: Text \t __label__num: Text]

用途:
用于fasttext.train_supervised()训练分类任务

"""

import pandas as pd
import config

# nrows: 计算方便只加载100行
# skiprows: =0: 忽略前n行, [0] 忽略第0行
# header=[0]: 0行为 columns
df_all = pd.read_csv(config.tianchi_news_train_path, sep='\t',
                     nrows=100, skiprows=0, header=[0])  # key: sep='\t'
print(df_all[:5])

# df_all['label']为int类型，转换成str类型
df_all['label_ft'] = '__label__' + \
    df_all['label'].astype(str)  # key: astype(str)

# df_all['label_ft'] = df_all['label'].apply(lambda x: '__label__'+str(x))

df_all[['text', 'label_ft']].to_csv(
    config.tianchi_news_fasttext_classify, index=False, header=None, sep='\t')
