"""
功能说明:
将文件夹 corpus/origin/ai/*.csv 合并到一个csv文件

输入数据格式:
csv的数据格式为QA数据:
colums = [question  answer]

输出数据格式:
csv的数据格式为QA数据:
colums = [question  answer]

用途:
合并多个相同格式的数据到一个文件中
"""

import os
import config
import pandas as pd

# 合并后的ai问答csv文件
ai_merge_path = os.path.join(config.root_outs_dir, 'ai_merge.csv')


df_merge = []

# for file in os.listdir(config.ai_corpus_path):
for root, dirs, files in os.walk(config.ai_corpus_path):
    for file in files:
        filename = os.path.join(root, file)
        # header=[0] 表示第一行为columns
        # sep = ',' 表示分隔符为 ,
        df = pd.read_csv(filename, header=[0], encoding='utf-8', sep=',')
        df_merge.append(df)

df_merge = pd.concat(df_merge)
# 生成 merge_df.csv文件
df_merge.to_csv(ai_merge_path, header=['question', 'answer'], index=False)



