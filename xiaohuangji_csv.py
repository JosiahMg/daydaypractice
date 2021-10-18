"""
功能说明:
将小黄鸡语料合并成QA格式的数据

输入数据格式:
conv的数据格式为:(E:分隔, M1: Question M2:Answer)

E
M 呵呵
M 是王若猫的。
E
M 不是
M 那是什么？
E

输出数据格式: csv格式
QA形式存储

用途:
用于assistant项目的数据集

"""

import config
import pandas as pd
from tqdm import tqdm


with open(config.xiaohuangji_corpus, encoding='utf-8') as f:
    flag = 0
    df = pd.DataFrame(columns=['question', 'answer'])
    df_dict = {}
    for line in tqdm(f.readlines()):
        if line.startswith('E'):
            continue
        elif line.startswith('M') and flag == 0:
            df_dict['question'] = line[2:].strip()
            flag = 1
        else:
            df_dict['answer'] = line[2:].strip()
            df = df.append([df_dict])
            flag = 0

# index=False必须指定，否则生成的文件最左侧会出现index
df.to_csv(config.xiaohuangji_csv, encoding='utf-8', sep=',', index=False)
