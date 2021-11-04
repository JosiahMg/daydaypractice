"""
功能说明:
将文件夹 corpus/origin/*.csv (config.ai_corpus_path) 合并到一个json文件

输入数据格式:
csv的数据格式为QA数据:
colums = [question:Text , answer:Text]

输出数据格式:
json数据格式:
format:
        {
            "what is python": {
                                "answer": "python is ...",
                                "q_cut": ['what', 'is', 'python'],
                                "entity": ['python']
                               },
            ...
        }

用途:
可以用于根据索引获取问答的答案，在QA对话系统的召回中会使用到
"""

import os
import config
import pandas as pd
import jieba
import json

# 合并后的ai问答csv文件
ai_merge_path = os.path.join(config.root_outs_dir, 'ai_merge.json')


qa_dict = {}
for file in os.listdir(config.ai_corpus_path):
    filename = os.path.join(config.ai_corpus_path, file)
    df = pd.read_csv(filename, encoding='utf-8', sep=',', header=[0])
    # process for  json
    for q, a in zip(df['question'], df['answer']):
        qa_dict[q] = {}
        qa_dict[q]['answer'] = a
        qa_dict[q]['q_cut'] = jieba.lcut(q)

# ensure_ascii=False: 保存的字符不使用ascii进行编码
# indent: 换行后缩进
json.dump(qa_dict, open(ai_merge_path, mode='w', encoding='utf-8'), ensure_ascii=False, indent=2)
