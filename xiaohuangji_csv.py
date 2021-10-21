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


def create2csv_all_in_one():
    """
    时候文件较小时，一次性读入到内存最后写入文件
    :return:
    """
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
                # 如果question和answer都存在时则保存
                if df_dict['question'] and df_dict['answer']:
                    df = df.append([df_dict])
                flag = 0

    # index=False必须指定，否则生成的文件最左侧会出现index
    df.to_csv(config.xiaohuangji_csv, encoding='utf-8', sep=',', index=False)


def create2csv_block(block=1000):
    """
    当文件较大时，没读入block数据就追加到csv文件中,这样占用内存较小，速度会快很多
    :param block: 每次读入多少行数据
    :return:
    """
    with open(config.xiaohuangji_corpus, encoding='utf-8') as f:
        flag = 0
        count = 0
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
                # 如果question和answer都存在时则保存
                if df_dict['question'] and df_dict['answer']:
                    if count % block == 0:
                        # mode='a': 追加模式, mode='w':覆盖模式
                        # header=False: 不添加coulums
                        df.to_csv(config.xiaohuangji_csv, mode=('a' if count else 'w'),
                                  encoding='utf-8', sep=',', index=False,
                                  header=(False if count else True))
                        df.drop(df.index, inplace=True)
                    df = df.append([df_dict])
                    count += 1
                flag = 0
        # 最终的数据也需要写入
        df.to_csv(config.xiaohuangji_csv, encoding='utf-8',  mode='a', sep=',', index=False, header=False)


if __name__ == '__main__':
    create2csv_block()

