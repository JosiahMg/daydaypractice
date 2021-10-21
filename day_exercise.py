"""
每天随机选择一个程序进行练习
"""

import os
import json
import numpy as np
from pprint import pprint
from typing import List, Text

# 保存复习进度的文件
candidate_file = 'candidates.json'
# 需要复习的文件夹
base_dirs = ['data-processing', 'models', 'python-package']
# 忽略不需要复习的文件
ignore_file = ['config.py', 'main.py']
# 需要复习的次数
exercise_times = 100
# 每次复习的个数
choice_size = 1


def init_candidate(all_files):
    """
    初始化每个文件的权重
    foramt:
    {
        filename: {
            "probability": float,
            "counter": int
        }
    }
    """
    num = len(all_files)
    value = 1.0/num
    return {file: {'probability': value, 'counter': 0} for file in all_files}


# 加载候选名单,files为当前总文件列表
def load_candidate(all_files: List[Text]):
    new_lists = []
    if not os.path.exists(candidate_file):
        candidates = init_candidate(all_files)
    else:
        candidates = json.load(open(candidate_file, 'r'))
        # 首先遍历词典，找到已经不存在的文件的key并删掉
        discard_weight = 0.0
        discard_files = []
        for key, value in candidates.items():
            if key not in all_files:
                discard_weight += value['probability']
                discard_files.append(key)
        each_discard_weight = discard_weight/len(all_files)

        # 删除不存在的文件的key
        for file in discard_files:
            del candidates[file]

        # 遍历所有文件，看看是否有新增加的文件
        for file in all_files:
            if file not in candidates:
                candidates[file] = {'probability': each_discard_weight, 'counter': 1}
                new_lists.append(file)
            else:
                candidates[file]['probability'] = candidates[file]['probability'] + each_discard_weight
    return new_lists, candidates


# 保存pkl文件
def save_load_candidate(candidates):
    json.dump(candidates, open(candidate_file, 'w'), ensure_ascii=False, indent=2)


# 更新candidate的权重
def update_candidate_weights(chosen_files, all_files, candidates):
    weights = 0.0
    num_files = len(all_files) - len(chosen_files)
    for file in chosen_files:
        weights += candidates[file]['probability']
        candidates[file]['probability'] = 0.0
        if candidates[file]['counter'] != exercise_times:
            candidates[file]['counter'] += 1

    each_weight = weights/num_files
    for key, _ in candidates.items():
        if key not in chosen_files:
            candidates[key]['probability'] += each_weight
    return candidates


def get_all_files():
    candidate_files = []
    for dir in base_dirs:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.py') and file not in ignore_file:
                    candidate_files.append(os.path.join(root, file))
    return candidate_files


# 统计所有文件的概率值是否为1
def candidate_weight_sum():
    if not os.path.exists(candidate_file):
        pprint('Candidate json file not exists')
    else:
        candidates = json.load(open(candidate_file, 'r'))
        print(np.sum([weight['probability'] for weight in list(candidates.values())]))


# 测试使用打印信息
def show_candidate_info():
    if not os.path.exists(candidate_file):
        pprint('Candidate json file not exists')
    else:
        candidates = json.load(open(candidate_file, 'r'))
        infos = zip(list(candidates.keys()), list(candidates.values()))
        infos = sorted(infos, key=lambda x: -x[1]['counter'])
        pprint(infos)


def softmax(x: List[int]):
    y = np.exp(x-np.max(x))
    fx = y / np.sum(y)
    return fx


# 获取random.choice的概率分布
def get_choice_probability(values):
    probas = []
    counters = []
    for value in values:
        probas.append(value['probability'])
        counters.append(exercise_times - value['counter'])

    counters = softmax(counters)
    result = [(pro/2.+counter/2.) for pro, counter in zip(probas, counters)]
    return result


def main_process():
    all_files = get_all_files()
    new_lists, candidates = load_candidate(all_files)
    choice_file =[]
    if new_lists:
        choice_file.extend(new_lists)
    else:
        files = []
        weights = []
        for key, value in candidates.items():
            files.append(key)
            weights.append(value)
        weights = get_choice_probability(weights)
        choice_file.extend(np.random.choice(files, size=choice_size, replace=False, p=weights))

        candidates = update_candidate_weights(choice_file, all_files, candidates)

    save_load_candidate(candidates)
    return [(file, candidates[file]['counter']) for file in choice_file]


if __name__ == '__main__':
    files = main_process()
    print(files)
    # print('-------Test--------')
    # candidate_weight_sum()
    # show_candidate_info()
