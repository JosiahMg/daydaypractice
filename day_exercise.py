import os
import json
import numpy as np
from pprint import pprint
from typing import List, Text


candidate_file = 'candidates.json'
base_dirs = ['data-processing', 'models', 'python-package']
ignore_file = ['config.py', 'main.py']

# 初始化每个文件的权重
def init_candidate(all_files):
    num = len(all_files)
    value = 1.0/num
    return {file: value for file in all_files}


# 加载候选名单,files为当前总文件列表
def load_candidate(all_files: List[Text]):
    new_lists = []
    new_candidates = {}
    if not os.path.exists(candidate_file):
        new_candidates = init_candidate(all_files)
    else:
        candidates = json.load(open(candidate_file, 'r'))
        # 首先遍历词典，找到已经不存在的文件的key并删掉
        discard_weight = 0.0
        for key, value in candidates.items():
            if key not in all_files:
                discard_weight += value
        each_discard_weight = discard_weight/len(all_files)
        # 遍历所有文件，看看是否有新增加的文件

        for file in all_files:
            if file not in candidates:
                new_candidates[file] = each_discard_weight
                new_lists.append(file)
            else:
                new_candidates[file] = candidates[file] + each_discard_weight
    return new_lists, new_candidates


# 保存pkl文件
def save_load_candidate(candidates):
    json.dump(candidates, open(candidate_file, 'w'))


# 纠正candidate的权重
def update_candidate_weights(chosen_files, all_files, candidates):
    weights = 0.0
    num_files = len(all_files) - len(chosen_files)
    for file in chosen_files:
        weights += candidates[file]
        candidates[file] = 0

    each_weight = weights/num_files
    for key, _ in candidates.items():
        if key not in chosen_files:
            candidates[key] += each_weight


def get_all_files():
    candidate_files = []
    for dir in base_dirs:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.py') and file not in ignore_file:
                    candidate_files.append(os.path.join(root, file))
    return candidate_files


def candidate_weight_sum():
    if not os.path.exists(candidate_file):
        pprint('Candidate json file not exists')
    else:
        candidates = json.load(open(candidate_file, 'r'))
        print(np.sum(list(candidates.values())))

# 测试使用打印信息
def show_candidate_info():
    if not os.path.exists(candidate_file):
        pprint('Candidate json file not exists')
    else:
        candidates = json.load(open(candidate_file, 'r'))
        infos = zip(list(candidates.keys()), list(candidates.values()))
        infos = sorted(infos, key=lambda x: -x[1])
        pprint(infos)


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
        choice_file.append(np.random.choice(files, p=weights))
        update_candidate_weights(choice_file, all_files, candidates)
    save_load_candidate(candidates)
    return choice_file


if __name__ == '__main__':
    files = main_process()
    print(files)
    # print('-------Test--------')
    # candidate_weight_sum()
    # show_candidate_info()
