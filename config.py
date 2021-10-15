import os

# 项目根目录
root_dir = os.path.dirname(__file__)
# 项目语料原始数据目录
root_corpus_dir = os.path.join(root_dir, 'corpus/origin')
# 处理后数据存放路径
root_outs_dir = os.path.join(root_dir, 'corpus/final')

# 所有ai问答的csv文件
ai_corpus_path = os.path.join(root_corpus_dir, 'ai')


