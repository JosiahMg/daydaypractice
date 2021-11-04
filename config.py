import os

# 项目根目录
root_dir = os.path.dirname(__file__)
# 项目语料原始数据目录
root_corpus_dir = os.path.join(root_dir, 'corpus/origin')
# 处理后数据存放路径
root_outs_dir = os.path.join(root_dir, 'corpus/final')

# 所有ai问答的csv文件
ai_corpus_path = os.path.join(root_corpus_dir, 'ai')

# 天池新闻脱敏数据集
tianchi_news_root_path = os.path.join(root_corpus_dir, 'tianchi_news')
tianchi_news_train_path = os.path.join(tianchi_news_root_path, 'train_set.csv')
tianchi_news_test_a_path = os.path.join(tianchi_news_root_path, 'test_a.csv')
tianchi_news_test_b_path = os.path.join(tianchi_news_root_path, 'test_b.csv')
# 生成的文件用于fasttext分类训练
tianchi_news_fasttext_classify = os.path.join(root_outs_dir, 'train_fasttext.csv')


# corpus in one time
gensim_corpus = os.path.join(root_corpus_dir, 'gensim_corpus.txt')

# 小黄鸡
xiaohuangji_corpus = os.path.join(root_corpus_dir, 'xiaohuangji.conv')
# 处理后的小黄鸡
xiaohuangji_csv = os.path.join(root_outs_dir, 'xiaohuangji.csv')

# name
name_train_path = os.path.join(root_corpus_dir, 'name/names_train.csv')
name_test_path = os.path.join(root_corpus_dir, 'name/names_test.csv')

# 通过字典的形式生成csv文件
dict_csv_path = os.path.join(root_outs_dir, 'dict_csv.csv')

# log.conf
log_conf = os.path.join(root_dir, 'log/log.conf')

# wine data
wine_path = os.path.join(root_corpus_dir, 'wine.csv')

# imdb sentiment pos and neg
imdb_train_path = os.path.join(root_corpus_dir, 'aclImdb/train')
imdb_test_path = os.path.join(root_corpus_dir, 'aclImdb/test')
imdb_vocab = os.path.join(root_outs_dir, 'imdb.vocab')

# user_dict and stopwords
stopword_path = os.path.join(root_dir, 'data-processing/stopwords/stopword.txt')
userdict_path = os.path.join(root_dir, 'data-processing/userdicts/userdict.txt')

# math
math_path = os.path.join(root_corpus_dir, 'math')
train_math_path = os.path.join(math_path, 'train.ape.json')
test_math_path = os.path.join(math_path, 'test.ape.json')
valid_math_path = os.path.join(math_path, 'valid.ape.json')

math_solve_path = os.path.join(root_outs_dir, 'math_solve.csv')