"""
功能描述:
1. 实现迭代器类
Corpus Streaming – One Document at a Time
通过迭代类型的class，防止一次性读取内存过多数据,使用迭代器解决,一次读取一个document

输入数据格式: config.gensim_corpus  (gensim_corpus.txt)
英文文本，通过空格隔开

2. 使用gensim的corpora类生成词典
语料：
[['human', 'interface', 'computer'],
                    ['survey', 'user', 'computer', 'system', 'response', 'time'],
                    ['eps', 'user', 'interface', 'system'],
                    ['system', 'human', 'system', 'eps'],
                    ['user', 'response', 'time'],
                    ['trees'],
                    ['graph', 'trees'],
                    ['graph', 'minors', 'trees'],
                    ['graph', 'minors', 'survey']]

"""
import config
from smart_open import open  # for transparently opening remote files
from gensim import corpora

corpus_file = config.gensim_corpus

processed_corpus = [['human', 'interface', 'computer'],
                    ['survey', 'user', 'computer', 'system', 'response', 'time'],
                    ['eps', 'user', 'interface', 'system'],
                    ['system', 'human', 'system', 'eps'],
                    ['user', 'response', 'time'],
                    ['trees'],
                    ['graph', 'trees'],
                    ['graph', 'minors', 'trees'],
                    ['graph', 'minors', 'survey']]

# 生成gensim的词典类型
dictionary = corpora.Dictionary(processed_corpus)

# 迭代器，读取语料库


class MyCorpus:
    def __iter__(self):  # KEYPOINT
        # for line in open(corpus_file):
        for line in open('https://radimrehurek.com/gensim_3.8.3/auto_examples/core/mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())  # KEYPOINT


corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
print(corpus_memory_friendly)

"""
    [(0, 1), (1, 1), (2, 1)]
    [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]
    [(2, 1), (5, 1), (7, 1), (8, 1)]
    [(1, 1), (5, 2), (8, 1)]
    [(3, 1), (6, 1), (7, 1)]
    [(9, 1)]
    [(9, 1), (10, 1)]
    [(9, 1), (10, 1), (11, 1)]
    [(4, 1), (10, 1), (11, 1)]
"""
for vector in corpus_memory_friendly:  # load one vector into memory at a time
    print(vector)
