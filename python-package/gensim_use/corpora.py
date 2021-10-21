"""
corpora模块用于对语料库构造一个bag-of-word模型
dictionary = corpora.Dictionary(List[List[Text]])

dictionary.token2id: 词典, word: index -> Dict[Text, int]
dictionary.doc2bow(List[List[Text]]): 生成词袋[(index, counter)] -> List[List[Tuple[int, int]]]

"""

from pprint import pprint
from gensim import corpora

processed_corpus = [['human', 'interface', 'computer'],
                    ['survey', 'user', 'computer', 'system', 'response', 'time'],
                    ['eps', 'user', 'interface', 'system'],
                    ['system', 'human', 'system', 'eps'],
                    ['user', 'response', 'time'],
                    ['trees'],
                    ['graph', 'trees'],
                    ['graph', 'minors', 'trees'],
                    ['graph', 'minors', 'survey']]

dictionary = corpora.Dictionary(processed_corpus)

"""
Dict[Text, int]

{'computer': 0, 'human': 1, 'interface': 2, 'response': 3, 'survey': 4, 'system': 5,
 'time': 6, 'user': 7, 'eps': 8, 'trees': 9, 'graph': 10, 'minors': 11 }
"""
pprint(dictionary.token2id)


new_doc = "Human human computer interaction"

new_vec = dictionary.doc2bow(new_doc.lower().split())

# [(0, 1), (1, 2)]
print(new_vec)
