"""
defaultdict 与 dict的区别：
dict在查看不存在的key值时会提示KeyError错误
defaultdict在查找key值时如果key不存在，会默认生成一个，默认生成的value值通过参数指定, 参数可以是一个函数
"""

from collections import defaultdict
from pprint import pprint

# 默认 value = 100
dict1 = defaultdict(lambda: 2)
# 默认 value int类型
dict2 = defaultdict(int)

# 默认 value list()
dict3 = defaultdict(list)

print(dict1['key'])
print(dict2['key'])
print(dict3['key'])


"""
将 List[Text] 格式的数据分词, 且词频超过1的变成 List[List[Text]]
"""
text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

frequency = defaultdict(int)
stoplist = set('for a of the and to in'.split(' '))

"""
[['human', 'machine', 'interface', 'lab', 'abc', 'computer', 'applications'],
 ['survey', 'user', 'opinion', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'management', 'system'],
 ['system', 'human', 'system', 'engineering', 'testing', 'eps'],
 ['relation', 'user', 'perceived', 'response', 'time', 'error', 'measurement'],
 ['generation', 'random', 'binary', 'unordered', 'trees'],
 ['intersection', 'graph', 'paths', 'trees'],
 ['graph', 'minors', 'iv', 'widths', 'trees', 'well', 'quasi', 'ordering'],
 ['graph', 'minors', 'survey']]
"""
texts = [[word for word in document.lower().split() if word not in stoplist] for document in text_corpus]

# Count word frequencies
for text in texts:
    for word in text:
        frequency[word] += 1


"""
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
# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

pprint(processed_corpus)
