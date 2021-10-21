"""
用途:
Counter是一个dict子类，主要是用来对你访问的对象的频率进行计数。
通常用于构造词典

常用方法：
elements()：返回一个迭代器，每个元素重复计算的个数，如果一个元素的计数小于1,就会被忽略。
most_common([n])：返回一个列表，提供n个访问频率最高的元素和计数
subtract([iterable-or-mapping])：从迭代对象中减去元素，输入输出可以是0或者负数
update([iterable-or-mapping])：从迭代对象计数元素或者从另一个 映射对象 (或计数器) 添加。
clear(): 清空
"""

from collections import Counter
import nltk

MAX_VOCAB_SIZE = 20

text = """Our legislation on participating in a bid is clear: no one can be prohibited, 
          said Mourao, adding the only thing the company must do is to demonstrate its 
          transparency (in keeping) with the rules that will be established for the process."""

text_token = nltk.word_tokenize(text)

# return: Counter(Dict[Text, int])
counter = Counter(text_token)

# return: iter  elems: List[int]
elems = list(counter.elements())

# return: List[Tuple[Text, int]]
ls = counter.most_common(MAX_VOCAB_SIZE)

# 构造词典, index从 2开始
vocab = {w[0]: index for index, w in enumerate(ls, 2)}
print(vocab)


# 创建方法2
counter2 = Counter()
for word in text_token:
    counter2[word] += 1

# 创建方法3
counter3 = Counter({'hello': 2, 'world': 3})

# 创建方法4
counter4 = Counter(hello=2, world=3)
