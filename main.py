import config
import os
import numpy as np
import pandas as pd
from time import time


t1 = time()

def func1():
    with open('corpus/origin/estate/train.reply.tsv', encoding='utf-8') as f:
        while True:
            yield f.readline()

d1 = func1()
for d in d1:
    print(d)
print(time()-t1)


t1 = time()

def func2():
    for line in open('corpus/origin/estate/train.reply.tsv', encoding='utf-8'):
        yield line

d2 = func2()

print(time()-t1)

