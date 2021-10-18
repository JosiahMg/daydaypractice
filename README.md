# 目录介绍

##  corpus
存放语料库

### final
处理后的数据
### origin
存放原始语料库  
[数据下载地址](https://pan.baidu.com/s/1-XbbP2O2n4OaTYG4z8yNvw)
提取码：y2k5
#### ai
人工智能问答语料库

# 代码简介

## csv_merge_csv
将文件夹 corpus/origin/ai/*.csv 合并到一个csv文件

## csv_merge_json
将文件夹 corpus/origin/ai/*.csv 合并到一个json文件

## csv_fasttext_classify
将 corpus/origin/tianchi_news/train_set.csv文件生成用于fasttext分类任务格式的训练数据

## corpus_stream
使用__iter__将数据存放迭代器，减少内存占用

## xiaohuangji_csv
将小黄鸡语料生成csv格式的文件用于assistant项目的语料