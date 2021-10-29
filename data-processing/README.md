# 代码简介

## corpus_stream
使用__iter__将数据存放迭代器，减少内存占用

## create_vocab.py
使用 name 数据集构造词典

## csv_fasttext_classify
将 corpus/origin/tianchi_news/train_set.csv文件生成用于fasttext分类任务格式的训练数据

## csv_merge_csv
将文件夹 corpus/origin/ai/*.csv 合并到一个csv文件

## csv_merge_json
将文件夹 corpus/origin/ai/*.csv 合并到一个json文件

## dataset_wine.py
使用 wine.csv 构造torch.Dataset

## jieba_tokeinzer.py
jieba分词器的使用

## vocab_template.py
使用 aclImdb 构造vocab

## xiaohuangji_csv
将xiaohuangji.conv小黄鸡语料生成csv格式的文件用于assistant项目的语料

