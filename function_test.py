#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-24 16:51
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    function_test.py
# @Project: rasa-3.x-component-examples
# @Package: 
# @Ref:


def only_alphanum_parse_test(text):
    result = "".join([c for c in text if ((c == " ") or str.isalnum(c))])
    # str.isalnum(c): 如果字符串是字母数字字符串，则返回 True，否则返回 False。
    print(result)
    return result


def sklearn_LogisticRegression_test():
    """
    :return: 逻辑回归（又名 logit，MaxEnt）分类器。
    """
    print("********************sklearn_LogisticRegression_test***********************")
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0, max_iter=500).fit(X, y)
    # max_iter设置大一的，或者缩小数据集，不然会报错
    result = clf.predict(X[:2, :])
    print(result)
    result_proba = clf.predict_proba(X[:2, :])
    print(result_proba)
    result_log_proba = clf.predict_log_proba((X[:2, :]))
    print(result_log_proba)

    result_socre = clf.score(X, y)
    print(result_socre)


def sklearn_TfidfVectorizer_test():
    print("********************TfidfVectorizer***********************")
    """
    :return: 将原始文档集合转换为 TF-IDF 特征矩阵。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    # 学习词汇和 idf，返回文档术语矩阵。
    print(vectorizer.get_feature_names())
    print(X.shape)


def bpemb_BPEmb_test():
    """
    :return: 一个 BPEmb 模型和与之交互的实用函数。
    """

    from bpemb import BPEmb
    # Load a BPEmb model for English:
    bpemb_en=BPEmb(lang="en")


    # "Load a BPEmb model for Chinese and choose the vocabulary size (vs),that is, the number of byte-pair symbols:"
    bpemb_zh=BPEmb(lang="zh", vs=100000)

    # "Choose the embedding dimension:",
    bpemb_es=BPEmb(lang="es", vs=50000, dim=300)

    print("Byte-pair encode text:", bpemb_en.encode("stratford"))

    print("This is anarchism", bpemb_en.encode("This is anarchism"))

    print("这是一个中文句子", bpemb_zh.encode("这是一个中文句子"))

    # Byte-pair encode text into IDs for performing an embedding lookup:
    ids = bpemb_zh.encode_ids("这是一个中文句子")
    print("这是一个中文句子的ids", ids)

    print("bpemb_zh的vectors 的shape", bpemb_zh.vectors.shape)

    embedded = bpemb_zh.vectors[ids]
    print("bpemb_zh embedded shape", embedded.shape)

    # Byte-pair encode and embed text:
    print("No entendemos por qué. 的embed的 shape", bpemb_es.embed("No entendemos por qué.").shape)

    print("Decode byte-pair-encoded text:", bpemb_en.decode(['▁this', '▁is', '▁an', 'arch', 'ism']))

    print("The encode-decode roundtrip is lossy:", bpemb_en.decode(bpemb_en.encode("This is anarchism 101")))

    print("This is due to the preprocessing being applied before encoding:",
          bpemb_en.preprocess("This is anarchism 101"))

    print("Decode byte-pair IDs:", bpemb_zh.decode_ids([25950, 695, 20199]))

if __name__ == '__main__':
    # only_alphanum_parse_test(text="hello world")
    # sklearn_LogisticRegression_test()
    # sklearn_TfidfVectorizer_test()
    bpemb_BPEmb_test()
