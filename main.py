#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-26 10:52
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    main.py
# @Project: rasa-3.x-component-examples
# @Package: 
# @Ref:
import os

# TF_CPP_MIN_LOG_LEVEL = 1 // 默认设置，为显示所有信息
# TF_CPP_MIN_LOG_LEVEL = 2 // 只显示error和warining信息
# TF_CPP_MIN_LOG_LEVEL = 3 // 只显示error信息
from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("====pipeline====" * 5)
print("===分词部分===")
from components.nlu_tok import AnotherWhitespaceTokenizer

print(AnotherWhitespaceTokenizer)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer

print(RegexFeaturizer)

print("===特征抽取部分===")
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import LexicalSyntacticFeaturizer

print(LexicalSyntacticFeaturizer)

from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import CountVectorsFeaturizer

print(CountVectorsFeaturizer)

from components.nlu_dense import BytePairFeaturizer

print(BytePairFeaturizer)

from components.nlu_sparse import TfIdfFeaturizer

print(TfIdfFeaturizer)

print("===意图分类部分===")
from components.nlu_clf import LogisticRegressionClassifier

print(LogisticRegressionClassifier)
from rasa.nlu.classifiers.logistic_regression_classifier import LogisticRegressionClassifier

print(LogisticRegressionClassifier)

print("===实体抽取部分===")
from components.nlu_ent import CapitalisedEntityExtractor

print(CapitalisedEntityExtractor)

print()
print()
print("====policies====" * 5)

# MemoizationPolicy
print("===策略部分===")
# 遵循“max_history”的确切示例的策略会在训练故事中发挥作用。
from components.policies.memoization import MemoizationPolicy

print(MemoizationPolicy)
from rasa.core.policies.memoization import MemoizationPolicy

print(MemoizationPolicy)

# RulePolicy
print("===规则策略部分===")
# 处理所有规则的策略。
from components.policies.rule_policy import RulePolicy

print(RulePolicy)
from rasa.core.policies.rule_policy import RulePolicy

print(RulePolicy)

print("===转换embedding到对话部分===")
# TEDPolicy
from components.policies.ted_policy import TEDPolicy

print(TEDPolicy)
# 变压器嵌入对话 （TED） 政策。
from rasa.core.policies.ted_policy import TEDPolicy

print(TEDPolicy)
