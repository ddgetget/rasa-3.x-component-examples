#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-24 16:51
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    test_dense.py
# @Project: rasa-3.x-component-examples
# @Package:
# @Ref:
import pytest
# 可用于描述对话轮次的数据容器。
from rasa.shared.nlu.training_data.message import Message
# 表示图中的持久化图组件。
from rasa.engine.storage.resource import Resource
# 在本地磁盘上存储并提供 `GraphComponents` 的输出。
from rasa.engine.storage.local_model_storage import LocalModelStorage
# 保存有关单个图形运行的信息。
from rasa.engine.graph import ExecutionContext
# 为实体提取创建特征。
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

# 自定义稠密特征组件
from .nlu_dense import BytePairFeaturizer

# 存储节点
node_storage = LocalModelStorage("tmp/storage")
# 资源节点
node_resource = Resource("tokenizer")
# 当前节点上下文
context = ExecutionContext(node_storage, node_resource)

# 分词，可以含有数字 【__init__(), create(),load()】
tokeniser = WhitespaceTokenizer(
    {
        "only_alphanum": False,
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_",
    }
)

# 自定义稠密特征组件，为什么不需要resource和storage，很奇怪？？？？？ 原来是因为这里不需要训练，所以资源和存储部分不需要
bpemb_feat = BytePairFeaturizer(
    config={
        "lang": "en",
        "dim": 25,
        "vs": 1000,
        "alias": "foobar",  # 别名
        "vs_fallback": True,
    },
    name=context.node_name,
)


# :class:`MarkDecorator` 对象的工厂 - 暴露为一个``pytest.mark`` 单例实例。
@pytest.mark.parametrize(
    "text, expected", [("hello", 1), ("hello world", 2), ("hello there world", 3)]
)
def test_dense_feats_added(text, expected):
    """Checks if the sizes are appropriate."""
    # Create a message
    # 创建一个message对象
    msg = Message({"text": text})

    # Process will process a list of Messages
    # 分词
    tokeniser.process([msg])
    # 特征表示，传给message，系统还需要和之前的特征值在列的方向商进行拼接
    bpemb_feat.process([msg])

    # Check that the message has been processed correctly
    # 序列特征，句子特征，
    # np.concatenate((self.features, additional_features.features), axis=-1)
    # axis=0:（5，4）+（2，4）==》（7，4）简言之：行增加了；或者说在第一个中括号上添加元素
    # axis=1:（5，4）+（5，2）==》（5，6）简言之：列增加了；或者说在第二个中括号上添加元素
    # axis=-1:（5，4）+（5，2）==》（5，6）简言之：列增加了；或者说在第二个中括号上添加元素
    # 这个代码因为中间的message，所以跨度比较大
    seq_feats, sent_feats = msg.get_dense_features("text")
    print(seq_feats, sent_feats)
    # 断言序列特征的shape是不是(序列个数，25）
    assert seq_feats.features.shape == (expected, 25)
    # 断言句子特征的shape是不是（1，25）
    assert sent_feats.features.shape == (1, 25)
