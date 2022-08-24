#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-24 16:51
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    test_tok.py
# @Project: rasa-3.x-component-examples
# @Package:
# @Ref:
import pytest
# 存储每一个节点的资源
from rasa.shared.nlu.training_data.message import Message
# 当前节点的资源信息
from rasa.engine.storage.resource import Resource
# 在本地磁盘上存储并提供 `GraphComponents`。
from rasa.engine.storage.local_model_storage import LocalModelStorage
# 保存当前单个图形运行的信息。
from rasa.engine.graph import ExecutionContext
# 为实体提取创建特征。
from .nlu_tok import AnotherWhitespaceTokenizer

node_storage = LocalModelStorage("tmp/storage")
node_resource = Resource("tokenizer")
# 表示图中的持久化图组件。output_fingerprint是uuid
context = ExecutionContext(node_storage, node_resource)
# 上下文信息

# 第一种创建的方法
tok_alphanum = AnotherWhitespaceTokenizer.create(
    config={
        "only_alphanum": True,
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_",
    },
    model_storage=node_storage,
    resource=node_resource,
    execution_context=context,
)
# 第二种创建的方法，没有字母数字
tok_no_alphanum = AnotherWhitespaceTokenizer(
    {
        "only_alphanum": False,
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_",
    }
)
# 将有字母和没有字母的拼接成列表
tokenisers = [tok_alphanum, tok_no_alphanum]


@pytest.mark.parametrize("tok", tokenisers)
def test_base_use(tok):
    # Create a message 创建一个message类，并填充数据，属性data下的text被赋值，features属性为空列表，output_properties属性为集合类型{'text'}
    msg = Message({"text": "hello world there"})

    # Process will process a list of Messages Process
    # 将处理消息列表
    tok.process([msg])

    # Check that the message has been processed correctly
    # 检查消息是否已正确处理
    assert [t.text for t in msg.get("text_tokens")] == ["hello", "world", "there"]


def test_specific_behavior():
    # 创建一个message类，并填充数据
    msg = Message({"text": "hello world 12345"})

    tok_no_alphanum.process([msg])
    # 调用不排除数字的情况
    assert [t.text for t in msg.get("text_tokens")] == ["hello", "world", "12345"]

    msg = Message({"text": "hello world #%!#$!#$"})
    # 只保存当前数据，当一行代码的数据没有了哈

    # Process will process a list of Messages
    tok_alphanum.process([msg])
    assert [t.text for t in msg.get("text_tokens")] == [
        "hello",
        "world",
    ]
