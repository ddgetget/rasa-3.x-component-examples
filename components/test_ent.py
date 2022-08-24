#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-24 20:44
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    test_ent.py
# @Project: rasa-3.x-component-examples
# @Package:
# @Ref:

import pytest
import pathlib

# 内置通信类
from rasa.shared.nlu.training_data.message import Message
# 当前组件资源类
from rasa.engine.storage.resource import Resource
# 当前资源存储类
from rasa.engine.storage.local_model_storage import LocalModelStorage
# 上下文类
from rasa.engine.graph import ExecutionContext
# 分词
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

# 自定义实体抽取类
from .nlu_ent import CapitalisedEntityExtractor


@pytest.fixture
def entity_extractor(tmpdir):
    """Generate a tfidf vectorizer with a tmpdir as the model storage."""
    # 生成 tfidf存储到tmpdir
    # 存储节点，/private/var/folders/z0/7kqx00cx0pq1dyl32g8z02cm0000gn/T/pytest-of-geng/pytest-17/test_sparse_feats_added_hello_0
    node_storage = LocalModelStorage(pathlib.Path(tmpdir))
    # 资源节点
    node_resource = Resource("sparse_feat")
    # 当前当前执行组件的上下文
    context = ExecutionContext(node_storage, node_resource)
    # 返回一个实体抽取的对象
    return CapitalisedEntityExtractor(
        config=CapitalisedEntityExtractor.get_default_config(),
        name=context.node_name,  # 这个值在任何逐渐都是None，目前还不清楚干嘛的》？？？？？？？
        resource=node_resource,
        model_storage=node_storage,
    )


tokeniser = WhitespaceTokenizer(
    {
        "only_alphanum": False,
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_",
    }
)


@pytest.mark.parametrize(
    "text, expected",
    [("hello World", ["World"]), ("Hello world", ["Hello"]), ("hello there world", [])],
)
def test_sparse_feats_added(entity_extractor, text, expected):
    """Checks if the sizes are appropriate."""
    # Create a message
    # 跟message.set(key,value,add_to_output=True)是一样的效果
    msg = Message({"text": text})

    # Process will process a list of Messages
    tokeniser.process([msg])
    entity_extractor.process([msg])
    # Check that the message has been processed correctly
    entities = msg.get("entities")
    # text: hello World entities： [{'entity': 'THING', 'start': 6, 'end': 11, 'value': 'World', 'confidence': 1.0}]
    # text: hello World entities： [{'entity': 'THING', 'start': 6, 'end': 11, 'value': 'World', 'confidence': 1.0}]
    # text: hello there world entities： []
    print("text:", text, "entities：", entities)
    assert [e["value"] for e in entities] == expected
