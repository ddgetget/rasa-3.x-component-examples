#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-24 20:44
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    nlu_ent.py
# @Project: rasa-3.x-component-examples
# @Package:
# @Ref:

import logging
from typing import Any, Text, Dict, List, Type
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    TEXT_TOKENS,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITIES,
)

# 返回具有指定名称的记录器，必要时创建它。
logger = logging.getLogger(__name__)


# 注册组件
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False
)
class CapitalisedEntityExtractor(EntityExtractorMixin, GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        # 添加上游组件，上游是token，不需要特征化哈，但也不唯一
        return [Tokenizer]

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        # 添加需要的包，当前默许爱，全部原创，所以不需要包
        return []

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        # 设置默认参数
        return {"entity_name": "THING"}

    def __init__(
            self,
            config: Dict[Text, Any],
            name: Text,
            model_storage: ModelStorage,
            resource: Resource,
    ) -> None:
        # 获取配置参数
        self.entity_name = config.get("entity_name")

        # We need to use these later when saving the trained component.
        # 我们需要稍后在保存经过训练的组件时使用这些。本项目是多余的代码
        self._model_storage = model_storage
        self._resource = resource

    def train(self, training_data: TrainingData) -> Resource:
        """
        这个方法这里是无效的！！！！！！！！！！！！！！！！
        :param training_data:
        :return:
        """

        # 根据训练数据进行，构建数据集及标签，目前存在怀疑，从何而来？？？？？这个也是没有用的
        X, y = self._create_training_matrix(training_data)
        # 这里是伪装的，没有这样的东西
        self.clf.fit(X, y)
        # 存储也没用，使用的是规则，而且判断大写字母开头就是thing标签，
        self.persist()

        return self._resource

    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, execution_context.node_name, model_storage, resource)

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            # 设置每一条message的实体信息到message
            self._set_entities(message)
        return messages

    def _set_entities(self, message: Message, **kwargs: Any) -> None:
        tokens: List[Token] = message.get(TEXT_TOKENS)
        extracted_entities = []  # 存储抽取的实体
        for token in tokens:
            if token.text[0].isupper():  # 如果字符串是大写字符串，则返回 True，否则返回 False。是大写，世界判断他是Thing标签，哈哈，这个实体抽取有意思哈
                # ("hello World", ["World"]）
                # {'entity': 'THING', 'start': 6, 'end': 11, 'value': 'World', 'confidence': 1.0}
                extracted_entities.append(
                    {
                        ENTITY_ATTRIBUTE_TYPE: self.entity_name,
                        ENTITY_ATTRIBUTE_START: token.start,
                        ENTITY_ATTRIBUTE_END: token.end,
                        ENTITY_ATTRIBUTE_VALUE: token.text,
                        "confidence": 1.0,  # 这里使用规则，所以置信度都为1
                    }
                )
        # 在data属性里面增加了entities，是一个列表；[{'entity': 'THING', 'start': 6, 'end': 11, 'value': 'World', 'confidence': 1.0}]
        # 将消息的属性设置为给定值。一句话可能是多个实体，所以是列表
        # message.get(ENTITIES, []) + extracted_entities意思是可以有多组实体抽取的方案
        message.set(ENTITIES, message.get(ENTITIES, []) + extracted_entities, add_to_output=True)

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """"
        无效方法！！！！！！！！！！！
        """
        self.process(training_data.training_examples)
        return training_data

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        """"
        无效方法！！！！！！！！！！！！！"""
        pass
