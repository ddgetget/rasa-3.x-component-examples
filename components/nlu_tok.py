#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-24 16:51
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    nlu_tok.py
# @Project: rasa-3.x-component-examples
# @Package:
# @Ref:
from __future__ import annotations
from typing import Any, Dict, List, Optional, Text

from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class AnotherWhitespaceTokenizer(Tokenizer):
    """Creates features for entity extraction."""

    # 为实体提取创建特征。

    @staticmethod
    def not_supported_languages() -> Optional[List[Text]]:
        """The languages that are not supported."""
        return ["zh", "ja", "th"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            # This *must* be added due to the parent class.
            "intent_tokenization_flag": False,
            # This *must* be added due to the parent class.
            "intent_split_symbol": "_",
            # This is a, somewhat silly, config that we pass
            "only_alphanum": True,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize the tokenizer."""
        super().__init__(config)
        self.only_alphanum = config["only_alphanum"]

    def parse_string(self, s):
        # 分词针对字母的处理
        if self.only_alphanum:
            # 如果仅仅需要字母
            return "".join([c for c in s if ((c == " ") or str.isalnum(c))])  # 针对每一个字母进行筛选，针对是空格和字母的直接拼接
        return s

    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> AnotherWhitespaceTokenizer:
        # 创建一个新组件（请参阅父类以获取完整的文档字符串）
        return cls(config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = self.parse_string(message.get(attribute))
        # 解析字符串，根据系统设置是否解析数字空格等
        words = [w for w in text.split(" ") if w]
        # 按空格分割，即分词效果，形成列表

        # if we removed everything like smiles `:)`, use the whole text as 1 token
        if not words:
            # 如果没有分割出来，但是需要列表类型，整个字符串当成列表的唯一值
            words = [text]

        # the ._convert_words_to_tokens() method is from the parent class.
        tokens = self._convert_words_to_tokens(words, text)
        # 将单词列表转换成token，通过words里面每一个word去索引text，进而获得每个单词索引，即为单词对应的ID，重复的看第一个，这里是列表形式，里面每一个是Token类型

        return self._apply_token_pattern(tokens)  # 将令牌模式应用于给定的令牌。对上面的重新写了一遍，将所有的token排成一列，并记录了每一恶搞token的begin和end
