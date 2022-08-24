# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-24 16:51
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    nlu_dense.py
# @Project: rasa-3.x-component-examples
# @Package:
# @Ref:
import numpy as np
import logging
from bpemb import BPEmb
from typing import Any, Text, Dict, List, Type
# 将正常模型配置转换为训练和预测图的配方。
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
# 上下文以及父类组件
from rasa.engine.graph import ExecutionContext, GraphComponent
# 当前节点资源
from rasa.engine.storage.resource import Resource
# 当前节点存储
from rasa.engine.storage.storage import ModelStorage
# 稠密特征父类，继承于Featurizer
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer
# 分词
from rasa.nlu.tokenizers.tokenizer import Tokenizer
# 数据集处理
from rasa.shared.nlu.training_data.training_data import TrainingData
# 特征，存储任何特征化器产生的特征。这个不是特征，是数据集的
from rasa.shared.nlu.training_data.features import Features
# 消息存储类
from rasa.shared.nlu.training_data.message import Message

# 密集的可特征化属性本质上是文本属性&别名
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
)
# 固定的几个字符串text，text_tokens,sentence,sequence
from rasa.shared.nlu.constants import (
    TEXT,
    TEXT_TOKENS,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)

# 返回具有指定名称的记录器，必要时创建它。
logger = logging.getLogger(__name__)


# 注册信息特征组件，并且不训练，直接加载训练好的模型
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class BytePairFeaturizer(DenseFeaturizer, GraphComponent):
    """
    自定义稠密特征组件
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        # 特征的前一个组件是分词
        return [Tokenizer]

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        # 当前组件用到饿了bpemb模型，并且直接加载训练好的特征，不再进行训练
        return ["bpemb"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        # 获取模型需要的一些参数，并且将父类稠密向量的参数也一并获取到
        return {
            **DenseFeaturizer.get_default_config(),
            # specifies the language of the subword segmentation model
            # 指定子词分割模型的语言
            "lang": None,
            # specifies the dimension of the subword embeddings
            # 指定子词嵌入的维度
            "dim": None,
            # specifies the vocabulary size of the segmentation model
            # 指定分割模型的词汇量
            "vs": None,
            # if set to True and the given vocabulary size can't be loaded for the given
            # model, the closest size is chosen
            # 如果设置为 True 并且给定的词汇量大小无法加载给定模型，选择最接近的尺寸
            "vs_fallback": True,
        }

    def __init__(
            self,
            config: Dict[Text, Any],
            name: Text,
    ) -> None:
        """Constructs a new byte pair vectorizer."""
        super().__init__(name, config)
        # The configuration dictionary is saved in `self._config` for reference.
        # 配置字典保存在`self._config`中以供参考，直接写config也可以
        # 这里不需要resource和storage的原因好似不需要训练哈
        self.model = BPEmb(
            lang=self._config["lang"],
            dim=self._config["dim"],
            vs=self._config["vs"],
            vs_fallback=self._config["vs_fallback"],
        )

    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new component (see parent class for full docstring)."""
        # 创建一个新组件（请参阅父类以获取完整的文档字符串）。
        # 相当于初始化当前对象,和__init__()方法功能是一样的
        return cls(config, execution_context.node_name)

    def process(self, messages: List[Message]) -> List[Message]:
        """Processes incoming messages and computes and sets features."""
        # 处理传入的消息并计算和设置特征。
        for message in messages:
            # 每一条消失处理三下，分别是text,response,action_text
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                # text,response,action_text，为什么要搞这么多消息属性？？？？？
                self._set_features(message, attribute)
        return messages

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Processes the training examples in the given training data in-place."""
        self.process(training_data.training_examples)
        return training_data

    def _create_word_vector(self, document: Text) -> np.ndarray:
        """Creates a word vector from a text. Utility method."""
        # 从文本创建词向量。公共方法。利用BPEmb进行编码一句话，将提供的文本编码为字节对 ID。hello对应【908，922，918】，相当于token
        encoded_ids = self.model.encode_ids(document)
        if encoded_ids:
            # 为什么只获取列表里面第一个？？？？==》》》》这里进行额近似计算，小数据影响较大
            return self.model.vectors[encoded_ids[0]]  # sharp （25，）
        # 如果无法获取任何token，那么用0填充
        return np.zeros((self.component_config["dim"],), dtype=np.float32)

    def _set_features(self, message: Message, attribute: Text = TEXT) -> None:
        """Sets the features on a single message. Utility method."""
        # 对每一恶搞message都进行text,response,action_text操作一遍
        # 设置单个消息的功能。公共方法。
        # 获取单个message的text_tokens，这里的text_tokens是属性data的，本身是一个列表，一句话的每一个单词列表集合
        tokens = message.get(TEXT_TOKENS)

        # If the message doesn't have tokens, we can't create features.
        if not tokens:
            return None

        # We need to reshape here such that the shape is equivalent to that of sparsely
        # generated features. Without it, it'd be a 1D tensor. We need 2D (n_utterance, n_dim).
        # 生成的特征。没有它，它将是一维张量。我们需要 2D（n_utterance，n_dim）。shape=(dims,) ===>（1，dims）
        text_vector = self._create_word_vector(document=message.get(TEXT)).reshape(
            1, -1
        )
        # 生成单词的特征，刚好是一个2维的
        word_vectors = np.array(
            [self._create_word_vector(document=t.text) for t in tokens]
        )
        # shape：（word_count，dims）
        # features: The features.
        # feature_type: Type of the feature, e.g. FEATURE_TYPE_SENTENCE.
        # attribute: Message attribute, e.g. INTENT or TEXT.
        # origin: Name of the component that created the features.
        # 包含attribute，features,origin,type等属性
        final_sequence_features = Features(
            word_vectors,
            FEATURE_TYPE_SEQUENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        # 将特征添加到message里面
        message.add_features(final_sequence_features)
        final_sentence_features = Features(
            text_vector,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        if not config["lang"]:
            raise ValueError("BytePairFeaturizer needs language setting via `lang`.")
        if not config["dim"]:
            raise ValueError(
                "BytePairFeaturizer needs dimensionality setting via `dim`."
            )
        if not config["vs"]:
            raise ValueError("BytePairFeaturizer needs a vector size setting via `vs`.")
