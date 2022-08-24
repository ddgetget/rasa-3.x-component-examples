#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-24 11:45
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    nlu_clf.py
# @Project: rasa-3.x-component-examples
# @Package:
# @Ref:

import logging
# python类型注解
from typing import Any, Text, Dict, List, Type
# 模型保存于家在成pkl,替代pickle，有效地处理包含大数据的Python对象(joblib)。转储& joblib。负载)。
from joblib import dump, load

from scipy.sparse import hstack, vstack, csr_matrix
# sklearn 的逻辑回归部分，用于本脚本的意图分类
from sklearn.linear_model import LogisticRegression

# 表示图中的持久化图组件。
from rasa.engine.storage.resource import Resource
# 作为需要持久化的' GraphComponents '的存储后端。
from rasa.engine.storage.storage import ModelStorage
# 将正常模型配置转换为训练和预测图的配方。
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
# 保存关于单个图运行的信息。接口的任何组件将运行在一个图形。
from rasa.engine.graph import ExecutionContext, GraphComponent
# 所有特征化器的基类。
from rasa.nlu.featurizers.featurizer import Featurizer
# 意图分类器。目前是空的，未实现
from rasa.nlu.classifiers.classifier import IntentClassifier
# 输出中最多有多少个标签排名，其他一切都将被切断，默认是10
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
# 保存加载的意图和实体训练数据。
from rasa.shared.nlu.training_data.training_data import TrainingData
# 可用于描述对话轮次的数据容器。
# 每一轮，由一组属性描述。在描述用户的行为，例如`TEXT` 和 `INTENT`。用于描述机器人操作动作，例如`ACTION_NAME` 。
from rasa.shared.nlu.training_data.message import Message

# 常量：TEXT = "text"
# 常量：INTENT = "intent"
from rasa.shared.nlu.constants import TEXT, INTENT

# 返回具有指定名称的记录器，必要时创建它。
logger = logging.getLogger(__name__)

# 此装饰器可用于向配方注册类。
# component_types：描述随后使用的组件的类型将组件放置在图表中。
# is_trainable： 如果组件需要训练，设置为'True'
# model_from：如果此组件需要预加载模型，则可以使用， 例如“SpacyNLP”或“MitieNLP”。
# 返回：注册好的类
"""ComponentType
MESSAGE_TOKENIZER = 0
MESSAGE_FEATURIZER = 1
INTENT_CLASSIFIER = 2
ENTITY_EXTRACTOR = 3
POLICY_WITHOUT_END_TO_END_SUPPORT = 4
POLICY_WITH_END_TO_END_SUPPORT = 5
MODEL_LOADER = 6
"""
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
# 意图分类器。目前是空的，未实现。主要实现GraphComponent里面的函数部分
class LogisticRegressionClassifier(IntentClassifier, GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        """应该在此组件之前包含在管道中的组件。"""
        return [Featurizer]

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        """此组件运行所需的任何额外 python 依赖项。"""
        return ["sklearn"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config.
        返回组件的默认配置。

        Default config and user config are merged by the `GraphNode` before the
        config is passed to the `create` and `load` method of the component.
        默认配置和用户配置在 `GraphNode` 之前合并, config 被传递给组件的 `create` 和 `load` 方法。
        Returns:
            The default config of the component.
            组件的默认配置。
        """
        return {"class_weight": "balanced", "max_iter": 100, "solver": "lbfgs"}

    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        """
        node_storage = LocalModelStorage(pathlib.Path(tmpdir))
        node_resource = Resource("sparse_feat")
        context = ExecutionContext(node_storage, node_resource)

        return LogisticRegressionClassifier(
            config=LogisticRegressionClassifier.get_default_config(),
            name=context.node_name,
            resource=node_resource,
            model_storage=node_storage,
        )
        :param config:
        :param name: 节点的名称
        :param model_storage: 节点的存储后端
        :param resource: 节点的持久化
        """
        self.name = name

        # 真正的逻辑回归模型
        self.clf = LogisticRegression(
            solver=config["solver"],
            max_iter=config["max_iter"],
            class_weight=config["class_weight"],
        )

        # We need to use these later when saving the trained component.
        # 我们需要稍后在保存经过训练的组件时使用这些。
        self._model_storage = model_storage
        self._resource = resource

    def _create_X(self, messages: List[Message]) -> csr_matrix:
        """This method creates a sparse X array that can be used for predicting"""
        # 当前方法创建了一个可用于预测的稀疏 X 数组
        X = []
        for e in messages:
            # First element is sequence features, second is sentence features
            sparse_feats = e.get_sparse_features(attribute=TEXT)[1]  # 获取features属性夏元组的第一个特征值，是序列特征
            # First element is sequence features, second is sentence features
            dense_feats = e.get_dense_features(attribute=TEXT)[1]  # 获取features属性夏元组的第二个特征值，是句子特征
            together = hstack(  # 这部分操作和训练部分是一致的，水平堆叠稀疏矩阵（按列），区别在于不需要标签课，因为这是预测部分
                [
                    csr_matrix(sparse_feats.features if sparse_feats else []),  # 压缩稀疏行矩阵
                    csr_matrix(dense_feats.features if dense_feats else []),  # 压缩稀疏行矩阵
                ]
            )
            X.append(together)
        return vstack(X)

    def _create_training_matrix(self, training_data: TrainingData):
        """
        This method creates a scikit-learn compatible (X, y)-pair for training
        the logistic regression model.
        此方法创建一个 scikit-learn 兼容的 (X, y)-pair 用于训练逻辑回归模型。
        """
        # 训练集
        X = []
        # 标签
        y = []
        for e in training_data.training_examples:
            # e为message类型，
            # e.data={"text":"","intent":""}(Dict),
            # e.features=[](List),
            # e.output_properties={'text'}(Set)

            # 判断是否含有意图 intent
            if e.get(INTENT):
                if e.get("text"):  # 判断是否含有文本
                    # First element is sequence features, second is sentence features
                    sparse_feats = e.get_sparse_features(attribute=TEXT)[1]  # Tuple（sequence features，sentence features）序列特征，
                    # First element is sequence features, second is sentence features
                    dense_feats = e.get_dense_features(attribute=TEXT)[1]
                    together = hstack(  # 水平堆叠稀疏矩阵（按列）
                        [
                            csr_matrix(sparse_feats.features if sparse_feats else []),  # 压缩稀疏行矩阵
                            csr_matrix(dense_feats.features if dense_feats else []),  # 压缩稀疏行矩阵
                        ]
                    )
                    X.append(together)
                    y.append(e.get(INTENT))
        return vstack(X), y

    def train(self, training_data: TrainingData) -> Resource:
        """
        训练当前节点，并将结果存储到本地
        :param training_data:
        :return:
        """
        #   (0, 2)	1.0
        #   (1, 3)	1.0
        #   (1, 4)	1.0
        #   (2, 1)	1.0
        #   (3, 0)	1.0
        X, y = self._create_training_matrix(training_data)  # X.shape=（total_size， features_size）  y=['greet', 'greet', 'goodbye', 'goodbye']

        # 逻辑回归模型训练
        self.clf.fit(X, y)
        # 当前节点的字眼处处到可持久化的目录下
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
        """Creates a new `GraphComponent`.
            创建一个新的`GraphComponent`
        Args:
            config: This config overrides the `default_config`.
            config: 这个配置覆盖了`default_config`。
            model_storage: Storage which graph components can use to persist and load
                themselves.
            model_storage：图组件可以用来持久化和加载的存储他们自己。
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            resource：此组件的资源定位器，可用于持久化并从 `model_storage` 加载自身。
            execution_context: Information about the current graph run.
            execution_context：有关当前图形运行的信息。

        Returns: An instantiated `GraphComponent`.
        返回：一个实例化的 `GraphComponent`。
        """
        return cls(config, execution_context.node_name, model_storage, resource)

    def process(self, messages: List[Message]) -> List[Message]:
        """
        模型预测部分
        :param messages: 带预测的数据集，已经分词，并特征化的数据集
        :return:
        """
        X = self._create_X(messages)
        # 把数据剔除标签部分，整理成和训练一样的数据格式

        # TODO：查询sklearn的逻辑回归部分，标签不进行ID化，为什么可以直接预测？
        pred = self.clf.predict(X)
        # 当前模型还在内存，直接预测即可，而且结果直接是标签，不需要字典ID查询  ['greet' 'greet' 'goodbye' 'goodbye']
        probas = self.clf.predict_proba(X)  # probs.shape=（4，2）
        for idx, message in enumerate(messages):
            intent = {"name": pred[idx], "confidence": probas[idx].max()}  # (Dict)intent={'name': 'greet', 'confidence': 0.58209005055583}
            intents = self.clf.classes_
            # 所有意图，intents.shape=(2,)  这个属性是逻辑回归的，并不是rasa框架的哈，['goodbye' 'greet']
            intent_info = {
                k: v
                for i, (k, v) in enumerate(zip(intents, probas[idx]))
                # 遍历预测的矩阵当前行的所有意图以及对应的概率
                if i < LABEL_RANKING_LENGTH
                # 系统默认设置10，当意图大于10个把后面的删除了，导致的结果是，当意图多于10个，本身预测出的结果有可能无法显示
            }
            # 这只是一句话的预测结果，包含前LABEL_RANKING_LENGTH个的意图以及概率
            intent_ranking = [
                {"name": k, "confidence": v} for k, v in intent_info.items()
            ]
            # 对intent_info增加字典的key值，{"intent_value":"confidence_value",...} 转换成 [{"name": "intent_value", "confidence": "confidence_value"},...]

            message.set("intent", intent, add_to_output=True)
            # (Message)给message增加一对键值对，是存储在data属性下哈，之前的data属性下的intent是字符串，现在类型是字典
            message.set("intent_ranking", intent_ranking, add_to_output=True)
            # (Message)给message增加一对键值对，是存储在data属性下哈，之前的data属性下的没有intent_ranking，现在类型是列表，表现了当前这句话前十个意图以及概率
        return messages

    def persist(self) -> None:
        """
        将当前节点存储到本地
        :return:
        """

        # self._model_storage.write_to(): 保留给定资源的数据。
        with self._model_storage.write_to(self._resource) as model_dir:
            """Persists data for a given resource.   
            self._model_storage.write_to(): 保留给定资源的数据。

            This `Resource` can then be accessed in dependent graph nodes via
            `model_storage.read_from`.
            然后可以通过以下方式在相关图节点中访问此“资源”`model_storage.read_from`。

            Args:
                resource: The resource which should be persisted.
                resource：应该被持久化的资源。

            Returns:
                A directory which can be used to persist data for the given `Resource`.
                可用于为给定“资源”保存数据的目录。
            """
            # model_dir: /private/var/folders/z0/7kqx00cx0pq1dyl32g8z02cm0000gn/T/pytest-of-geng/pytest-6/test_sparse_feats_added0/sparse_feat
            dump(self.clf, model_dir / f"{self.name}.joblib")

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a component using a persisted version of itself.
        使用自身的持久版本创建组件。

        If not overridden this method merely calls `create`.
        如果没有被覆盖，这个方法只会调用`create`。

        Args:
            config: The config for this graph component. This is the default config of
                the component merged with config specified by the user.
            config：此图形组件的配置。这是默认配置组件与用户指定的配置合并。
            model_storage: Storage which graph components can use to persist and load
                themselves.
            model_storage：图组件可以用来持久化和加载的存储他们自己。
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            resource：此组件的资源定位器，可用于持久化并从 `model_storage` 加载自身。
            execution_context: Information about the current graph run.
            execution_context：有关当前图形运行的信息。
            kwargs: Output values from previous nodes might be passed in as `kwargs`.
            kwargs：来自先前节点的输出值可以作为 `kwargs` 传入。

        Returns:
            An instantiated, loaded `GraphComponent`.
            实例化、加载的 `GraphComponent`。
        """
        with model_storage.read_from(resource) as model_dir:
            classifier = load(model_dir / f"{resource.name}.joblib")
            component = cls(
                config, execution_context.node_name, model_storage, resource
            )
            component.clf = classifier
            return component

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
