import pytest
import pathlib
import numpy as np
# 可用于描述对话轮次的数据容器。
from rasa.shared.nlu.training_data.message import Message
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.graph import ExecutionContext
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from .nlu_clf import LogisticRegressionClassifier


@pytest.fixture
def classifier(tmpdir):
    # components/test_clf.py::test_sparse_feats_added tmpdir:) /private/var/folders/z0/7kqx00cx0pq1dyl32g8z02cm0000gn/T/pytest-of-geng/pytest-2/test_sparse_feats_added0
    print("tmpdir:)", tmpdir)
    """Generate a classifier for tests."""
    # output_fingerprint：一个特定实例化的唯一标识符`资源`。用于区分相同的特定持久性保存到缓存时的“资源”。
    node_storage = LocalModelStorage(pathlib.Path(tmpdir))
    """Represents a persisted graph component in the graph.
    表示图中的持久化图组件。

    Attributes:
        name: The unique identifier for the `Resource`. Used to locate the associated
            data from a `ModelStorage`. Normally matches the name of the node which
            created it.
        name：“资源”的唯一标识符。用于定位关联的来自“模型存储”的数据。通常匹配节点的名称创造了它。
        output_fingerprint: An unique identifier for a specific instantiation of a
            `Resource`. Used to distinguish a specific persistence for the same
            `Resource` when saving to the cache.
        output_fingerprint：一个特定实例化的唯一标识符`资源`。用于区分相同的特定持久性保存到缓存时的“资源”。
    """
    # geng@LongGengYungs-MacBook-Pro sparse_feat % pwd
    # /private/var/folders/z0/7kqx00cx0pq1dyl32g8z02cm0000gn/T/pytest-of-geng/pytest-2/test_sparse_feats_added0/sparse_feat
    # geng@LongGengYungs-MacBook-Pro sparse_feat % ls
    # None.joblib		oov_words.json		vocabularies.pkl
    node_resource = Resource("sparse_feat")
    # 保存有关单个图形运行的信息。
    # setattr,getattr这里的父类使用了这个方法，增减对象属性
    context = ExecutionContext(node_storage, node_resource)
    print("context.node_name:)",context.node_name)
    return LogisticRegressionClassifier(
        config=LogisticRegressionClassifier.get_default_config(),
        name=context.node_name,
        resource=node_resource,
        model_storage=node_storage,
    )


@pytest.fixture
def featurizer(tmpdir):
    """Generate a featurizer for tests."""
    # tmpdir: /private/var/folders/z0/7kqx00cx0pq1dyl32g8z02cm0000gn/T/pytest-of-geng/pytest-3/test_sparse_feats_added0
    # 加载本地的sparse_featr存储部分，这个是在系统盘缓存盘里面
    # <rasa.engine.storage.local_model_storage.LocalModelStorage object at 0x7fae90e6cac0>
    node_storage = LocalModelStorage(pathlib.Path(tmpdir))
    # Resource(name='sparse_feat', output_fingerprint='5ae99abd4e774dc69cea0c078c569972')
    node_resource = Resource("sparse_feat")
    # 根据节点存储和资源构建上下文
    # ExecutionContext(model_id=Resource(name='sparse_feat', output_fingerprint='5ae99abd4e774dc69cea0c078c569972'), should_add_diagnostic_data=False, is_finetuning=False, node_name=None)
    context = ExecutionContext(node_storage, node_resource)
    return CountVectorsFeaturizer(
        config=CountVectorsFeaturizer.get_default_config(),
        resource=node_resource,
        model_storage=node_storage,
        execution_context=context,
    )

# 为实体提取创建特征。
tokeniser = WhitespaceTokenizer(
    {
        "only_alphanum": False,
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_",
    }
)


def test_sparse_feats_added(classifier, featurizer):
    """Checks if the sizes are appropriate."""
    # 检查尺寸是否合适。
    # Create training data.
    # 创建训练数据

    training_data = TrainingData(
        [
            Message({"text": "hello", "intent": "greet"}),
            Message({"text": "hi there", "intent": "greet"}),
            Message({"text": "ciao", "intent": "goodbye"}),
            Message({"text": "bye", "intent": "goodbye"}),
        ]
    )
    # First we add tokens.分词，系统默认是按照空格分的
    # training_data.training_examples（List）：[message,message,...],
    # 未运行process之前，training_data.nlu_examples==training_data.training_examples，data属性只含有intent,和text
    # training_data.nlu_examples列表的每个data属性增加text_tokens，和intent_tokens
    # 且运行process之后，training_data.nlu_examples==training_data.training_examples
    tokeniser.process(training_data.training_examples)

    # Next we add features.
    # 未训练之前，training_data.features=[]
    # 训练之后，training_data的features属性=[text_features_1,text_features_2,intent_features]
    featurizer.train(training_data)
    """Trains the featurizer.
    训练特征化器。

    Take parameters from config and
    construct a new count vectorizer using the sklearn framework.
    从配置中获取参数并使用 sklearn 框架构建一个新的计数向量器。
    """
    featurizer.process(training_data.training_examples)
    # 处理传入消息并计算和设置特征。

    # Train the classifier.
    classifier.train(training_data)
    # 使用自己的逻辑回归模型训练结果，冰存储到本电脑系统盘

    # Make predictions.
    classifier.process(training_data.training_examples)
    # 使用自己的逻辑回归模型预测，并将结果存储到message对象里面

    # Check that the messages have been processed correctly
    # 直接在message里面查询即可
    for msg in training_data.training_examples:
        name, conf = msg.get("intent")["name"], msg.get("intent")["confidence"]
        # 查询data属性里面的intent即可获得意图相关信息
        assert name in ["greet", "goodbye"]
        # 断言意图是否在当前项目里面
        assert 0 < conf
        # 断言当前的置信度是大于0，不符合的说明概率计算错误
        assert conf < 1
        # 断言当前的置信度是分小于1，不符合的说明概率计算错误
        ranking = msg.get("intent_ranking")
        # 获取message里面存取的前10个意图对应的概率
        assert {i["name"] for i in ranking} == {"greet", "goodbye"}
        # 并断言每一个意图都在计算范围内
        assert np.isclose(np.sum([i["confidence"] for i in ranking]), 1.0)
        # 计算所有意图概率之和是否为1，返回一个布尔数组，计算两个数组的偏差。
