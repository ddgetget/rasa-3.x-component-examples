import logging
from typing import Any, Text, Dict, List, Type
# 将原始文档集合转换为 TF-IDF 特征矩阵。
from sklearn.feature_extraction.text import TfidfVectorizer
# 将当前模型配置转换为训练和预测图的参数配置。用于设置是token，还是feature，分类，或是实体，并注明当前组件是否需要训练
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
# 保存有关单个图形运行的信息。所有组件的父类组件
from rasa.engine.graph import ExecutionContext, GraphComponent
# 表示图中的持久化图组件。
from rasa.engine.storage.resource import Resource
# 用作需要持久性的 `GraphComponents` 的存储后端。
from rasa.engine.storage.storage import ModelStorage
# 所有稀疏特征化器的基类。继承自Featurizer(所有特征化器的基类。)
from rasa.nlu.featurizers.sparse_featurizer.sparse_featurizer import SparseFeaturizer
# tokenizer的基类。
from rasa.nlu.tokenizers.tokenizer import Tokenizer
# 保存加载的意图和实体训练数据。
from rasa.shared.nlu.training_data.training_data import TrainingData
# 存储任何特征化器产生的特征,区别于Featurizer，这里一定要分清
from rasa.shared.nlu.training_data.features import Features
# 可用于描述对话轮次的数据容器。
from rasa.shared.nlu.training_data.message import Message

# 密集的可特征化属性本质上是文本属性[TEXT, RESPONSE, ACTION_TEXT], 例如：rasa.shared.nlu.constants.TEXT
# 字符串：alias， 别名
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
)
# 将任意 Python 对象保存到一个文件中。并且加载到内存
from joblib import dump, load
# 字符串：text，text_tokens，sentence，sequence
from rasa.shared.nlu.constants import (
    TEXT,
    TEXT_TOKENS,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)

# 返回具有指定名称的记录器，必要时创建它。
logger = logging.getLogger(__name__)


# 此装饰器可用于向配方注册类。
# DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER，枚举以在图表中正确分类和放置自定义组件。这里是1，因为是特征，0代表分词
# 这里是用了tf-idf，所以需要进行训练
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=True
)
class TfIdfFeaturizer(SparseFeaturizer, GraphComponent):
    """
    tf-idf为稀疏响亮，因为初始化的是one-hot，并继承GraphComponent，实现内部特定的函数，达到可以特征表示
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        # 应该在此组件之前包含在管道中的组件。这一层是特征层，前一层是分词层，即tokenizer
        return [Tokenizer]

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        # 此组件运行所需的任何额外 python 依赖项。
        # 这里使用sklearn的tf-idf进行特征表示，所以徐亚引入sklearn，方便环境里面没有安装sklearn，有个良好的错误提示
        return ["sklearn"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        # 返回组件的默认配置。
        return {
            **SparseFeaturizer.get_default_config(),  # 获取父类组件的一些默认信息，在追加自己的参数，整块跟__init__继承部分类似
            "analyzer": "word",
            "min_ngram": 1,
            "max_ngram": 1,
        }

    def __init__(
            self,
            config: Dict[Text, Any],
            name: Text,
            model_storage: ModelStorage,
            resource: Resource,
    ) -> None:
        """Constructs a new tf/idf vectorizer using the sklearn framework."""
        # 使用 sklearn 框架构造一个新的 tf-idf 矢量化器。
        super().__init__(name, config)
        # Initialize the tfidf sklearn component
        # 初始化 tfidf sklearn 组件
        self.tfm = TfidfVectorizer(
            analyzer=config["analyzer"],
            ngram_range=(config["min_ngram"], config["max_ngram"]),
        )

        # We need to use these later when saving the trained component.
        # 保存的这个训练好的组件，我们后面会用到
        self._model_storage = model_storage
        self._resource = resource

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the component from training data."""
        # 从训练数据中获取每一句话，并装进列表里面，（List）texts=['hello']
        texts = [e.get(TEXT) for e in training_data.training_examples if e.get(TEXT)]
        # 使用tf-idf训练
        self.tfm.fit(texts)
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
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, execution_context.node_name, model_storage, resource)

    def _set_features(self, message: Message, attribute: Text = TEXT) -> None:
        """Sets the features on a single message. Utility method."""
        # 设置单个消息的功能。公用方法。
        tokens = message.get(TEXT_TOKENS)

        # If the message doesn't have tokens, we can't create features.
        if not tokens:
            return None

        # Make distinction between sentence and sequence features
        text_vector = self.tfm.transform([message.get(TEXT)])
        word_vectors = self.tfm.transform([t.text for t in tokens])

        final_sequence_features = Features(
            word_vectors,
            FEATURE_TYPE_SEQUENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sequence_features)
        final_sentence_features = Features(
            text_vector,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)

    def process(self, messages: List[Message]) -> List[Message]:
        """Processes incoming message and compute and set features."""
        for message in messages:  # 遍历每一个message，一个message里面只有一个单词的时候，是(List)messages.features=[3(属性个数)*2(序列特征，句子特征)]=[6(总特征数)]
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:  # 针对灭一句话，处理这些result=['text', 'response', 'action_text']属性
                self._set_features(message, attribute)
        return messages

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Processes the training examples in the given training data in-place."""
        self.process(training_data.training_examples)
        return training_data

    def persist(self) -> None:
        """
        Persist this model into the passed directory.
        将此模型持久保存到传递的目录中。

        Returns the metadata necessary to load the model again. In this case; `None`.
        返回再次加载模型所需的元数据。在这种情况下; `无`。
        """
        with self._model_storage.write_to(self._resource) as model_dir:
            # 保存tfidfvectorizer模型到系统盘
            dump(self.tfm, model_dir / "tfidfvectorizer.joblib")

    # classmethod 修饰符对应的函数不需要实例化，不需要 self 参数，但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等。
    @classmethod
    def load(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Loads trained component from disk."""
        # 从磁盘加载经过训练的组件。对应上面的train方法
        try:
            with model_storage.read_from(resource) as model_dir:
                # 这里模型的文件名称需要对应
                tfidfvectorizer = load(model_dir / "tfidfvectorizer.joblib")
                # 加载稀疏特征响亮
                component = cls(
                    config, execution_context.node_name, model_storage, resource
                )
                # 这里可以理解成创建当前对象，相当于__init__()，和create()方法是一致的，还原一个组件而已
                component.tfm = tfidfvectorizer
                # 对组件里面的tf-idf模型赋值成刚加载的那个，其实这里说明了，那别人训练好的模型，直接使用也可以
        except (ValueError, FileNotFoundError):
            # 当模型的名称写错的时候，就会报下面的错误
            logger.debug(
                f"Couldn't load metadata for component '{cls.__name__}' as the persisted "
                f"model data couldn't be loaded."
            )
        return component
        # 返回一个还原的组件

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        """
        根据配置参数计算一些数值
        """
        pass
