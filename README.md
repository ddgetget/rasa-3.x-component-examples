# rasa-3.x-component-examples
for rasa-3.x-component-examples doc

Rasa Open Source 使用传入的模型配置来构建有向无环图。此图描述了模型配置中的项目之间的依赖关系以及数据在它们之间的流动方式。这有两个主要好处：
- Rasa Open Source 可以使用计算图来优化模型的执行。这方面的示例是有效缓存训练步骤或并行执行独立步骤。
- Rasa Open Source 可以灵活地表示不同的模型架构。只要图保持非循环，Rasa 开源理论上可以根据模型配置将任何数据传递给任何图组件，而无需将底层软件架构与使用的模型架构联系起来。

当将模型配置转换为计算图策略时，NLU 组件成为该图中的节点。虽然模型配置中的策略和 NLU 组件之间存在区别，但是当它们被放置在图中时，区别就被抽象掉了。此时策略和 NLU 组件成为抽象图组件。在实践中，这由 GraphComponent 接口表示：策略和 NLU 组件都必须从该接口继承，才能与 Rasa 的图兼容和可执行。
![graph_architecture](https://rasa.com/docs/rasa/img/graph_architecture.png)
## 操作
### 开始组件
```commandline
from rasa.core.policies.policy import Policy
from rasa.engine.recipes.default_recipe import DefaultV1Recipe

# TODO: Correctly register your graph component
@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT], is_trainable=True
)
class MyPolicy(Policy):
    ...
```
### 组件接口
```commandline
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Type, Dict, Text, Any, Optional

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage


class GraphComponent(ABC):
    """Interface for any component which will run in a graph."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return []

    @classmethod
    @abstractmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new `GraphComponent`.

        Args:
            config: This config overrides the `default_config`.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.

        Returns: An instantiated `GraphComponent`.
        """
        ...

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> GraphComponent:
        """Creates a component using a persisted version of itself.

        If not overridden this method merely calls `create`.

        Args:
            config: The config for this graph component. This is the default config of
                the component merged with config specified by the user.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.
            kwargs: Output values from previous nodes might be passed in as `kwargs`.

        Returns:
            An instantiated, loaded `GraphComponent`.
        """
        return cls.create(config, model_storage, resource, execution_context)

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config.

        Default config and user config are merged by the `GraphNode` before the
        config is passed to the `create` and `load` method of the component.

        Returns:
            The default config of the component.
        """
        return {}

    @staticmethod
    def supported_languages() -> Optional[List[Text]]:
        """Determines which languages this component can work with.

        Returns: A list of supported languages, or `None` to signify all are supported.
        """
        return None

    @staticmethod
    def not_supported_languages() -> Optional[List[Text]]:
        """Determines which languages this component cannot work with.

        Returns: A list of not supported languages, or
            `None` to signify all are supported.
        """
        return None

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return []

```

# 训练组件
## 训练组件运行顺序
里面没有出现的，is_trainable=False，pipeline和policies依次运行
```
2022-08-24 11:59:36 INFO     rasa.engine.training.hooks  - Starting to train component 'RegexFeaturizer'.
2022-08-24 11:59:36 INFO     rasa.engine.training.hooks  - Finished training component 'RegexFeaturizer'.
2022-08-24 11:59:36 INFO     rasa.engine.training.hooks  - Starting to train component 'LexicalSyntacticFeaturizer'.
2022-08-24 11:59:36 INFO     rasa.engine.training.hooks  - Finished training component 'LexicalSyntacticFeaturizer'.
2022-08-24 11:59:36 INFO     rasa.engine.training.hooks  - Starting to train component 'CountVectorsFeaturizer'.
2022-08-24 11:59:36 INFO     rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer  - 78 vocabulary items were created for text attribute.
2022-08-24 11:59:36 INFO     rasa.engine.training.hooks  - Finished training component 'CountVectorsFeaturizer'.
2022-08-24 11:59:36 INFO     rasa.engine.training.hooks  - Starting to train component 'CountVectorsFeaturizer'.
2022-08-24 11:59:36 INFO     rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer  - 704 vocabulary items were created for text attribute.
2022-08-24 11:59:36 INFO     rasa.engine.training.hooks  - Finished training component 'CountVectorsFeaturizer'.
2022-08-24 11:59:45 INFO     rasa.engine.training.hooks  - Starting to train component 'TfIdfFeaturizer'.
2022-08-24 11:59:45 INFO     rasa.engine.training.hooks  - Finished training component 'TfIdfFeaturizer'.
2022-08-24 11:59:45 INFO     rasa.engine.training.hooks  - Starting to train component 'LogisticRegressionClassifier'.
2022-08-24 11:59:46 INFO     rasa.engine.training.hooks  - Finished training component 'LogisticRegressionClassifier'.
2022-08-24 11:59:46 INFO     rasa.engine.training.hooks  - Starting to train component 'MemoizationPolicy'.
2022-08-24 11:59:46 INFO     rasa.engine.training.hooks  - Finished training component 'MemoizationPolicy'.
2022-08-24 11:59:46 INFO     rasa.engine.training.hooks  - Starting to train component 'RulePolicy'.
2022-08-24 11:59:47 INFO     rasa.engine.training.hooks  - Finished training component 'RulePolicy'.
2022-08-24 11:59:47 INFO     rasa.engine.training.hooks  - Starting to train component 'TEDPolicy'.
2022-08-24 12:00:04 INFO     rasa.engine.training.hooks  - Finished training component 'TEDPolicy'.
```


# 参考链接
1. [github-rasa-3.x-component-examples](https://github.com/RasaHQ/rasa-3.x-component-examples)
2. [github-rasa](https://github.com/RasaHQ/rasa)
3. [rasa-page](https://rasa.com/)
4. [custom-graph-components](https://rasa.com/docs/rasa/custom-graph-components)
5. [github-bpemb](https://github.com/bheinzerling/bpemb)