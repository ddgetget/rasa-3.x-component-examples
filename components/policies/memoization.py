#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-26 10:50
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    memoization.py
# @Project: rasa-3.x-component-examples
# @Package: 
# @Ref:

from __future__ import annotations
import zlib

import base64
import json
import logging

from tqdm import tqdm
from typing import Optional, Any, Dict, List, Text
from pathlib import Path

import rasa.utils.io
import rasa.shared.utils.io
# 上下文信息组件
from rasa.engine.graph import ExecutionContext
# 将正常模型配置转换为训练和预测图的配方。
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
# 表示图中的持久化图组件。
from rasa.engine.storage.resource import Resource
# 用作需要持久性的“GraphComponents”的存储后端。
from rasa.engine.storage.storage import ModelStorage
# （Dict）State状态, Domain域指定机器人的策略在其中执行的范围。Domain子类提供机器人可以采取的操作，意图以及它可以识别的实体。
from rasa.shared.core.domain import State, Domain
# 一个操作描述了所采取的操作及其结果。它包含一个操作和一个事件列表。操作将被追加到' Tracker.turns '中最新的' Turn '。
from rasa.shared.core.events import ActionExecuted
# 实际跟踪器特性器的基类。
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
# 将跟踪器历史截断为' max_history '长序列。
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
# 文件名："featurizer.json"
from rasa.core.featurizers.tracker_featurizers import FEATURIZER_FILE
# 如果在进行文件IO时发生错误将引发
from rasa.shared.exceptions import FileIOException
# 存储关于“策略”预测的信息。
# 所有对话策略的通用父类
# 策略支持的训练数据类型的枚举。
from rasa.core.policies.policy import PolicyPrediction, Policy, SupportedData
#维护对话的状态。
from rasa.shared.core.trackers import DialogueStateTracker
# 一个跟踪器包装器，缓存跟踪器的状态创建。
from rasa.shared.core.generator import TrackerWithCachedStates
# 如果日志级别设置为WARNING或ERROR则返回' True '，否则返回' False '。
from rasa.shared.utils.io import is_logging_disabled

# # 最低优先级用于机器学习策略。
# DEFAULT_POLICY_PRIORITY = 1  # 1
# UNLIKELY_INTENT_POLICY_PRIORITY = DEFAULT_POLICY_PRIORITY + 1  # 2
# MEMOIZATION_POLICY_PRIORITY = UNLIKELY_INTENT_POLICY_PRIORITY + 1  # 3

# DEFAULT_MAX_HISTORY = None  # 默认情况下，核心策略历史是不受限制的
# POLICY_MAX_HISTORY = "max_history"
# POLICY_PRIORITY = "priority"
from rasa.core.constants import (
    MEMOIZATION_POLICY_PRIORITY,
    DEFAULT_MAX_HISTORY,
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
)
# ACTION_LISTEN_NAME = "action_listen"
from rasa.shared.core.constants import ACTION_LISTEN_NAME

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=True
)
class MemoizationPolicy(Policy):
    # memoization实际上就是记录所有的状态，和状态对应的action。将这些数据持久化之后等待预测时使用。
    #
    # 预测时则是读出这些数据并查询当前状态对应的action
    """A policy that follows exact examples of `max_history` turns in training stories.

    Since `slots` that are set some time in the past are
    preserved in all future feature vectors until they are set
    to None, this policy implicitly remembers and most importantly
    recalls examples in the context of the current dialogue
    longer than `max_history`.

    This policy is not supposed to be the only policy in an ensemble,
    it is optimized for precision and not recall.
    It should get a 100% precision because it emits probabilities of 1.1
    along it's predictions, which makes every mistake fatal as
    no other policy can overrule it.

    If it is needed to recall turns from training dialogues where
    some slots might not be set during prediction time, and there are
    training stories for this, use AugmentedMemoizationPolicy.
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""  # 返回默认参数
        # please make sure to update the docs when changing a default parameter
        return {
            "enable_feature_string_compression": True,
            "use_nlu_confidence_as_score": False,
            POLICY_PRIORITY: MEMOIZATION_POLICY_PRIORITY,
            POLICY_MAX_HISTORY: DEFAULT_MAX_HISTORY,
        }

    def _standard_featurizer(self) -> MaxHistoryTrackerFeaturizer:
        # Memoization policy always uses MaxHistoryTrackerFeaturizer
        # without state_featurizer
        return MaxHistoryTrackerFeaturizer(
            state_featurizer=None, max_history=self.config[POLICY_MAX_HISTORY]
        )

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: Optional[TrackerFeaturizer] = None,
        lookup: Optional[Dict] = None,
    ) -> None:
        """Initialize the policy."""
        super().__init__(config, model_storage, resource, execution_context, featurizer)
        self.lookup = lookup or {}

    def _create_lookup_from_states(
        self,
        trackers_as_states: List[List[State]],
        trackers_as_actions: List[List[Text]],
    ) -> Dict[Text, Text]:
        """Creates lookup dictionary from the tracker represented as states.
        从表示为状态的跟踪器创建查找字典。

        Args:
            trackers_as_states: representation of the trackers as a list of states
            trackers_as_actions: representation of the trackers as a list of actions

        Returns:
            lookup dictionary
        """
        lookup: Dict[Text, Text] = {}

        if not trackers_as_states:
            return lookup

        assert len(trackers_as_actions[0]) == 1, (
            f"The second dimension of trackers_as_action should be 1, "
            f"instead of {len(trackers_as_actions[0])}"
        )

        ambiguous_feature_keys = set()

        pbar = tqdm(
            zip(trackers_as_states, trackers_as_actions),
            desc="Processed actions",
            disable=is_logging_disabled(),
        )
        for states, actions in pbar:
            action = actions[0]

            feature_key = self._create_feature_key(states)
            if not feature_key:
                continue

            if feature_key not in ambiguous_feature_keys:
                if feature_key in lookup.keys():
                    if lookup[feature_key] != action:
                        # delete contradicting example created by partial history augmentation from memory
                        # 从内存中删除部分历史扩展创建的矛盾示例
                        ambiguous_feature_keys.add(feature_key)
                        del lookup[feature_key]
                else:
                    lookup[feature_key] = action
            pbar.set_postfix({"# examples": "{:d}".format(len(lookup))})

        return lookup

    def _create_feature_key(self, states: List[State]) -> Optional[Text]:
        if not states:
            return None

        # we sort keys to make sure that the same states represented as dictionaries have the same json strings quotes are removed for aesthetic reasons
        # 我们对键进行排序，以确保以字典表示的相同状态具有相同的json字符串，出于美观的原因，引号被删除
        feature_str = json.dumps(states, sort_keys=True).replace('"', "")
        if self.config["enable_feature_string_compression"]:
            compressed = zlib.compress(
                bytes(feature_str, rasa.shared.utils.io.DEFAULT_ENCODING)
            )
            return base64.b64encode(compressed).decode(
                rasa.shared.utils.io.DEFAULT_ENCODING
            )
        else:
            return feature_str

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:
        # only considers original trackers (no augmented ones)
        # 只考虑原始跟踪器(不考虑增强跟踪器)
        training_trackers = [
            t
            for t in training_trackers
            if not hasattr(t, "is_augmented") or not t.is_augmented
        ]
        training_trackers = SupportedData.trackers_for_supported_data(
            self.supported_data(), training_trackers
        )

        (
            trackers_as_states,
            trackers_as_actions,
        ) = self.featurizer.training_states_and_labels(training_trackers, domain)
        self.lookup = self._create_lookup_from_states(
            trackers_as_states, trackers_as_actions
        )
        logger.debug(f"Memorized {len(self.lookup)} unique examples.")

        self.persist()
        return self._resource

    def _recall_states(self, states: List[State]) -> Optional[Text]:
        return self.lookup.get(self._create_feature_key(states))

    def recall(
        self,
        states: List[State],
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]],
    ) -> Optional[Text]:
        """Finds the action based on the given states.
        根据给定的状态查找操作。

        Args:
            states: List of states.
            states: 状态列表。
            tracker: The tracker.
            tracker: 跟踪。
            domain: The Domain.
            domain: 域。
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.
            rule_only_data: 特定于规则的槽和循环，因此应该被这个政策忽略。

        Returns:
            The name of the action.
            动作的名称。
        """
        return self._recall_states(states)

    def _prediction_result(
        self, action_name: Text, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        result = self._default_predictions(domain)
        if action_name:
            if (
                self.config["use_nlu_confidence_as_score"]
                and tracker.latest_message is not None
            ):
                # the memoization will use the confidence of NLU on the
                # latest user message to set the confidence of the action
                # memoization将使用NLU对最新用户消息的置信度来设置动作的置信度
                score = tracker.latest_message.intent.get("confidence", 1.0)
            else:
                score = 1.0

            result[domain.index_for_action(action_name)] = score

        return result

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.
        预测机器人在看到跟踪器后应该采取的下一个行动。

        Args:
            tracker: the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.shared.core.domain.Domain`
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.

        Returns:
             The policy's prediction (e.g. the probabilities for the actions).
        """
        result = self._default_predictions(domain)

        states = self._prediction_states(tracker, domain, rule_only_data=rule_only_data)
        logger.debug(f"Current tracker state:{self.format_tracker_states(states)}")
        predicted_action_name = self.recall(
            states, tracker, domain, rule_only_data=rule_only_data
        )
        if predicted_action_name is not None:
            logger.debug(f"There is a memorised next action '{predicted_action_name}'")
            result = self._prediction_result(predicted_action_name, tracker, domain)
        else:
            logger.debug("There is no memorised next action")

        return self._prediction(result)

    def _metadata(self) -> Dict[Text, Any]:
        return {"lookup": self.lookup}

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "memorized_turns.json"

    def persist(self) -> None:
        """Persists the policy to storage."""
        # 将策略持久化到存储。
        with self._model_storage.write_to(self._resource) as path:
            # not all policies have a featurizer
            # 并不是所有的策略都有一个特性器
            if self.featurizer is not None:
                self.featurizer.persist(path)

            file = Path(path) / self._metadata_filename()

            rasa.shared.utils.io.create_directory_for_file(file)
            rasa.shared.utils.io.dump_obj_as_json_to_file(file, self._metadata())

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> MemoizationPolicy:
        """Loads a trained policy (see parent class for full docstring)."""
        # 加载一个经过训练的策略(完整的文档字符串请参阅父类)。在rasa_run的时候调用，这里使用config参数，所以会调用get_default_config函数
        featurizer = None
        lookup = None

        try:
            # 需要调查什么时候写的这个文件？？？已查明，在持久化persist里面写的
            with model_storage.read_from(resource) as path:
                # C:\Users\ADMINI~1\AppData\Local\Temp\tmpuygvb2qi\train_components.policies.memoization.MemoizationPolicy0\memorized_turns.json
                metadata_file = Path(path) / cls._metadata_filename()
                """部分动作的路径
                {'lookup': {'eJyLrq6NBQAEOAGx': 'action_listen', 
                'eJyLrq7VUaguKEoti09MLsnMz7NSqIYw4vMSc1OtFKCcnMziktQ8oNrS4tQioJrMPCC3xEohvSg1taS2NhYA6dwbhQ==': 'utter_greet', 
                'eJyLrq7VUaguKEoti09MLsnMz7NSqIYw4vMSc1OtFKCcnMziktQ8oNrS4tQioJrMPCC3xEohvSg1taSWgBmlJSWpRfEQpbhMiAUAQvc0lw==': 'action_listen', 
                'eJyLrq7VUaguKEoti09MLsnMz7NSqIYw4vMSc1OtFKCcnMziktQ8oNrS4tQioJrMPCC3xEohvSg1taSWgBmlJSWpRfEQpeSZQMAVufn5KSDzE4EGxQIAPEFQdQ==': 'utter_happy', 
                'eJyLrq7VUaguKEoti09MLsnMz7NSqIYw4vMSc1OtFKCcnMziktQ8oNrS4tQioJrMPCC3xEohvSg1taSWgBmlJSWpRfEQpeSZQMAVufn5KSDzE4l0SkZiQUElfmNiAVKZa5w=': 'action_listen', 
                'eJyLrq7VUaguKEoti09MLsnMz7NSqIYw4vMSc1OtFKCcnMziktQ8oNrS4tQioJrMPCC3xEohvSg1taSWgBmlJSWpRfEQpeSZQMAVufn5KfGleRmJBQWVtbWxAN75UWc=': 'utter_cheer_up', 
                'eJydjlEKgCAQRK/iATqBV4lYxJYUchVdgxDv3oL9G/0MM/B4zNr6olrKeIGx7CNp1UYBMgG1esfpCyMJWwtmYTzJZK2OjMh94qjMmGGg/wyTFyHGHSo5k9L97Yx1KFnTzLU9XiFuqQ==': 'utter_did_that_help', 
                'eJytjlEKgCAQRK/iATqBV4lYRIcUahVbg5DunlD/RvQzzMDjMWM9B1VTxk7GSoisVb0LsVmh1TOWsAm4sWVDbkzgNkWrOQNydhxFBJlu9Juh82KN0VFhb1I63p2xHi1L+sPlgiPxRshj6QqnC1EGjek=': 'action_listen', 
                'eJytj9EJwzAMRFfxAJ3Aq4QgRKzUglg2ilwoxrvXkP67tP0Rd/B4nJbWb64VpQfgZpzFu3YFEEzk3bscfBrJYOtJOhiWUc27uxJZnziqGSlc6HeGyYqUc4AqEUt5fjZmizRuLf9wBQ5gEQ0iHT8KJ3/ivrOm3tcXH2yoGw==': 'utter_happy', 
                'eJytj0EKAyEMRa/iAXoCr1JKCGOmBsYomVgoMnevYLfFoe0m5MPn5eXajotrRekBuBhn8a6NBQQTefcOG+9G0rt1J+0dlh7Nu7sS2TFhVDNSGNXvCBOLlHOAKhFLeZ6TWSL1Wcs/WIEDWESDSNuPwMmfuK6s6ZzVuPcRcXsBe37Blg==': 'action_listen', 
                'eJytj1EKwyAQRK/iAXoCr1LCInGIQrKKWQNFvHsX7L+h7c8yA4/H7LP1h2m54CK3SkxsTRuB2B2w5lP2eApY2XqiKBNZq1izFUD6xFFFUGig3xkmK46UPFUOLufXvTFrgN6a/+Hy0ZMEJxSw/yic/OnBqlje0A2nVg==': 'utter_goodbye', 
                'eJytj9EJwzAMRFfxAJ3Aq4Qg3PiIDYlsHLkQTHavIP0NDm1/hA6Op6ehHQ/TcsGL3CQxsTXtXIjdCms+YYmbgLVbNxTtRNYo1swFkKPDqCIodFa/I3Qs1pQ8VQ4u5/2ezBSgs+Z/sHz0JMEJBSw/Ajt/evBNp1mPPndcIcY3+fTA0w==': 'action_listen'}}
                """
                metadata = rasa.shared.utils.io.read_json_file(metadata_file)
                lookup = metadata["lookup"]
                # Path(path)计算对应平台的文件路径  FEATURIZER_FILE=“'featurizer.json'”
                """
                {
                "py/object": "rasa.core.featurizers.tracker_featurizers.MaxHistoryTrackerFeaturizer",
                "state_featurizer": null,
                "max_history": null,
                "remove_duplicates": true
                }
                """
                if (Path(path) / FEATURIZER_FILE).is_file():
                    featurizer = TrackerFeaturizer.load(path)

        except (ValueError, FileNotFoundError, FileIOException):
            logger.warning(
                f"Couldn't load metadata for policy '{cls.__name__}' as the persisted "
                f"metadata couldn't be loaded."
            )

        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            featurizer=featurizer,
            lookup=lookup,
        )


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=True
)
class AugmentedMemoizationPolicy(MemoizationPolicy):
    """The policy that remembers examples from training stories for `max_history` turns.

    If it is needed to recall turns from training dialogues
    where some slots might not be set during prediction time,
    add relevant stories without such slots to training data.
    E.g. reminder stories.

    Since `slots` that are set some time in the past are
    preserved in all future feature vectors until they are set
    to None, this policy has a capability to recall the turns
    up to `max_history` from training stories during prediction
    even if additional slots were filled in the past
    for current dialogue.
    """

    @staticmethod
    def _strip_leading_events_until_action_executed(
        tracker: DialogueStateTracker, again: bool = False
    ) -> Optional[DialogueStateTracker]:
        """Truncates the tracker to begin at the next `ActionExecuted` event.

        Args:
            tracker: The tracker to truncate.
            again: When true, truncate tracker at the second action.
                Otherwise truncate to the first action.

        Returns:
            The truncated tracker if there were actions present.
            If none are found, returns `None`.
        """
        idx_of_first_action = None
        idx_of_second_action = None

        applied_events = tracker.applied_events()

        # we need to find second executed action
        for e_i, event in enumerate(applied_events):
            if isinstance(event, ActionExecuted):
                if idx_of_first_action is None:
                    idx_of_first_action = e_i
                else:
                    idx_of_second_action = e_i
                    break

        # use first action, if we went first time and second action, if we went again
        idx_to_use = idx_of_second_action if again else idx_of_first_action
        if idx_to_use is None:
            return None

        # make second ActionExecuted the first one
        events = applied_events[idx_to_use:]
        if not events:
            return None

        truncated_tracker = tracker.init_copy()
        for e in events:
            truncated_tracker.update(e)

        return truncated_tracker

    def _recall_using_truncation(
        self,
        old_states: List[State],
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]],
    ) -> Optional[Text]:
        """Attempts to match memorized states to progressively shorter trackers.

        This method iteratively removes the oldest events up to the next action
        executed and checks if the truncated event sequence matches some memorized
        states, until a match has been found or until the even sequence has been
        exhausted.

        Args:
            old_states: List of states.
            tracker: The tracker.
            domain: The Domain.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.

        Returns:
            The name of the action.
        """
        logger.debug("Launch DeLorean...")

        # Truncate the tracker based on `max_history`
        truncated_tracker: Optional[
            DialogueStateTracker
        ] = _trim_tracker_by_max_history(tracker, self.config[POLICY_MAX_HISTORY])
        truncated_tracker = self._strip_leading_events_until_action_executed(
            truncated_tracker
        )
        while truncated_tracker is not None:
            states = self._prediction_states(
                truncated_tracker, domain, rule_only_data=rule_only_data
            )

            if old_states != states:
                # check if we like new futures
                memorised = self._recall_states(states)
                if memorised is not None:
                    logger.debug(f"Current tracker state {states}")
                    return memorised
                old_states = states

            # go back again
            truncated_tracker = self._strip_leading_events_until_action_executed(
                truncated_tracker, again=True
            )

        # No match found
        logger.debug(f"Current tracker state {old_states}")
        return None

    def recall(
        self,
        states: List[State],
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]],
    ) -> Optional[Text]:
        """Finds the action based on the given states.

        Uses back to the future idea to change the past and check whether the new future
        can be used to recall the action.

        Args:
            states: List of states.
            tracker: The tracker.
            domain: The Domain.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.

        Returns:
            The name of the action.
        """
        predicted_action_name = self._recall_states(states)
        if predicted_action_name is None:
            # let's try a different method to recall that tracker
            return self._recall_using_truncation(
                states, tracker, domain, rule_only_data=rule_only_data
            )
        else:
            return predicted_action_name


def _get_max_applied_events_for_max_history(
    tracker: DialogueStateTracker, max_history: Optional[int]
) -> Optional[int]:
    """Computes the number of events in the tracker that correspond to max_history.

    To ensure that the last user utterance is correctly included in the prediction
    states, return the index of the most recent `action_listen` event occuring
    before the tracker would be truncated according to the value of `max_history`.

    Args:
        tracker: Some tracker holding the events
        max_history: The number of actions to count

    Returns:
        The number of events, as counted from the end of the event list, that should
        be taken into accout according to the `max_history` setting. If all events
        should be taken into account, the return value is `None`.
    """
    if not max_history:
        return None
    num_events = 0
    num_actions = 0
    for event in reversed(tracker.applied_events()):
        num_events += 1
        if isinstance(event, ActionExecuted):
            num_actions += 1
            if num_actions > max_history and event.action_name == ACTION_LISTEN_NAME:
                return num_events
    return None


def _trim_tracker_by_max_history(
    tracker: DialogueStateTracker, max_history: Optional[int]
) -> DialogueStateTracker:
    """Removes events from the tracker until it has `max_history` actions.
    从跟踪器中移除事件，直到它有' max_history '动作。

    Args:
        tracker: Some tracker.
        跟踪器:一些追踪。
        max_history: Number of actions to keep.
        max_history:要保留的动作数。

    Returns:
        A new tracker with up to `max_history` actions, or the same tracker if
        `max_history` is `None`.
        一个动作最多为' max_history '的新跟踪器，或者相同的跟踪器if“max_history”是“没有”。
    """
    max_applied_events = _get_max_applied_events_for_max_history(tracker, max_history)
    if not max_applied_events:
        return tracker

    applied_events = tracker.applied_events()[-max_applied_events:]
    new_tracker = tracker.init_copy()
    for event in applied_events:
        new_tracker.update(event)
    return new_tracker
