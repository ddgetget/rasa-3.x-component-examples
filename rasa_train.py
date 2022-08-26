#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-26 10:06
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    rasa_train.py
# @Project: rasa-3.x-component-examples
# @Package: 
# @Ref:
import os

import rasa


"""Runs Rasa Core and NLU training in `async` loop.
# 在“异步”循环中运行 Rasa Core 和 NLU 训练。Translating...
Args:
    domain: Path to the domain file.
    域：域文档的路径。
    config: Path to the config for Core and NLU.
    配置：核心和 NLU 配置的路径。
    training_files: Paths to the training data for Core and NLU.
    training_files：核心和 NLU 的训练数据的路径。
    output: Output path.
    输出：输出路径。
    dry_run: If `True` then no training will be done, and the information about
        whether the training needs to be done will be printed.
    dry_run：如果“True”，则不会进行任何训练，并且将打印有关是否需要完成培训的信息。

    force_training: If `True` retrain model even if data has not changed.
    force_training：如果“True”，即使数据没有更改，也会重新训练模型。
    fixed_model_name: Name of model to be stored.
    fixed_model_name：要存储的模型的名称。
    persist_nlu_training_data: `True` if the NLU training data should be persisted
        with the model.
    persist_nlu_training_data：如果 NLU 训练数据应与模型一起持久化，则为“True”。

    core_additional_arguments: Additional training parameters for core training.
    core_additional_arguments：核心训练的附加训练参数。

    nlu_additional_arguments: Additional training parameters forwarded to training
        method of each NLU component.
    nlu_additional_arguments：转发到每个NLU组件的训练方法的附加训练参数。

    model_to_finetune: Optional path to a model which should be finetuned or
        a directory in case the latest trained model should be used.
    model_to_finetune：应微调的模型的可选路径，或应使用最新训练的模型的目录。

    finetuning_epoch_fraction: The fraction currently specified training epochs
        in the model configuration which should be used for finetuning.
    finetuning_epoch_fraction：模型配置中当前指定的训练周期的分数，应用于微调。


Returns:
    An instance of `TrainingResult`.
"""
rasa.train(
    domain="domain.yml",
    config="config.yml",
    training_files="data",
)
