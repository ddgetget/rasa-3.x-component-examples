#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-26 10:05
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    rasa_run.py
# @Project: rasa-3.x-component-examples
# @Package: 
# @Ref:
import os

import rasa


rasa.run(
    model="models",
    endpoints="endpoints.yml",
    credentials="credentials.yml"
)
