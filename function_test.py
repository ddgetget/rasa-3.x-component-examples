#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-08-24 16:51
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    function_test.py
# @Project: rasa-3.x-component-examples
# @Package: 
# @Ref:


def only_alphanum_test_parse(text):
    result = "".join([c for c in text if ((c == " ") or str.isalnum(c))])
    # str.isalnum(c): 如果字符串是字母数字字符串，则返回 True，否则返回 False。
    print(result)
    return result


if __name__ == '__main__':
    only_alphanum_test_parse(text="hello world")
    print(["sgadh" for c in [1, 2] if (1 == 1)])
