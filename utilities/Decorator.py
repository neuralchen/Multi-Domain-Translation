#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Decorator.py
# Created Date: Tuesday September 24th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 28th September 2019 12:46:42 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################




import time

def time_it(fn):
    def new_fn(*args):
        start = time.time()
        result = fn(*args)
        end = time.time()
        duration = end - start
        print('%.4f seconds are consumed in executing function:%s'\
              %(duration, fn.__name__))
        return result
    return new_fn