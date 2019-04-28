#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
便利関数
"""
from datetime import datetime as dt
from datetime import timedelta

def str_to_total_days(str_datetime, format='%Y-%m-%d %H:%M:%S'):
    '''
    1900-01-01 00:00:00 から str_datetime が何日経過しているかをfloatで返却する。
    
    EXAMPLE
    -------------
    str_datetime = '2019-04-01 12:34:56'
    str_to_total_days(str_datetime, format='%Y-%m-%d %H:%M:%S')
     > 17987.14925925926
    '''
    return dt.strptime(str_datetime, format).timestamp() / (60*60*24)

def total_days_to_str(total_days, format='%Y-%m-%d %H:%M:%S', kind='jst'):
    '''
    EXAMPLE
    -------------
    total_days = 17987.14925925926
    total_days_to_str(total_days, format='%Y-%m-%d %H:%M:%S')
     > '2019-04-01 12:34:56'
    '''
    if kind == 'utc':
        datetime = dt.utcfromtimestamp(total_days * (60*60*24))
    elif kind == 'jst':
        datetime = dt.utcfromtimestamp(total_days * (60*60*24)) + timedelta(hours=9)
    return datetime.strftime(format)

