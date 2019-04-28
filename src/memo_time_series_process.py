#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
# 本プログラムの目的
 (user_id, item_id, datetime) の (n, 3)サイズの配列データに対して、
 データ抽出速度を改善するための記述を確認する。

# 高速化の参考
[numpyとインデックスのススメ]
    http://kunai-lab.hatenablog.jp/entry/2018/04/08/134924
[Cythonと複合とかdtypeの取り扱いとか]
    http://sinhrks.hatenablog.com/entry/2015/07/11/223124
[公式：pandasの高速化]
    http://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html
"""

from datetime import datetime as dt
import numpy as np
import pandas as pd
from util_recommender_system import ID



############################
# 便利関数
def str_to_total_seconds(str_datetime, format='%Y-%m-%d %H:%M:%S'):
    '''
    EXAMPLE
    -------------
    str_datetime = '2019-04-01 12:34:56'
    str_to_total_seconds(str_datetime, format='%Y-%m-%d %H:%M:%S')
     > 1554089696
    '''
    return int(dt.strptime(str_datetime, format).timestamp())

def total_seconds_to_str(total_seconds, format='%Y-%m-%d %H:%M:%S'):
    '''
    EXAMPLE
    -------------
    total_seconds = 1554089696
    total_seconds_to_str(total_seconds, format='%Y-%m-%d %H:%M:%S')
     > '2019-04-01 12:34:56'
    '''
    datetime = dt.utcfromtimestamp(total_seconds)
    return dt.utcfromtimestamp(total_seconds).strftime(format)

####### TEST ############
df = pd.read_csv('tests/data/user_item_time.csv')
df['total_sec'] = np.vectorize(str_to_total_seconds)(df['datetime'].values)
df['datetime'] = pd.to_datetime(df['datetime'])
#########################



user_id = 'u_100'
item_id = 'i_100'
datetime = '2020-12-31'

def a(user_id, item_id, datetime):
    bo_index  = df['user_id'] == user_id
    bo_index &= df['item_id'] == item_id
    bo_index &= df['datetime'] < datetime
    return df.loc[bo_index, :]

def b(user_id, item_id, datetime):
    bo_index  = df.user_id == user_id
    bo_index &= df.item_id == item_id
    bo_index &= df.datetime < datetime
    return df.loc[bo_index, :]

def c(user_id, item_id, datetime):
    bo_index  = df.user_id.values == user_id
    bo_index &= df.item_id.values == item_id
    bo_index &= df.datetime < datetime
    return df.loc[bo_index, :]

def d(user_id, item_id, datetime):
    # bo_index処理で最速
    bo_index  = df.user_id.values == user_id
    bo_index &= df.item_id.values == item_id
    bo_index &= df.datetime.values < np.datetime64(datetime)
    return df.loc[bo_index, :]


def e(user_id, item_id, datetime):
    # .loc も .iloc も bo_index をとる場合は変わらない。
    bo_index  = df.user_id.values == user_id
    bo_index &= df.item_id.values == item_id
    bo_index &= df.datetime.values < np.datetime64(datetime)
    return df.iloc[bo_index, :]

def f(user_id, item_id, datetime):
    # 以下のようにnumpy.arrayから直接とると高速化する。
    bo_index  = df.user_id.values == user_id
    bo_index &= df.item_id.values == item_id
    bo_index &= df.datetime.values < np.datetime64(datetime)
    return df.datetime.values[bo_index]

# 文字型を明示する運用を行う
class g:
    def __init__(self, df):
        '''dtype未指定: それほど改善しない'''
        self.user_ids = df.user_id.values
        self.item_ids = df.item_id.values
        self.datetimes = df.datetime.values
    def run(self, user_id, item_id, datetime):
        bo_index  = self.user_ids == user_id
        bo_index &= self.item_ids == item_id
        bo_index &= self.datetimes < np.datetime64(datetime)
        return self.datetimes[bo_index]
        
class h:
    def __init__(self, df):
        '''dtype未指定: ID.transform をCython化したせいもあり、高速化された'''
        self.user_id_tf = ID.id_transformer()
        self.item_id_tf = ID.id_transformer()
        self.user_ids = np.array(self.user_id_tf.fit_transform(df.user_id.values), dtype=int)
        self.item_ids = np.array(self.item_id_tf.fit_transform(df.item_id.values), dtype=int)
        self.datetimes = df.datetime.values
    def run(self, user_id, item_id, datetime):
        bo_index  = self.user_ids == self.user_id_tf.transform([user_id])[0]
        bo_index &= self.item_ids == self.item_id_tf.transform([item_id])[0]
        bo_index &= self.datetimes < np.datetime64(datetime)
        return self.datetimes[bo_index]

# numba を使った高速化。 (→あまり速くはならなかった)
import numba
@numba.jit
def run_jit(user_ids, item_ids, datetimes, user_id, item_id, datetime):
    bo_index  = user_ids==user_id
    bo_index &= item_ids==item_id
    bo_index &= datetimes<np.datetime64(datetime)
    return bo_index        

class h_jit:
    def __init__(self, df):
        '''hの一部処理をjit化したもの。それほど早くなっていない。 '''
        self.user_id_tf = ID.id_transformer()
        self.item_id_tf = ID.id_transformer()
        self.user_ids = np.array(self.user_id_tf.fit_transform(df.user_id.values), dtype=int)
        self.item_ids = np.array(self.item_id_tf.fit_transform(df.item_id.values), dtype=int)
        self.datetimes = df.datetime.values
    def run(self, user_id, item_id, datetime):
        _user_id = self.user_id_tf.transform([user_id])[0]
        _item_id = self.item_id_tf.transform([item_id])[0]
        bo_index = run_jit(self.user_ids, self.item_ids, self.datetimes, _user_id, _item_id, datetime)
        return self.datetimes[bo_index]


class i:
    def __init__(self, df):
        '''dtype未指定: さらにdatetimeも整数化する
        　　→　これは逆に低速化した。時系列処理はnumpyに任せた方が良い'''
        self.user_id_tf = ID.id_transformer()
        self.item_id_tf = ID.id_transformer()
        self.user_ids = np.array(self.user_id_tf.fit_transform(df.user_id.values), dtype=int)
        self.item_ids = np.array(self.item_id_tf.fit_transform(df.item_id.values), dtype=int)
        self.datetimes = df.total_sec # df.datetime の整数管理
    def run(self, user_id, item_id, datetime):
        bo_index  = self.user_ids == self.user_id_tf.transform([user_id])[0]
        bo_index &= self.item_ids == self.item_id_tf.transform([item_id])[0]
        bo_index &= self.datetimes < str_to_total_seconds(datetime, format='%Y-%m-%d')
        return self.datetimes[bo_index]

class j:
    def __init__(self, df):
        '''h のrunの処理をtransform_single_idに置き換えて、list処理を省略した。あんまり早くならない'''
        self.user_id_tf = ID.id_transformer()
        self.item_id_tf = ID.id_transformer()
        self.user_ids = np.array(self.user_id_tf.fit_transform(df.user_id.values), dtype=int)
        self.item_ids = np.array(self.item_id_tf.fit_transform(df.item_id.values), dtype=int)
        self.datetimes = df.datetime.values
    def run(self, user_id, item_id, datetime):
        bo_index  = self.user_ids == self.user_id_tf.transform_single_id(user_id)
        bo_index &= self.item_ids == self.item_id_tf.transform_single_id(item_id)
        bo_index &= self.datetimes < np.datetime64(datetime)
        return self.datetimes[bo_index]

class k:
    def __init__(self, df):
        '''jの処理のbo_indexを入れ子方式にしてみた →　高速化　15.5/22.6'''
        self.user_id_tf = ID.id_transformer()
        self.item_id_tf = ID.id_transformer()
        self.user_ids = np.array(self.user_id_tf.fit_transform(df.user_id.values), dtype=int)
        self.item_ids = np.array(self.item_id_tf.fit_transform(df.item_id.values), dtype=int)
        self.datetimes = df.datetime.values
    def run(self, user_id, item_id, datetime):
        bo_index00 = self.user_ids == self.user_id_tf.transform_single_id(user_id)
        bo_index01 = self.item_ids[bo_index00] == self.item_id_tf.transform_single_id(item_id)
        bo_index02 = self.datetimes[bo_index00][bo_index01] < np.datetime64(datetime)
        return self.datetimes[bo_index00][bo_index01][bo_index02]

# あらかじめ sort していたら処理時間が変わるかの検証
array = np.random.choice(range(10000), size=1000000)
sorted_array = array.copy()
sorted_array.sort()
def sp():
    return array[array==0]
def sps():
    return sorted_array[sorted_array==0]
%timeit sp()
%timeit sps()
# [結論]　速度は変わらず。

# 次に k に対して、Cythonを施した場合の速度を検証。
%load_ext cython

%%cython
import numpy as np
from util_recommender_system import ID
from numpy cimport ndarray
cpdef cython_run(
        ndarray[long, ndim=1] user_ids,
        ndarray[long, ndim=1] item_ids,
        ndarray datetimes,
        int user_id,
        int item_id,
        str datetime                
        ):
        bo_index00 = user_ids == user_id
        bo_index01 = item_ids[bo_index00] == item_id
        bo_index02 = datetimes[bo_index00][bo_index01] < np.datetime64(datetime)
        return datetimes[bo_index00][bo_index01][bo_index02]
        
class k_cy:
    def __init__(self, df):
        '''ここまでで最速だったkをCythonを応用してみたが、速度は悪化した'''
        self.user_id_tf = ID.id_transformer()
        self.item_id_tf = ID.id_transformer()
        self.user_ids = np.array(self.user_id_tf.fit_transform(df.user_id.values), dtype=int)
        self.item_ids = np.array(self.item_id_tf.fit_transform(df.item_id.values), dtype=int)
        self.datetimes = df.datetime.values
    def run(self, user_id, item_id, datetime):
        _user_id = self.user_id_tf.transform_single_id(user_id)
        _item_id = self.item_id_tf.transform_single_id(item_id)
        return cython_run(self.user_ids, self.item_ids, self.datetimes, 
                          _user_id, _item_id, datetime)

class l:
    def __init__(self, df):
        '''jのbo_index を整数indexにしてみた →　高速化　13.0/15.5'''
        self.user_id_tf = ID.id_transformer()
        self.item_id_tf = ID.id_transformer()
        self.user_ids = np.array(self.user_id_tf.fit_transform(df.user_id.values), dtype=int)
        self.item_ids = np.array(self.item_id_tf.fit_transform(df.item_id.values), dtype=int)
        self.datetimes = df.datetime.values
    def run(self, user_id, item_id, datetime):
        index00 = np.where(self.user_ids == self.user_id_tf.transform_single_id(user_id))
        index01 = np.where(self.item_ids[index00] == self.item_id_tf.transform_single_id(item_id))
        index02 = np.where(self.datetimes[index00][index01] < np.datetime64(datetime))
        return self.datetimes[index00][index01][index02]


# 速度検証
print(a(user_id, item_id, datetime))
%timeit a(user_id, item_id, datetime)

print(b(user_id, item_id, datetime))
%timeit b(user_id, item_id, datetime)

print(c(user_id, item_id, datetime))
%timeit c(user_id, item_id, datetime)

print(d(user_id, item_id, datetime))
%timeit d(user_id, item_id, datetime)

print(e(user_id, item_id, datetime))
%timeit e(user_id, item_id, datetime)

print(f(user_id, item_id, datetime))
%timeit f(user_id, item_id, datetime)

self = g(df)
print(self.run(user_id, item_id, datetime))
%timeit self.run(user_id, item_id, datetime)

self = h(df)
print(self.run(user_id, item_id, datetime))
%timeit self.run(user_id, item_id, datetime)

self = i(df)
print(self.run(user_id, item_id, datetime))
%timeit self.run(user_id, item_id, datetime)

self = j(df)
print(self.run(user_id, item_id, datetime))
%timeit self.run(user_id, item_id, datetime)

self = h_jit(df)
print(self.run(user_id, item_id, datetime))
%timeit self.run(user_id, item_id, datetime)

self = k(df)
print(self.run(user_id, item_id, datetime))
%timeit self.run(user_id, item_id, datetime)

self = k_cy(df)
print(self.run(user_id, item_id, datetime))
%timeit self.run(user_id, item_id, datetime)

self = l(df)
print(self.run(user_id, item_id, datetime))
%timeit self.run(user_id, item_id, datetime)

# マルチプロセス　の稼働検証
from multiprocessing import Pool
import multiprocessing as multi

h_jit_ins = h_jit(df)
h_ins = h(df)
def proccess_f(args):
    return f(*args)
def proccess_h(args):
    return h_ins.run(*args)
def proccess_h_jit(args):
    return h_jit_ins.run(*args)
p = Pool(5)
list_of_arg = [['u_001', 'i_001', '2020-12-31']] * 10000
%timeit result = [f(*arg) for arg in list_of_arg] # シングルスレッド
%timeit result = p.map(proccess_f, list_of_arg) 
%timeit result = p.map(proccess_h, list_of_arg) 
%timeit result = p.map(proccess_h_jit, list_of_arg) 
p.close()


