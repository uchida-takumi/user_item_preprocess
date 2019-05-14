#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
このクラスでは、[user_id, item_id, datetime] の3次元行列データの効率的な処理をまとめた。
効率化にはnumpyを利用した。
Cython, numba は利用していない。numpyを効率的に使った場合はそれらによって改善しなかった。
ただし、ID情報のint化管理を行うモジュール ID は Cython によって高速化を行った。
'''

import numpy as np
from user_item_preprocess import ID
from user_item_preprocess import util


''' test code
import pandas as pd
df = pd.read_csv('tests/data/user_item_time.csv')

user_ids  = df['user_id']
item_ids  = df['item_id']
datetimes = df['datetime']

user_id = 'u_100'
item_id = 'i_100'
datetime = '2021-01-18 18:19:16'

self = preprocesser(user_ids, item_ids, datetimes)
self.get_past_cnt_by_user_item_datetime(user_id, item_id, datetime)
self.get_past_cnt_by_user_datetime(user_id, datetime)
'''

class preprocesser:
    def __init__(self, user_ids, item_ids, datetimes, datetime_format='%Y-%m-%d %H:%M:%S'):
        """
        このクラスでは、[user_id, item_id, datetime] の3次元行列データの効率的な処理をまとめた。
        
        ARUMENTs
        --------------------
        user_ids [array like object]: 
            user_id の1次元の配列で、要素数はサンプル数だけある。
        item_ids [array like object]: 
            item_id の1次元の配列で、要素数はサンプル数だけある。
        datetimes [array like object which element is str]: 
            datetimes の1次元の配列で、要素数はサンプル数だけある。
            要素はdatetime_formatと同じ日付形式のstrである必要がある。        
        * user_ids, item_ids は内部的にint型に変換されて管理される。 
        * datetimes は内部的にfloat型の日数に変換されて管理される。 
        datetime_format [str]:
            datetimes の日付形式のstr
        """
        self.str_to_total_days = lambda datetime : util.str_to_total_days(datetime, datetime_format) 
        self.total_days_to_str = lambda datetime : util.total_days_to_str(datetime, datetime_format) 
        self.user_id_tf = ID.id_transformer()
        self.item_id_tf = ID.id_transformer()
        self.user_ids = np.array(self.user_id_tf.fit_transform(user_ids), dtype=int)
        self.item_ids = np.array(self.item_id_tf.fit_transform(item_ids), dtype=int)
        vfunc = np.vectorize(self.str_to_total_days)
        self.datetimes = vfunc(datetimes)
        
    
    def get_past_cnt(self, datetime, user_id=None, item_id=None, diff_days=[7,30,90], is_cut=False):
        """
        user_id, item_id, datetime を指定して、その組み合わせの過去データが何個あるかを
        カウントする。diff_daysを指定することで、何日前までをカウントするかを指定できる。
        
        ARGUMENTs
        -----------------
        datetime [str]:
            日付の文字列。　ex) '2019-01-01 12:13:14'
        user_id:
            ユーザーID
        item_id:
            アイテムID
        diff_days [list of int]:
            何日前ごとにカウントするかの指定。　ex) [7,30,90]
        is_cut [bool]:
            Trueの時にdiff_daysを区間とみなしてカウントする。
            例えば、dff_days=[7,30,90]なら、
            0-7日前、7-30日前、30-90日前、90-inf日前のデータをカウントする。
        
        EXAMPLE of RETURN
        -----------------
        {
            7: 0, # datetimeから過去7日間には0個。
            30: 1, 
            90: 3, 
        }
        """
        # 入力を内部処理用に変換する。
        _user_id, _item_id, _datetime = self._transform_inputs(user_id, item_id, datetime)
        
        # 組み合わせを満たすindexを取得する。
        indexes = self._get_index(_user_id, _item_id, _datetime)
        
        # 得られたindexに対応する配列datetimesを取得
        _datetimes = self._np_array_roop_index(self.datetimes, indexes)

        # 入力されたdatetimeとの時差をとる。
        diff_datetimes = _datetime - _datetimes
        
        # 集計        
        past_cnt_dict = self._get_past_cnt_dict(diff_datetimes, diff_days)
        
        # is_cut に応じて、区間カウントする。
        if is_cut:
            past_cnt_dict = self._cut(past_cnt_dict)

        return past_cnt_dict


    
    def _transform_inputs(self, user_id=None, item_id=None, datetime=None):
        """
        クライアントから入力されるuser_id, item_id, datetimeを内部処理用に変換する。
        Noneで渡された場合はNoneで返却する。
        """
        _user_id, _item_id, _datetime = None, None, None
        if user_id is not None:            
            _user_id = self.user_id_tf.transform_single_id(user_id)
        if item_id is not None:
            _item_id = self.item_id_tf.transform_single_id(item_id)
        if datetime is not None:                            
            _datetime = self.str_to_total_days(datetime)
        return _user_id, _item_id, _datetime
        
    
    def _np_array_roop_index(self, np_array, indexes):
        """
        numpy.arrayのインデックス処理を高速化するための実装実装。
        
        EXAMPLEs
        --------------
        # 入力
        np_array = np.array(range(10))
        indexes = [[3,4,5], None, [1,2]]
        
        # 実現したい挙動
        answer = np_array[[3,4,5]][:][[1,2]]
        print(answer)
         > array([4, 5])
        
        # 関数として実行
        result = self._np_array_roop_index(np_array, indexes)
        print(result)
         > array([4, 5])
        """
        _np_array = np_array[:]
        for index in indexes:
            if index is not None:
                _np_array = _np_array[index]
        return _np_array
            
        
    
    def _get_index(self, _user_id=None, _item_id=None, _datetime=None):
        """
        入力された_user_id, _item_id, _datetimeの組み合わせを満たすindexを返却する。
        入力は全てself._transform_inputs()で変換済みのもの。
        ただし、datetimeの条件は、self.datetimes < _datetime である。
        
        内部の処理が複雑になっているのは、なるべくメモリへの問い合わせ量を減らして高速化するため。
        """
        index_user, index_item, index_date = None, None, None
        # 入力を内部処理用に変換する
        if _user_id is not None:
            index_user = np.where(self.user_ids == _user_id)            
        if _item_id is not None:
            _item_ids = self._np_array_roop_index(self.item_ids, [index_user])
            index_item = np.where(_item_ids == _item_id)
        if _datetime is not None:
            _datetimes = self._np_array_roop_index(self.datetimes, [index_user, index_item])
            index_date = np.where(_datetimes < _datetime)                        

        return index_user, index_item, index_date
                                
        
    def _get_past_cnt_dict(self, diff_datetimes, diff_days):
        '''
        diff_days に指定された日数ごとに、diff_datetimesを集計する。
        '''
        past_cnt_dict = dict()
        if diff_days:
            for diff_day in diff_days:
                past_day_cnt = (diff_datetimes < diff_day).sum()
                #past_cnt_dict['past_{}day_cnt'.format(diff_day)] = past_day_cnt
                past_cnt_dict[diff_day] = past_day_cnt                
        return past_cnt_dict

    def _cut(self, past_cnt_dict):
        '''
        
        
        EXAMPLE
        -----------------
        past_cnt_dict = {7:10, 30:25, 90:50}
        self._convert_cut(past_cnt_dict)
         > {7: 10, 30: 15, 90: 25}
        '''
        _diff_days = sorted(past_cnt_dict.keys())
        _past_cnt_dict = {}
        for i, d in enumerate(_diff_days):
            if i == 0:
                _past_cnt_dict[d] = past_cnt_dict[d]
            else:
                _past_cnt_dict[d] = past_cnt_dict[d] - past_cnt_dict[_diff_days[i-1]]
        return _past_cnt_dict
            
    
        
        
