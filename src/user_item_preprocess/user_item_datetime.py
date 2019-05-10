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
        
    def get_past_cnt_by_user_item_datetime(self, user_id, item_id, datetime, diff_days=[7,30,90], is_cut=False):
        """
        user_id, item_id, datetime を指定して、その組み合わせの過去データが何個あるかを
        カウントする。diff_daysを指定することで、何日前までをカウントするかを指定できる。
        
        ARGUMENTs
        -----------------
        user_id:
            ユーザーID
        item_id:
            アイテムID
        datetime [str]:
            日付の文字列。　ex) '2019-01-01 12:13:14'
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
        # 入力を内部処理用にint変換する
        _user_id = self.user_id_tf.transform_single_id(user_id)
        _item_id = self.item_id_tf.transform_single_id(item_id)
        _datetime = self.str_to_total_days(datetime)
        
        # 検索条件に合致するデータのindexを取得する
        index00 = np.where(self.user_ids == _user_id)
        index01 = np.where(self.item_ids[index00] == _item_id)
        index02 = np.where(self.datetimes[index00][index01] < _datetime)
        
        # 合致したデータを集計処理する
        _datetimes = self.datetimes[index00][index01][index02]
        diff_datetimes = _datetime - _datetimes

        return self._get_past_cnt_dict(diff_datetimes, diff_days)
    
    def get_past_cnt_by_user_datetime(self, user_id, datetime, diff_days=[7,30,90], is_cut=False):
        """
        user_id, datetime を指定して、その組み合わせの過去データが何個あるかを
        カウントする。 item_idの違いは考慮しない。
        diff_daysを指定することで、何日前までをカウントするかを指定できる。
        
        ARGUMENTs
        -----------------
        user_id:
            ユーザーID
        datetime [str]:
            日付の文字列。　ex) '2019-01-01 12:13:14'
        diff_days [list of int]:
            何日前ごとにカウントするかの指定。　ex) [7,30,90]
        is_cut [bool]:
            Trueの時にdiff_daysを区間とみなしてカウントする。
            例えば、dff_days=[7,30,90]なら、
            0-7日前、7-30日前、30-90日前のデータをカウントする。
        
        RETURN
        -----------------
        {
            7: 0, # datetimeから過去7日間には0個。
            30: 1, 
            90: 3, 
        }
        """
        # 入力を内部処理用にint変換する
        _user_id = self.user_id_tf.transform_single_id(user_id)
        _datetime = self.str_to_total_days(datetime)
        
        # 検索条件に合致するデータのindexを取得する
        index00 = np.where(self.user_ids == _user_id)
        index02 = np.where(self.datetimes[index00] < _datetime)
        
        # 合致したデータを集計処理する
        _datetimes = self.datetimes[index00][index02]
        diff_datetimes = _datetime - _datetimes
        
        # 
        past_cnt_dict = self._get_past_cnt_dict(diff_datetimes, diff_days)
        if is_cut:
            past_cnt_dict = self._cut(past_cnt_dict)
        return past_cnt_dict
    
    def _get_past_cnt_dict(self, diff_datetimes, diff_days):
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
            
    
        
        
