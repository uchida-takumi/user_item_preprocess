#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統計計算を定義します。
"""

from scipy.stats import entropy
def list_entropy(list_of_ids):
    """
    リストに格納されたIDの多様性（エントロピー）を計算します。
    EXAMPLEs
    ------------
    list_of_ids = [1,1,2,3,2]
    list_entropy(list_of_ids)
     > 1.0549201679861442

    list_of_ids = ['1','1',5,'1','1']
    list_entropy(list_of_ids)
     > 0.5004024235381879

    list_of_ids = [1,2]
    list_entropy(list_of_ids)
     > 0.6931471805599453

    """
    _list_of_ids = list(list_of_ids) # np.arrayなども、listに変換する
    set_ = list(set(_list_of_ids))
    prob = [_list_of_ids.count(s) / len(_list_of_ids) for s in set_]
    return entropy(prob)


