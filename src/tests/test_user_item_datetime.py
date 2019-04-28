#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
テストコマンド
 > python -m unittest src.tests.test_user_item_datetime
'''


import unittest
from user_item_preprocess.user_item_datetime import preprocesser

class TEST01(unittest.TestCase):
    def setUp(self):
        user_ids  = [1,1,1,1,1,
                     2,2,2]
        item_ids  = [1,2,1,1,1,
                     3,1,3]
        datetimes = ['2019-01-01','2019-02-01','2019-03-01','2019-04-01','2019-05-01',
                     '2019-01-01','2019-02-01','2019-03-01']
        datetime_format = '%Y-%m-%d'
        
        self.user_item_datetime = preprocesser(user_ids, item_ids, datetimes, datetime_format)

    
    def test01_01(self):
        user_id = 1
        item_id = 1
        datetime = '2019-04-15'
        diff_days = [7,30,60,90,120,150]
        
        result = self.user_item_datetime.get_past_cnt_by_user_item_datetime(
                user_id, item_id, datetime, diff_days)
        
        # Assertion
        self.assertEqual(result['past_all_cnt'],    3)
        self.assertEqual(result['past_150day_cnt'], 3)
        self.assertEqual(result['past_120day_cnt'], 3)
        self.assertEqual(result['past_90day_cnt'],  2)
        self.assertEqual(result['past_60day_cnt'],  2)
        self.assertEqual(result['past_30day_cnt'],  1)
        self.assertEqual(result['past_7day_cnt'],   0)
        

    def tearDown(self):
        pass


if __name__ == '__main__':
    print("""このテストを実行する前に、以下のコマンドを実行して変更したプログラムを反映してください。
          > cd [setup.py があるディレクトリ]
          > pip install -U . 
          """)
    unittest.main()