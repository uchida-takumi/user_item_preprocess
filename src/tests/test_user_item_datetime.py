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
        
        result = self.user_item_datetime.get_past_cnt(
                datetime, user_id, item_id, diff_days)
        
        # Assertion
        self.assertEqual(result[150], 3)
        self.assertEqual(result[120], 3)
        self.assertEqual(result[90],  2)
        self.assertEqual(result[60],  2)
        self.assertEqual(result[30],  1)
        self.assertEqual(result[7],   0)

    def test01_02(self):
        user_id = 1
        item_id = None
        datetime = '2019-04-15'
        diff_days = [7,30,60,90,120,150]
        
        result = self.user_item_datetime.get_past_cnt(
                datetime, user_id, item_id, diff_days)
        
        # Assertion
        self.assertEqual(result[150], 4)
        self.assertEqual(result[120], 4)
        self.assertEqual(result[90],  3)
        self.assertEqual(result[60],  2)
        self.assertEqual(result[30],  1)
        self.assertEqual(result[7],   0)
        
    def test01_03(self):
        user_id = None
        item_id = 1
        datetime = '2019-04-15'
        diff_days = [7,30,60,90,120,150]
        
        result = self.user_item_datetime.get_past_cnt(
                datetime, user_id, item_id, diff_days)
        
        # Assertion
        self.assertEqual(result[150], 4)
        self.assertEqual(result[120], 4)
        self.assertEqual(result[90],  3)
        self.assertEqual(result[60],  2)
        self.assertEqual(result[30],  1)
        self.assertEqual(result[7],   0)


    def tearDown(self):
        pass


if __name__ == '__main__':
    print("""このテストを実行する前に、以下のコマンドを実行して変更したプログラムを反映してください。
          > cd [setup.py があるディレクトリ]
          > pip install -U . 
          """)
    unittest.main()