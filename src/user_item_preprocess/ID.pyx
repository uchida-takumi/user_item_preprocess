#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高速化のためにCython化しました。
setup.py で ID.c, ID.cpython-*.so への変換、コンパイルを定義しています。
以下のコマンドで、コンパイルされ、それ以降は通常のpythonモジュールのように、pythonコード中でimport
できます。

# コンパイルコマンド(--inplace オプションによりカレントディレクトリに.soファイルが出力)
python setup.py build_ext --inplace

# pythonコマンド上でのimport方法
import ID
"""

import os
import json
import pickle
cdef class id_transformer:
    cdef public str SAVE_FILE_NAME_id_convert_dict
    cdef public str SAVE_FILE_NAME_inverse_id_convert_dict
    cdef public dict id_convert_dict
    cdef public dict inverse_id_convert_dict
    
    def __init__(self):
        """
        transform ids to the index which start from 0.
        """
        self.SAVE_FILE_NAME_id_convert_dict = 'id_convert_dict.pickle'
        self.SAVE_FILE_NAME_inverse_id_convert_dict = 'inverse_id_convert_dict.pickle'

    def save(self, dir='/tmp'):
        """
        Save this instance as files.
        Cython class instance can't be pickled directory.
        So, to save or load the instance, use self.save
        
        ARGUMENT
        -------------
        dir [str]:
            path of the directory where to save.
        """
        f_name = os.path.join(dir, self.SAVE_FILE_NAME_id_convert_dict)
        with open(f_name, 'wb') as f:
            pickle.dump(self.id_convert_dict, f)
        f_name = os.path.join(dir, self.SAVE_FILE_NAME_inverse_id_convert_dict)
        with open(f_name, 'wb') as f:
            pickle.dump(self.inverse_id_convert_dict, f)
        
    def load(self, dir='/tmp'):
        """
        Load saved instance.
        Cython class instance can't be pickled directory.
        So, to save or load the instance, use self.save

        ARGUMENT
        -------------
        dir [str]:
            path of the directory where to load.
        """
        f_name = os.path.join(dir, self.SAVE_FILE_NAME_id_convert_dict)
        self.id_convert_dict = pickle.load(open(f_name, 'rb'))
        f_name = os.path.join(dir, self.SAVE_FILE_NAME_inverse_id_convert_dict)
        self.inverse_id_convert_dict = pickle.load(open(f_name, 'rb'))

    cpdef fit(self, ids):
        """
        ARGUMETs:
            ids [array-like object]: 
                array of id of user or item.        
        """
        cdef list ids_ = sorted(list(set(ids)))
        self.id_convert_dict = {i:index for index,i in enumerate(ids_)}
        self.inverse_id_convert_dict = {item:key for key,item in self.id_convert_dict.items()}
    
    cpdef list transform(self, ids, unknown=None):
        """
        ARGUMETs:
            ids [array-like object]: 
                array of id of user or item.                
        """
        return [self.id_convert_dict.get(i, unknown) for i in ids]

    cpdef transform_single_id(self, single_id, unknown=None):
        return self.id_convert_dict.get(single_id, unknown)

    cpdef list fit_transform(self, ids):
        self.fit(ids)
        return self.transform(ids)
        
    cpdef list inverse_transform(self, indexes, unknown=None):
        """
        ARGUMETs:
            indexes [array-like object]: 
                array of index which are transformed by self.fit              
        """
        return [self.inverse_id_convert_dict.get(ind, unknown) for ind in indexes]

    cpdef inverse_transform_single_id(self, single_index, unknown=None):
        return self.inverse_id_convert_dict.get(single_index, unknown)
    
    cpdef fit_update(self, ids):
        """
        ARGUMETs:
            ids [array-like object]: 
                array of id of user or item.        
        """
        cdef list ids_ = sorted(list(set(ids)))
        ids_ = [id_ for id_ in ids_ if id_ not in self.id_convert_dict.keys()]
        now_max_id = max(self.id_convert_dict.values())

        new_id_convert_dict = {i:now_max_id+1+index for index,i in enumerate(ids_)}
        self.id_convert_dict.update(new_id_convert_dict)
        inverse_new_id_convert_dict = {item:key for key,item in new_id_convert_dict.items()}
        self.inverse_id_convert_dict.update(inverse_new_id_convert_dict)
