#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages

# Cython が環境にインストールされているかどうかを確認し、ない場合は普通に *.c からコンパイルする。
try:
    from Cython.Distutils import build_ext
    IS_USE_CYTHON = True
except ImportError:
    IS_USE_CYTHON = False

if IS_USE_CYTHON:
    ext = '.pyx'
    cmdclass = {'build_ext':build_ext}
else:
    ext = '.c'
    cmdclass = {}

ext_modules = [
    Extension('user_item_preprocess.ID', # モジュール名
              sources=['src/user_item_preprocess/ID'+ext],
              )
]

setup(
    name="user_item_preprocess", # パッケージ名
    version="0.1.0",
    packages=['user_item_preprocess'],
    package_dir={'':'src'},
    ext_modules=ext_modules, # cythonモジュールがある場合は指定
    cmdclass={'build_ext': build_ext},
    install_requires=[],
    extras_require={},
    entry_points={}
)