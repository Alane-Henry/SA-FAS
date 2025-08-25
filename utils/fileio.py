# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现了文件的读写io接口函数：load and dump。

"""

import json
import pickle
import pandas
import numpy as np
from pathlib import Path


def load(file, file_format=None, **kwargs):
    """Load data from json/pickle/txt/csv files.

    This method provides a unified api for loading data from serialized files.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "txt", "pickle/pkl".

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        file_format = file.split('.')[-1]

    if file_format == 'json':
        with open(file, 'r') as f:
            obj = json.load(f, **kwargs)
        return obj
    elif file_format in ['pkl', 'pickle']:
        with open(file, 'rb') as f:
            obj = pickle.load(f, **kwargs)
        return obj
    elif file_format == 'csv':
        obj = pandas.read_csv(file, header=None, **kwargs)
        return np.array(obj)
    elif 'np' in file_format:
        return np.load(file)
    else:
        obj = [line.strip().split() for line in open(file).readlines()]
        return np.array(obj, dtype=object)


def dump(obj, file, file_format=None, **kwargs):
    """Dump data to json/pickle/txt/csv strings or files.

    This method provides a unified api for dumping data as strings or to files.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object): Dump to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        file_format = file.split('.')[-1]
    if file_format not in ['txt', 'json', 'pkl', 'pickle']:
        raise TypeError(f'Unsupported format: {file_format}')

    if file_format == 'json':
        with open(file, 'w') as f:
            json.dump(obj, f)
    elif file_format in ['pkl', 'pickle']:
        with open(file, 'wb') as f:
            pickle.dump(obj, f)
    elif file_format == 'txt':
        obj = pandas.DataFrame(np.array(obj))
        obj.to_csv(file, sep=' ', header=False, index=False, **kwargs)
    elif file_format == 'csv':
        obj = pandas.DataFrame(np.array(obj))
        obj.to_csv(file, header=False, index=False, **kwargs)

