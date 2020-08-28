#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pickle import load, dump
import os


def pickle_load(path):
    """
    Load Pickle

    Wrapper for loading pickle files.

    Parameters
    ----------
    path : str
        String containing the path to the Pickle file to be loaded.
    """
    with open(path, 'rb') as f:
        return load(f)


def pickle_dump(obj, path):
    """
    Dump Pickle

    Wrapper for serializing Python objects as Pickle files.

    Parameters
    ----------
    obj : Python object
        Python object to be serialized.
    path : str
        String containing the path to the Pickle file to be created.
    """
    with open(path, 'wb') as f:
        dump(obj, f)


def reduce_memory(df):
    """
    Memory Consumption Reduction

    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.

    Parameters
    ----------
    df : pandas DataFrame

    Returns
    -------
    pandas DataFrame
        Memory-optimized pandas object
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min \
                        and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min \
                        and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min \
                        and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min \
                        and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min \
                        and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min \
                        and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'
          .format(100 * (start_mem - end_mem) / start_mem))
    return df


def get_prj_path(proj_name):
    """
    Project Absolute Path

    Returns a string describing the absolute path of the project root folder.

    Parameters
    ----------
    proj_name : str
        String describing the project root folder name.
    """
    curr = os.getcwd().split('/')
    while curr[-1] != proj_name:
        curr.pop()
    path = '/'.join(curr)
    return path
