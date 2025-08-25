# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现了指定路径模型加载。

"""

import os
import torch
import copy


def module_name(name: str):
    if name.startswith('module.'):
        return name[7:]
    else:
        return 'module.' + name

def check_state_sizes(model_sizes, loaded_sizes):
    model_keys = {}
    # try to match with key
    for name in model_sizes:
        if name in loaded_sizes:
            model_keys[name] = name
        elif module_name(name) in loaded_sizes:
            model_keys[name] = module_name(name)
    if len(model_keys) > 0:
        return model_keys
    # try to match with size
    for mk, lk in zip(model_sizes, loaded_sizes):
        if model_sizes[mk] != loaded_sizes[lk]:
            return model_keys
        model_keys[mk] = lk
    else:
        return model_keys



def load_model(load_path, model, optimizer = None, allow_size_mismatch = True):
    """
    Load model from file
    If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
    If optimizer not passed, only the model weights will be loaded
    """

    # -- load dictionary
    assert os.path.isfile( load_path ), "Error when loading the model, provided path not found: {}".format( load_path )
    try:
        checkpoint = torch.load(load_path)
    except:
        checkpoint = torch.load(load_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        loaded_state_dict = checkpoint['state_dict']
    elif 'teacher' in checkpoint:
        loaded_state_dict = checkpoint['teacher']
    else:
        loaded_state_dict = checkpoint

    if allow_size_mismatch:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        loaded_sizes = { k: v.shape for k,v in loaded_state_dict.items() }
        model_state_dict = model.state_dict()
        model_sizes = { k: v.shape for k,v in model_state_dict.items() }
        model_keys = check_state_sizes(model_sizes, loaded_sizes)
        for k, v in model_keys.items():
            new_state_dict[k] = loaded_state_dict[v]
        loaded_state_dict = new_state_dict

    # -- copy loaded state into current model and, optionally, optimizer
    print(model.load_state_dict(loaded_state_dict, strict = not allow_size_mismatch))
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer
    return model
