# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现了Torch函数库字典形式调用。

"""

from .config import Config
from .fileio import load, dump
from .logger import get_root_logger
from .seed import seed_everywhere
from .dist import get_rank, get_world_size, init_distributed
from .cv_util import distance_pt
from .load_model import load_model


__all__ = ['Config', 'load', 'dump', 'get_root_logger', 'seed_everywhere',
           'get_rank', 'get_world_size', 'init_distributed', 'distance_pt']
