# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现了固定随机种子函数。

"""

import os
import torch
import random
import numpy as np


def seed_everywhere(seed=1):
    """set seed to everywhere.

    Args:
        seed (int): random seed.
    """
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)