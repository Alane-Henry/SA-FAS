# -*- coding: UTF-8 -*-

"""
本文件实现了Torch.nn.initialize函数库调用。

"""

import torch.nn as nn


def constant_init(module, val, bias=0):
    """Constant initialize.

    Args:
        module (nn.Module): torch.nn.Module layer to be initialized.
        val (int): initialize value.
        bias (int): initialize bias
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """Xavier initialize.

    Args:
        module (nn.Module): torch.nn.Module layer to be initialized.
        gain (int): initialize gain.
        bias (int): initialize bias
    """
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    """Normal initialize.

    Args:
        module (nn.Module): torch.nn.Module layer to be initialized.
        mean (int): the mean of the normal distribution.
        std (int): the standard deviation of the normal distribution.
        bias (int): initialize bias.
    """
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    """Uniform initialize.

    Args:
        module (nn.Module): torch.nn.Module layer to be initialized.
        a (int): the lower bound of the uniform distribution.
        b (int): the upper bound of the uniform distribution.
        bias (int): initialize bias.
    """
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, bias=0, mode='fan_out', nonlinearity='relu', distribution='normal'):
    """Kaiming initialize.

    Args:
        module (nn.Module): torch.nn.Module layer to be initialized.
        a (int): the lower bound of the uniform distribution.
        bias (int): initialize bias.
        mode (str): either ``'fan_in'`` (default) or ``'fan_out'``.
        nonlinearity (str): the non-linear function (`nn.functional` name).
    """
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
