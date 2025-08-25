# -*- coding: UTF-8 -*-
# !/usr/bin/env python3


"""
本文件实现了CDC(Searching Central Difference Convolutional
    Networks for Face Anti-Spoofing) OP: https://arxiv.org/pdf/2003.04092v1.pdf

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CDCConv2d(nn.Conv2d):
    """implementation of `Central Difference
    Convolutional <https://arxiv.org/pdf/2003.04092v1.pdf>

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 theta=0.7):
        super(CDCConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias)
        self.theta = torch.tensor(theta)

    def forward(self, input: Tensor) -> Tensor:
        out = F.conv2d(input, self.weight, self.bias,
                       self.stride, self.padding, self.dilation, self.groups)
        if torch.abs(self.theta - 0.0) < 1e-8:
            return out
        else:
            kernel_diff = self.weight.sum((2, 3), keepdim=True)
            out_diff = F.conv2d(input, kernel_diff, self.bias,
                                self.stride, 0, self.dilation, self.groups)
            out = out - self.theta * out_diff
            return out
