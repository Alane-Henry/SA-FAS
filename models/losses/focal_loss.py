# -*- coding: UTF-8 -*-

"""
本文件实现了Focal Loss。

"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss.
    This is an implementation of paper `Focal Loss for Dense Object
    Detection <https://arxiv.org/pdf/1708.02002.pdf>`.
    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """
    def __init__(self,
                 gamma=2,
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target):
        """forward"""
        logp = self.ce(pred, target)
        pt = torch.exp(-logp)
        loss = (1 - pt) ** self.gamma * logp
        return loss.mean()

