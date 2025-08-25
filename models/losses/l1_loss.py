# -*- coding: UTF-8 -*-

"""
本文件实现了L1 Loss 和 Smooth L1 Loss。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 margin=0.2):
        super(L1Loss, self).__init__()
        assert reduction in ('mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.margin = margin
        self.eps = 1e-16

    def forward(self, pred, target):
        """forward"""
        assert pred.size() == target.size()
        if self.reduction == 'sum':
            loss = torch.sum(torch.abs(pred - target))
        else:
            num_reg = torch.sum(torch.logical_not(torch.eq(pred, target)))
            loss = torch.sum(F.relu(torch.abs(pred - target)- self.margin),) / (num_reg + self.eps)
        return loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """
    def __init__(self,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        assert beta > 0
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def _smooth_l1(self, pred, target, beta=1.0):
        """smooth l1 function"""
        diff = torch.abs(pred - target)
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        return loss

    def forward(self, pred, target):
        """forward"""
        assert pred.size() == target.size() and target.numel() > 0
        if self.reduction == 'sum':
            loss = torch.sum(self._smooth_l1(pred, target, self.beta))
        else:
            loss = torch.mean(self._smooth_l1(pred, target, self.beta))
        return loss
