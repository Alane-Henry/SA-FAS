# -*- coding: UTF-8 -*-

"""
本文件实现了KL Div loss。

"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDLoss(nn.Module):
    """KLDivLoss.

    Args:
        temp_factor (float, optional): temperature scaling.
        loss_weight (float, optional): The weight of loss.

    """

    def __init__(self,
                 temp_factor=4.0,
                 loss_weight=1.0):
        super(KLDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.loss_weight = loss_weight
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, pred, target):
        """ forward"""
        log_p = torch.log_softmax(pred / self.temp_factor, dim=1)
        loss = self.kl_div(log_p, target) * (self.temp_factor ** 2) / pred.size(0)

        return loss


class BKLDLoss(nn.Module):
    """Binary Kullback-Leibler divergence loss.

    Args:
        loss_weight (float, optional): The weight of loss.

    """

    def __init__(self,
                 loss_weight=1.0):
        super(BKLDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logit_q, logit_p):
        """ forward"""
        logit_q = torch.softmax(logit_q, dim=1)
        logit_p = torch.softmax(logit_p, dim=1)
        loss = self.kl_div(logit_q.log(), logit_p) + self.kl_div((1 - logit_q).log(), 1 - logit_p)

        return loss
