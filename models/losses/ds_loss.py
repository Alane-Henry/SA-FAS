# -*- coding: UTF-8 -*-

"""
本文件实现了Dense similarity loss。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSLoss(nn.Module):
    """Triplet Loss.

    This is an implementation of paper `Consistency Regularization
    for Deep Face Anti-Spoofing <https://arxiv.org/abs/2111.12320>`.

    Args:
        with_norm (bool, optional): Whether to normalize the predict vector
            before calculate loss.
        loss_weight (float, optional): The weight of loss.

    """

    def __init__(self,
                 with_norm=True,
                 loss_weight=1.0):
        super(DSLoss, self).__init__()
        self.with_norm = with_norm
        self.loss_weight = loss_weight

    def forward(self, h, f):
        """ forward
        Args:
             H (Variance): shape of (batch_size, C, H, W).
             F (Variance): shape of (batch-size, C, H, W)
        """
        n, c, sh, sw = h.shape
        h = F.normalize(h.view(n, c, sh * sw).permute(0, 2, 1), dim=2)
        f = F.normalize(f.view(n, c, sh * sw), dim=1)

        loss = 1 - torch.bmm(h, f).mean()

        return loss