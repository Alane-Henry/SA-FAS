# -*- coding: UTF-8 -*-

"""
本文件实现的Maximum Mean Discrepancy Loss(MMD Loss)。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy Loss.

    Args:
        kernel_mul (float, optional): bandwidth.
        kernel_num (int, optional): number of kernel.
        with_norm (bool, optional): Whether to normalize the predict vector
            before calculate loss.
        loss_weight (float, optional): The weight of loss.

    """

    def __init__(self,
                 kernel_mul=2.0,
                 kernel_num=5.0,
                 fix_sigma=None,
                 with_norm=False,
                 loss_weight=1.0):
        super(MMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = int(kernel_num)
        self.fix_sigma = fix_sigma
        self.with_norm = with_norm
        self.loss_weight = loss_weight
        self.eps = 1e-16

    def _pairwise_distance(self, samples):
        """calculate distance matrix """
        if self.with_norm:
            samples = F.normalize(samples, dim=1)
        dist = torch.matmul(samples, samples.t())
        square_norm = torch.sum(samples.pow(2), dim=1)
        dist = square_norm.unsqueeze(1) - 2 * dist + square_norm.unsqueeze(0)

        return dist

    def _get_guassian_kernel(self, source, target):
        """get guassian kernel"""
        samples = torch.cat((source, target), dim=0)
        size = samples.shape[0]
        dist = self._pairwise_distance(samples)
        bandwidth = self.fix_sigma if self.fix_sigma else dist.sum() / (size ** 2 - size)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidths = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel = sum([torch.exp(-dist / bw) for bw in bandwidths])
        return kernel

    def forward(self, source, target):
        """
        Args:
             source (Variance): shape of (n, vector_size).
             target (Variance): shape of (m, vector_size).
        """
        n, m = source.shape[0], target.shape[0]
        kernel = self._get_guassian_kernel(source, target)
        loss = kernel[:n, :n].sum() / (n * n) + kernel[n:, n:].sum() / (m * m) - 2 * kernel[:n, n:].sum() / (n * m)
        return loss

