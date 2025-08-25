# -*- coding: UTF-8 -*-

"""
本文件实现了多种形式的triplet loss,包括一般形式的 TripletLoss 和 不对称 AsymTripletLoss。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet Loss.

    This is an implementation of paper `In Defense of the Triplet Loss
    for Person Re-Identification <https://arxiv.org/abs/1703.07737>`.

    Args:
        type (str, optional): To select `Batch All` or `Batch Hard`, Default BatchAll.
        margin (float, optional): The margin of triplet loss.
        with_norm (bool, optional): Whether to normalize the predict vector
            before calculate loss.
        loss_weight (float, optional): The weight of loss.

    """

    def __init__(self,
                 type='BatchAll',
                 margin=0.5,
                 with_norm=True,
                 loss_weight=1.0):
        super(TripletLoss, self).__init__()
        if type not in ['BatchAll', 'BatchHard']:
            raise TypeError
        self.type = type
        self.margin = margin
        self.with_norm = with_norm
        self.loss_weight = loss_weight
        self.eps = 1e-16

    def _pairwise_distance(self, pred):
        """calculate distance matrix """
        if self.with_norm:
            pred = F.normalize(pred, dim=1)
        dist = torch.matmul(pred, pred.t())
        square_norm = torch.sum(pred.pow(2), dim=1)
        dist = square_norm.unsqueeze(1) - 2 * dist + square_norm.unsqueeze(0)
        dist = torch.sqrt(F.relu(dist))

        return dist

    def _get_all_triplet_mask(self, target):
        """get triplet mask"""
        ind_not_equal = torch.logical_not(torch.eye(target.shape[0], dtype=torch.bool, device=target.device))
        dist_inds = ind_not_equal.unsqueeze(2) & ind_not_equal.unsqueeze(1) & ind_not_equal.unsqueeze(0)
        label_equal = torch.eq(target.unsqueeze(1), target.unsqueeze(0))
        ap_mask = label_equal.unsqueeze(2)
        an_mask = torch.logical_not(label_equal).unsqueeze(1)
        mask = ap_mask & an_mask & dist_inds
        return mask

    def _get_hard_triplet_mask(self, target):
        """gert hard triplet mask"""
        ind_not_equal = torch.logical_not(torch.eye(target.shape[0], dtype=torch.bool, device=target.device))
        label_equal = torch.eq(target.unsqueeze(1), target.unsqueeze(0))
        ap_mask = label_equal & ind_not_equal
        an_mask = torch.logical_not(label_equal)
        return ap_mask, an_mask

    def forward(self, pred, target):
        """
        Args:
             pred (Variance): shape of (batch_size, vector_size).
             target (Variance): shape of (batch-size,1)
        """
        dist = self._pairwise_distance(pred)

        if self.type == 'BatchHard':
            ap_mask, an_mask = self._get_hard_triplet_mask(target[:, 0])
            ap_dist = torch.max(dist * ap_mask, dim=1, keepdim=True)

            max_an_dist = torch.max(dist, dim=1, keepdim=True)
            max_an_dist = max_an_dist * torch.logical_not(an_mask)
            an_dist = torch.min(dist + max_an_dist, dim=1, keepdim=True)

            loss = torch.mean(F.relu(ap_dist - an_dist + self.margin))
        else:
            mask = self._get_all_triplet_mask(target[:, 0])
            ap_dist = dist.unsqueeze(2)
            an_dist = dist.unsqueeze(1)

            loss = ap_dist - an_dist + self.margin
            loss = F.relu(loss * mask)
            num_triplet = torch.sum(loss > 0).type(torch.float32)

            loss = loss.sum() / (num_triplet + self.eps)
        return loss


class AsymTripletLoss(TripletLoss):
    """Asymmetric Triplet Loss.
    Anchor is only assigned to positive samples(target==0),and the ideal constraint
    effect is as follows:
    *************************************
         3   1  2 1   2
        1 3         2 1 3
       1  1   0 0      3
      2 3    0 0 0     1  3
        1     0 0     1  2 1
       2  3           1
           1 2    2 3
    *************************************
    Args:
        type (str, optional): To select `Batch All` or `Batch Hard`, Default BatchAll.
        margin (float, optional): The margin of triplet loss.
        with_norm (bool, optional): Whether to normalize the predict vector
            before calculate loss.
        loss_weight (float, optional): The weight of loss.

    """

    def __init__(self,
                 type='BatchAll',
                 margin=0.5,
                 with_norm=True,
                 loss_weight=1.0):
        super(AsymTripletLoss, self).__init__(type, margin, with_norm, loss_weight)

    def _get_all_triplet_mask(self, target):
        """get triplet mask"""
        ind_not_equal = torch.logical_not(torch.eye(target.shape[0], dtype=torch.bool, device=target.device))
        dist_inds = ind_not_equal.unsqueeze(2) & ind_not_equal.unsqueeze(1) & ind_not_equal.unsqueeze(0)
        add_matrix = target.unsqueeze(1) + target.unsqueeze(0)

        ap_mask = torch.eq(add_matrix, 0)
        an_mask = torch.logical_not(torch.eq(target.unsqueeze(1), target.unsqueeze(0)))
        mask = ap_mask.unsqueeze(2) & an_mask.unsqueeze(1) & dist_inds
        return mask

    def _get_hard_triplet_mask(self, target):
        """get hard triplet mask"""
        raise NotImplementedError
