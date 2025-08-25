# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现 dataloader sampler.
"""

import numpy as np
from torch.utils.data import Sampler, DistributedSampler


class BalanceSampler(Sampler):
    """Samplers balanced by category

    Args:
        dataset (torch.utils.data.Dataset): dataset to sample.
        samples_per_gpu (int): number sample per gpu.

    """
    def __init__(self,
                 dataset,
                 samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_group = (self.group_sizes > 0).sum()
        assert samples_per_gpu % self.num_group == 0
        self.max_size = int(
            self.group_sizes.max() / self.samples_per_gpu + 1) * self.samples_per_gpu
        self.num_samples = self.max_size * self.num_group

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            np.random.shuffle(indice)
            assert len(indice) == size
            indice = np.tile(indice, int(self.max_size / len(indice) + 1))[:self.max_size]
            assert len(indice) == self.max_size
            indices.append(indice)
        indices = np.array(indices).transpose()
        np.random.shuffle(indices)
        indices = indices.flatten()
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(range(len(indices) // self.samples_per_gpu))]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistBalanceSampler(DistributedSampler):
    """Distributed Samplers balanced by category

    Args:
        dataset (torch.utils.data.Dataset): dataset to sample.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        samples_per_gpu (int): number sample per gpu.

    """
    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None):
        super(DistBalanceSampler, self).__init__(dataset, num_replicas, rank)
        assert hasattr(dataset, 'flag')
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_group = (self.group_sizes > 0).sum()
        self.max_size = int(
            self.group_sizes.max() / self.num_replicas + 1) * self.num_replicas
        self.total_size = self.max_size * self.num_group
        self.num_samples = int(self.total_size / self.num_replicas)

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            indice = np.tile(indice, int(self.max_size / len(indice) + 1))[:self.max_size]
            assert len(indice) == self.max_size
            indices.append(indice)
        indices = np.array(indices).transpose()
        np.random.shuffle(indices)
        indices = indices.flatten().astype(np.int64).tolist()
        assert len(indices) == self.total_size

        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


class SwitchSampler(Sampler):
    """Switch flag per batch.

        Args:
            dataset (torch.utils.data.Dataset): dataset to sample.
            flag_per_batch (int): number flag per batch.
            samples_per_gpu (int): number sample per gpu.

        """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 flag_per_batch=2):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.flag_per_batch = flag_per_batch
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_group = (self.group_sizes > 0).sum()
        assert samples_per_gpu % self.flag_per_batch == 0
        self.max_size = int(
            self.group_sizes.max() / self.samples_per_gpu + 1) * self.samples_per_gpu
        self.num_samples = self.max_size * self.num_group

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            indice = np.tile(indice, int(self.max_size / len(indice) + 1))[:self.max_size]
            assert len(indice) == self.max_size
            indices.append(indice)
        indices = np.array(indices).transpose()
        np.random.shuffle(indices)
        indices_list = []
        num = self.samples_per_gpu // self.flag_per_batch
        for i in range(self.max_size // num):
            for j in range(self.num_group // self.flag_per_batch):
                batch = indices[i * num: (i + 1) * num, j * self.flag_per_batch: (j + 1) * self.flag_per_batch]
                indices_list.append(batch.flatten())
        indices = np.concatenate(indices_list)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class HardSampler(Sampler):
    """Samplers balanced by hard sample score.

    Args:
        dataset (torch.utils.data.Dataset): dataset to sample.
        samples_per_gpu (int): number sample per gpu.

    """
    def __init__(self,
                 dataset,
                 samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu

    def __iter__(self):
        # TODO
        pass

    def __len__(self):
        pass
