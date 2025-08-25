# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现高层 api 创建器.
"""

import math
import torch
import warnings
import torch.optim as opt
import torch.optim.lr_scheduler as lrs
import models.fas as fas
import datasets as ds
from torch.utils.data import DataLoader, RandomSampler
from .sampler import BalanceSampler, DistBalanceSampler, SwitchSampler
import numpy as np 
import random

def build_models(cfg, **kwargs):
    """build model from config dict.

    Args:
        cfg (dict): the config dict of model.
    """

    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    cfg_ = cfg.copy()
    model_class = cfg_.pop('Class', 'fas')
    model_type = cfg_.pop('type')
    model = eval(f'{model_class}.{model_type}')(**cfg_, **kwargs)
    return model


def build_datasets(cfg, **kwargs):
    """build dataset from config dict.

    Args:
        cfg (dict): the config dict of dataset.
    """

    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    cfg_ = cfg.copy()
    dataset_type = cfg_.pop('type')
    dataset = eval(f'ds.{dataset_type}')(**cfg_, **kwargs)
    return dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    # worker_seed = worker_seed % 42 + worker_id
    # print(f'worker_id: {worker_id}, seed: {worker_seed}')
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloaders(cfg, dataset, num_replicas=None, rank=None, **kwargs):
    """build dataloader from config dict.

    Args:
        cfg (dict): the config dict of dataloader.
        dataset (torch.utils.data.Dataset): the dataset to load.
        num_replicas (int, optional): number of replicas for distributed.
        rank (int, optional): current rank.
    """

    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    cfg_ = cfg.copy()
    sampler = None
    distributed = num_replicas is not None and rank is not None
    shuffle = cfg_.pop('shuffle', False)
    if shuffle:
        if dataset.test_mode:
            warnings.warn('Using BalanceSampler when test mode is True.')
        if distributed:
            sampler = DistBalanceSampler(dataset, num_replicas, rank)
            cfg_.pop('sampler', 'BalanceSampler')
        else:
            sampler_type = cfg_.pop('sampler', 'BalanceSampler')
            if sampler_type == 'RandomSampler' or sampler_type is None:
                sampler = None
            else:
                sampler = eval(sampler_type)(dataset, cfg['samples_per_gpu'])
    num_gpus = cfg_.pop('num_gpus')
    if distributed:
        batch_size = cfg_.pop('samples_per_gpu')
    else:
        batch_size = cfg_.pop('samples_per_gpu') * num_gpus
    num_workers = cfg_.pop('workers_per_gpu')
    g = torch.Generator()
    generate_seed = cfg_.pop('seed', 42)
    g.manual_seed(6148914691236517205)  # from yolov5
    # print(f'generator seed: {generate_seed}')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        collate_fn=getattr(dataset, 'collate_fn', None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last= not dataset.test_mode,
        worker_init_fn=seed_worker,
        generator=g,
        **cfg_,
        **kwargs)
    return dataloader


def build_optimizers(cfg, model, **kwargs):
    """build optimizer from config dict.

    Args:
        cfg (dict): the config dict of optimizer.
        model (nn.Module): the model to optimize.
    """

    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    cfg_ = cfg.copy()
    optimizer_type = cfg_.pop('type')
    diff_lr_layer = cfg_.pop('diff_lr_layer', None)
    lr = cfg_.get('lr', 1e-3)
    if diff_lr_layer:
        parameters = [{'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in diff_lr_layer.keys())], 'lr':lr}]
        for k,v in diff_lr_layer.items():
            key = [n for n,p in model.named_parameters() if k in n]
            pdict = {'params':[p for n,p in model.named_parameters() if k in n], 'lr':lr*v}
            print('lr: {} => layers: {}'.format(lr*v, key))
            parameters.append(pdict)     
    else:
        parameters = model.parameters()
    optimizer = eval(f'opt.{optimizer_type}')(parameters, **cfg_, **kwargs)
    return optimizer


def build_schedulers(cfg, optimizer, **kwargs):
    """build scheduler from config dict.

    Args:
        cfg (dict): the config dict of scheduler.
        optimizer (torch.optim) the torch optimizer.
    """

    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    cfg_ = cfg.copy()
    scheduler_type = cfg_.pop('type')
    if scheduler_type == 'CosineLR':
        T = cfg_.pop('total_epochs', 50)
        gamma = cfg_.pop('gamma', 0.1)
        lambda1 = lambda epoch: gamma if 0.5 * (1 + math.cos(math.pi * epoch / T)) < gamma else 0.5 * (
                1 + math.cos(math.pi * epoch / T))
        scheduler = lrs.LambdaLR(optimizer, lr_lambda=lambda1, **kwargs)
    elif scheduler_type == 'ExpLR':
        T = cfg_.pop('total_epochs', 30)
        gamma = cfg_.pop('gamma', 0.6)
        steps = cfg_.pop('steps', 5)
        lambda1 = lambda epoch: gamma ** (steps * epoch/T)
        scheduler = lrs.LambdaLR(optimizer, lr_lambda=lambda1, **kwargs)
    else:
        scheduler = eval(f'lrs.{scheduler_type}')(optimizer, **cfg_, **kwargs)
    return scheduler
