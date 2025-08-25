# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import time
import torch
import argparse
from torch.nn.parallel import DataParallel
from utils import Config, get_root_logger, seed_everywhere
from apis import Runner, build_models, build_datasets, build_dataloaders


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config', help='path to train config file')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--load_from', help='the checkpoint file to load from')
    parser.add_argument('--gpus', type=int, default=None, help='the number of gpus to use')
    args = parser.parse_args()

    return args


def main():
    """main"""
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # set random seed
    if cfg.get('seed'):
        seed_everywhere(cfg.seed)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    cfg.data.test.ann_files=['Protocol-val-test_fakelabel.txt'] 


    cfg.check_cfg.load_from = args.load_from

    
    if isinstance(args.gpus, int):
        cfg.gpu_ids = range(args.gpus)

    os.makedirs(os.path.expanduser(os.path.abspath(cfg.work_dir)), exist_ok=True)
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')
    cfg.log_cfg.filename = log_file

    model = build_models(cfg.model)
    model = DataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    dataset = build_datasets(cfg.data.test)
    dataloader = build_dataloaders(cfg.data.test_loader, dataset)

    try:
        logger.info(f'Test dataset: {dataset.groups}')
    except:
        pass

    runner = Runner(
        model,
        logger,
        work_dir=cfg.work_dir,
        log_cfg=cfg.log_cfg,
        eval_cfg=cfg.eval_cfg,
        check_cfg=cfg.check_cfg)


    runner.test(dataloader, filename=f'{timestamp}_result.txt', ann_nums=dataset.ann_nums)


if __name__ == '__main__':
    main()
