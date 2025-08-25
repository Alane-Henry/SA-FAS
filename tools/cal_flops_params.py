# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现了torch 模型占用资源、计算量衡量

"""

import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info
from thop import profile, clever_format

sys.path.append(".")
from apis import build_models
from utils import Config, seed_everywhere


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description='Cal flops, params, mem, time')
    parser.add_argument('config', help='path to train config file')
    parser.add_argument('--size', nargs='+', type=int, default=[16,3,224,224], help='input tensor size')
    parser.add_argument('--use_cpu', action='store_true', help="whether use cpu")
    parser.add_argument('--loop_num', type=int, default=10, help="loop test time")
    args = parser.parse_args()

    return args


def get_logger(name, log_file, log_level):
    logger = logging.getLogger(name)

    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    file_handler = logging.FileHandler(log_file, 'w')
    handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)

    return logger


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total


def main():
    """main"""
    args = parse_args()

    dst_path = os.path.splitext(__file__)[0]
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_logger("Params & FLOPs", os.path.join(dst_path,
                        f'{timestamp}.log'), log_level='INFO')

    logger.info(f"Load {args.config}")
    cfg = Config.fromfile(args.config)
    # set random seed
    if cfg.get('seed'):
        seed_everywhere(cfg.seed)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.test_cfg.return_label = False
    cfg.model.test_cfg.return_feature = False

    if args.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    repetitions = args.loop_num

    model = build_models(cfg.model).to(device)
    model.eval()

    dummy_input = torch.randn(args.size, dtype=torch.float).to(device)
    print(dummy_input.shape)
    
    # fv
    macs = FlopCountAnalysis(model, (dummy_input,)).total() / 1e9
    params = print_model_parm_nums(model)
    logger.info(f"fv {macs:.3f}G FLOPS, sum {params}")
    # pt
    flops, params = get_model_complexity_info(model, tuple(args.size[1:]), as_strings=True, print_per_layer_stat=False,)
    logger.info(f"pt {flops}, {params}")
    # thop
    macs, params = profile(model, inputs=(dummy_input, ),verbose=True)
    macs, params = clever_format([macs, params], "%.3f")
    logger.info(f"thop {macs} FLOPS, {params}")

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    timings=np.zeros((repetitions,1))

    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()

            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    std_syn = np.std(timings)
    mean_syn = np.sum(timings) / repetitions
    mean_fps = 1000. / mean_syn
    logger.info(f'Mean@{args.size[0]} {mean_syn:.3f}ms Std@{args.size[0]} {std_syn:.3f}ms FPS@{args.size[0]} {mean_fps:.2f}')

if __name__ == "__main__":
    main()