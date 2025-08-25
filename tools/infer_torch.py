# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现了torch inference

"""

import os
import cv2
import sys
import math
import time
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

sys.path.append(".")
from apis import build_models
from utils.cv_util import distance_pt
from utils import Config, seed_everywhere


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description='Infer torch')
    parser.add_argument('config', help='path to train config file')
    parser.add_argument('--load_from', help='the checkpoint file to load from')
    parser.add_argument('--img_prefix', default="/root/work/data/liveness", help='the ann file to inference')
    parser.add_argument('--ann', default="data/test/sub_4w_FAS_test_bdv1.txt", help='the ann file to inference')
    parser.add_argument('--crop_method', default="4_lmks_of_72", help="crop method")
    parser.add_argument('--enlarge', default=3.0, help="crop enlarge ratio")
    parser.add_argument('--img_size', default=(336,336), help="crop img size")
    args = parser.parse_args()

    return args

def crop_warp(img, lamk, crop_method, enlarge=3.0, img_size=(224, 224)):
    "warp crop face"
    # coordinates of both eyes
    if crop_method == "4_lmks_of_72":
        lamk_inds =  [[21,38], [21, 38, 58, 62]]
    elif crop_method == "2eyes_lmks_of_72":
        lamk_inds = [[21,38], [21,38]]
    elif lamk_inds == "all_lmks_of_72":
        lamk_inds = [[21,38], []]
    else:
        raise ValueError("crop method no implement!")

     # set left, right eyes index
    idx1, idx2 = lamk_inds[0][0], lamk_inds[0][1]

    angle = 0
    if (lamk[idx1] > 0).all() and (lamk[idx2] > 0).all():
        angle += np.arctan2(lamk[idx2, 1] - lamk[idx1, 1], lamk[idx2, 0] - lamk[idx1, 0]) * 180.0 / np.pi

    if len(lamk_inds[1]) == 4:
        ct = np.mean(lamk[lamk_inds[1], :], axis=0)

        # lamk_inds should be lt, rtm ld, rd
        _w = 0.5*distance_pt(lamk[lamk_inds[1][0]], lamk[lamk_inds[1][1]]) + \
                0.5*distance_pt(lamk[lamk_inds[1][2]], lamk[lamk_inds[1][3]])
        _h = 0.5*distance_pt(lamk[lamk_inds[1][0]], lamk[lamk_inds[1][2]]) + \
                0.5*distance_pt(lamk[lamk_inds[1][1]], lamk[lamk_inds[1][3]])
        wh = np.sqrt(_w*_h)

        if wh == 0:
            print("wh not be zero!")
            wh = 10

    elif len(lamk_inds[1]) == 2:
        ct = np.mean(lamk[lamk_inds[1], :], axis=0)
        wh = distance_pt(lamk[lamk_inds[1][0]], lamk[lamk_inds[1][1]])

    else:
        rd = np.max(lamk, axis=0)
        lamk[lamk == -1] = np.inf
        lt = np.min(lamk, axis=0)
        ct, wh = (rd + lt) / 2, (rd[0] - lt[0])

        rd = np.max(lamk, axis=0)
        lamk[lamk == -1] = np.inf
        lt = np.min(lamk, axis=0)

        ct, wh = (rd + lt) / 2, (rd[0] - lt[0])

    M = cv2.getRotationMatrix2D((ct[0], ct[1]), angle, img_size[0] / (wh * enlarge))
    M[0, 2] = M[0, 2] - (ct[0] - img_size[0] / 2.0)
    M[1, 2] = M[1, 2] - (ct[1] - img_size[0] / 2.0)

    # draw lamks
    # for pt in lamk[lamk_inds[1]]:
    #     img = cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0,0,255), -1)

    img = cv2.warpAffine(img, M, img_size)
    cv2.imwrite("warp.jpg", img)
    return img

def color_normalize(img, mean=[0,0,0], std=[1,1,1]):
    """normalize image"""
    img = img.astype(np.float32)
    mean, std = np.array(mean), np.array(std)
    cv2.subtract(img, np.float64(mean.reshape(1, -1)), img)
    cv2.multiply(img, 1 / np.float64(std.reshape(1, -1)), img)
    img = img.transpose(2, 0, 1)
    return img


def preprocess(img, pts, crop_method, enlarge, img_size):
    """preprocess input image"""
    img = crop_warp(img, pts, crop_method, enlarge, img_size)
    # img = cv2.resize(img, img_size)
    img = color_normalize(img, [127.5,127.5,127.5],[255,255,255])
    img = img[np.newaxis, :, :, :]
    return img

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


def main():
    """main"""
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dst_path = os.path.splitext(__file__)[0]
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_logger("Inference torch", os.path.join(dst_path, 
                f'{timestamp}.log'), log_level='INFO')
    logger.info(f"Load {args.config}")
    logger.info(f"Model {args.load_from}")
    logger.info(f"infer {args.ann}")

    # set random seed
    if cfg.get('seed'):
        seed_everywhere(cfg.seed)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.test_cfg.return_label = False
    cfg.model.test_cfg.return_feature = False

    model = build_models(cfg.model)

    state_dict = dict()
    checkpoint = torch.load(args.load_from, map_location='cpu')
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[7:]] = v
    model.eval()
    model.load_state_dict(state_dict, strict=True)
    model.to("cuda")

    scores, labels = [], []
    fout = open(os.path.join(dst_path, f"{timestamp}_score.txt"),'w')
    with open(args.ann, 'r') as f:
        lines = f.readlines()
       
        for line in tqdm(lines):
            data = line.strip().split()
            img = cv2.imread(os.path.join(args.img_prefix, data[0]))
            # pts = np.array([float(d) for d in data[1:145]]).reshape(72, 2)
            pts = np.zeros((72, 2))
            try:
                img = preprocess(img, pts, args.crop_method, args.enlarge, args.img_size)
            except:
                continue
            
            img = torch.from_numpy(img).to("cuda")
            results = model(img)
            print(results)

            score = 1-results[0]
            scores.append(score) # 0 index -- liveness score
            labels.append(int(data[-1]))
            fout.write('{} {}\n'.format(score, int(data[-1])))

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    ind4e = abs(np.array(fpr)-0.0001).argmin()
    ind3e = abs(np.array(fpr)-0.001).argmin()
    ind2e = abs(np.array(fpr)-0.01).argmin()

    Trr4e, threshold4e = tpr[ind4e], thresholds[ind4e]
    Trr3e, threshold3e = tpr[ind3e], thresholds[ind3e]
    Trr2e, threshold2e = tpr[ind2e], thresholds[ind2e]

    logger.info(f"AUC={roc_auc}")
    logger.info(f"TRR={Trr2e}@FRR=1e-2, threshold={threshold2e}")
    logger.info(f"TRR={Trr3e}@FRR=1e-3, threshold={threshold3e}")
    logger.info(f"TRR={Trr4e}@FRR=1e-4, threshold={threshold4e}")


if __name__ == '__main__':
    main()
