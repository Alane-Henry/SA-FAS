# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现可视化工具.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold


class VisualizeTSNE(object):
    """Visualize TSNE for features.

    Args:
        work_dir (str): work dir to log file or result file.
        tsne_cfg (dict): the config dict of tsne.
        figsize (tuple): figure size to draw.
    """
    def __init__(self,
                 work_dir,
                 filename,
                 tsne_cfg=None,
                 figsize=(10, 10)):
        self.eval_cfg = tsne_cfg
        self.title = os.path.basename(work_dir)
        self.work_dir = os.path.join(work_dir, 'results')
        os.makedirs(self.work_dir, exist_ok=True)

        self.marks = tsne_cfg.pop('marks', None)
        self.maxsamples = tsne_cfg.pop('maxsamples', 10000)
        self.filename = os.path.basename(filename).split('.')[0] + tsne_cfg.pop('filename', '_tsne.png')
        self.tsne = manifold.TSNE(init='pca', n_components=2, random_state=501)
        self.figsize = figsize
        self.colors = ['r', 'k', 'g', 'b', 'y', 'c', 'm', 'orange']
        self.c = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        self.m = ['.', 'x', '1', '2', '3', '4', '+']


    def __call__(self, feats, labels, paths=None, filename=None, title=None):
        """call function.
        Args:
            feats (np.array()): the features shape of (N, d).
            labels (np.array()): the labels shape of (N, ).
            paths (np.array()): the paths shape of (N, ).
        """
        groups = np.bincount(labels.astype(np.uint8))
        if self.marks is not None:
            assert len(groups) == len(self.marks)
            
        np.savez(os.path.join(self.work_dir, os.path.splitext(self.filename)[0]+'.npz'), feats=feats, labels=labels, paths=paths if paths is not None else None)

        if len(feats) > self.maxsamples:
            indices = np.random.choice(feats.shape[0], self.maxsamples, replace=False)
            feats = feats[indices, :]
            labels = labels[indices]
            paths = paths[indices] if paths is not None else None
        # if paths is None:
        #     domains = np.array(['']*len(labels))
        # else:
        #     domains = []
        #     for p in paths:
        #         if 'pos/' in p or 'neg/' in p or 'pass_attack_labeled' in p:
        #             domains.append('train')
        #         else:
        #             domains.append(p.strip('/').split('/')[0])
        #     domains = np.array(domains)
        domains = np.array(['']*len(labels) if paths is None else [p.strip('OpenSource').strip('/').split('/')[0] for p in paths])
        positions = self.tsne.fit_transform(feats)
        positions = (positions - positions.min(0)) / (positions.max(0) - positions.min(0))
        domainset = sorted(set(domains), reverse=True)
        plt.figure(figsize=self.figsize)
        plt.switch_backend('agg')
        for i, g in enumerate(groups):
            for j, d in enumerate(domainset):
                if g == 0:
                    continue
                posi = positions[(labels == i)&(domains == d), :]
                if self.marks is None:
                    np.random.seed(j)
                    plt.scatter(posi[:, 0], posi[:, 1], s=5, 
                        c=self.colors[j] if j < len(self.colors) else '#'+''.join(np.random.choice(self.c, 6)),
                        marker=self.m[i], linewidths=0.2, 
                        alpha=0.75, label=f'label={d} {i}')
                else:
                    plt.scatter(posi[:, 0], posi[:, 1], **self.marks[i])
        plt.legend(prop=dict(size=4))
        plt.tick_params(labelsize=3)
        title = title if title else self.title
        plt.title(title)
        filename = os.path.join(self.work_dir, filename) if filename else self.filename
        plt.savefig(os.path.join(self.work_dir, filename), dpi=1000, bbox_inches='tight')


class VisualizeLog(object):
    """Visualize train logs.

    Args:
        work_dir (str): work dir to log file or result file.
        figsize (tuple): figure size to draw.
        eval_types (str, list): eval types to plot.
        loss_types (list, optional): loss types to plot.

    """
    def __init__(self,
                 work_dir,
                 plog_cfg=None,
                 figsize=(15, 10)):
        self.figsize = figsize
        self.plog_cfg = plog_cfg.copy()
        self.loss_types = self.plog_cfg .pop('loss_types', None)
        self.eval_types = self.plog_cfg .pop('eval_types', 'auc')
        if not isinstance(self.eval_types, list):
            self.eval_types = [self.eval_types]
        self.title = os.path.basename(work_dir)
        self.work_dir = os.path.join(work_dir, 'results')
        os.makedirs(self.work_dir, exist_ok=True)

        self.colors = ['r', 'g', 'b', 'y', 'c', 'm']
        self.eval_names = ['auc', 'acc', 'acer', 'apcer', 'bpcer',
                           'trr@1e-4', 'trr@1e-3', 'trr@1e-2', 'trr@1e-1']

    def _parse_log(self, log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        iter = 0
        evals = list()
        val_losses = list()
        loss_thr = list()
        losses = list()
        loss_names = []
        for line in lines:
            if 'INFO' in line and 'loss' in line:
                infos = re.split(': |, ', re.split('min, ', line)[1])
                if iter == 0:
                    loss_names = infos[0::2]

                iter = int(re.findall(r"(?<=Iter: )\d+", line)[0])
                loss = list(map(float, infos[1::2]))
                loss.append(iter)
                losses.append(loss)
            elif 'auc' in line and 'trr' in line:
                self.eval_names = re.findall(r"\S+", line)[2:-2]
            elif '* >>' in line and 'acc' not in line and 'thr' not in line:
                res = list(map(float, re.findall(r"\d+\.?\d*", line)))
                res.append(iter)
                evals.append(res)
            elif 'Val Loss:' in line:
                res = list(map(float, re.findall(r"\d+\.?\d*", line.split('INFO')[-1])))
                res.append(iter)
                val_losses.append(res)
            elif 'swad_thr' in line:
                res = list(map(float, re.findall(r"[-+]?\d+\.?\d*", line.split('INFO')[-1])))
                loss_thr = [[res[-1], res[0]], [res[-1], res[1]]]

        return evals, val_losses, loss_thr, losses, loss_names

    def _movingaverage(self, interval, window_size=10, mode='valid'):
        window= np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, mode)

    def __call__(self, log_file, start=0.0):
        """call function

        Args:
            log_file (str): log file dir.
            start (float): initial proportion
        """
        name = os.path.basename(log_file).split('.')[0]

        evals, val_losses, loss_thr, losses, loss_names = self._parse_log(log_file)

        evals = np.array(evals)
        val_losses = np.array(val_losses)
        loss_thr = np.array(loss_thr)
        losses = np.array(losses)
        if len(evals) == 0 or evals.shape[1] != 10:
            return

        plt.figure(num=0, figsize=self.figsize)
        plt.switch_backend('agg')
        for i, k in enumerate(self.eval_names):
            if k not in self.eval_types:
                continue
            plt.plot(evals[:, -1], evals[:, i], self.colors[i % 6], label=k)
        plt.grid(ls='--')
        plt.legend(loc='lower right')
        plt.ylabel('Eval Score')
        plt.xlabel('Iters')
        plt.title(self.title)
        plt.savefig(os.path.join(self.work_dir, f'{name}_eval.png'))

        if self.loss_types is None:
            return
        plt.figure(num=1, figsize=self.figsize)
        plt.switch_backend('agg')
        if len(val_losses):
            plt.plot(val_losses[int(start*len(val_losses)):, -1], 
                    val_losses[int(start*len(val_losses)):, 0], 'k', label='val_loss')
        if len(loss_thr):
            plt.plot(loss_thr[:, -1], loss_thr[:, 0], 'm', label='swad_thr', linestyle='--')
        for i, k in enumerate(loss_names):
            if self.loss_types == 'all' or k in self.loss_types:
                plt.plot(self._movingaverage(losses[int(start*len(losses)):, -1]), 
                        self._movingaverage(losses[int(start*len(losses)):, i]), self.colors[i % 6], label=k)
        plt.grid(ls='--')
        plt.legend(loc='upper right')
        plt.ylabel('Loss')
        plt.xlabel('Iters')
        plt.title(self.title)
        plt.savefig(os.path.join(self.work_dir, f'{name}_loss.png'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='visualizer')
    parser.add_argument('log', type=str, default='', help='logfile')
    parser.add_argument('--start', '-s', type=float, default=0.0, help='start')
    args = parser.parse_args()
    vis_log = VisualizeLog(os.path.dirname(args.log), dict(
        loss_types='all',
        eval_types=['trr@1e-2', 'trr@1e-3']))
    vis_log(args.log, start=args.start)