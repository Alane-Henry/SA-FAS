# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现模型评估器.
"""
import sys
sys.path.append("..")
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils.fileio import load, dump


class Metric(object):
    """Metric tools for binary classification results.

    Args:
        logger (logger): the logger for train/test.
        work_dir (str): work dir to log file or result file.
        eval_cfg (dict): the config dict of evaluation.
    """

    def __init__(self,
                 logger,
                 work_dir=None,
                 eval_cfg=None):
        self.logger = logger
        self.eval_cfg = eval_cfg
        self.work_dir = os.path.join(work_dir, 'results')
        self.score_type = eval_cfg.pop('score_type', 'acc')
        self.trr_list = eval_cfg.pop('trr', [1e-4,1e-3,1e-2,1e-1]) # list of trr metrics (multiplied by 1e-4)
        assert len(self.trr_list) == 4
        os.makedirs(self.work_dir, exist_ok=True)

        self._apcer = 1.0
        self._bpcer = 1.0
        self._acer = 1.0
        self._acc = 0.0
        self._auc = 0.0
        self._trr = [0.0, 0.0, 0.0, 0.0]

    def _eval_acer(self, preds, labels, thr):
        """calculate acer"""
        preds = preds >= thr

        tp = ((labels == 0) & preds).sum()
        fp = ((labels != 0) & preds).sum()
        fn = ((labels == 0) & ~preds).sum()
        tn = ((labels != 0) & ~preds).sum()

        apcer = 1.0 if fp + tn == 0 else fp / float(fp + tn)
        bpcer = 1.0 if fn + tp == 0 else fn / float(fn + tp)
        acer = (apcer + bpcer) / 2.0

        return [acer, apcer, bpcer]

    def _eval_trrs(self, preds, labels):
        """calculate trrs"""
        trrs = list()
        frrs = list()
        thrs = list()
        pos_scores = np.sort(preds[labels == 0], axis=0)
        neg_scores = np.sort(preds[labels != 0], axis=0)

        for frr in np.arange(0, 1.0, 1e-4):
            thr = pos_scores[int(len(pos_scores) * frr)]
            trr = np.sum(neg_scores < thr) * 1.0 / len(neg_scores)
            frrs.append(frr)
            trrs.append(trr)
            thrs.append(thr)

        return [frrs, trrs, thrs]

    def _plot_roc(self, frrs, trrs, thrs, prefix=None):
        """plot roc cures"""
        roc_name = "roc.png"
        if prefix is not None:
            roc_name = f"{prefix}_roc.png" 
        plt.title('ROC')
        plt.switch_backend('agg')
        plt.rcParams['figure.figsize'] = (6.0, 6.0)
        plt.plot(frrs, trrs, 'b', label='AUC = {:.4f}'.format(self._auc))
        plt.grid(ls='--')

        for i in range(4):
            ind = int(np.power(10, i)) - 1
            plt.annotate('(1e{},{:.4f})'.format(int(frrs[ind]), trrs[ind]),
                         xy=(frrs[ind], trrs[ind] - 0.01), c='r')
            plt.scatter(frrs[ind], trrs[ind], marker='x', c='r',
                        label='Thr@1e{}: {:.4f}'.format(int(frrs[ind]), thrs[ind]))
        plt.legend(loc='lower right')
        plt.ylabel('Ture Reject Rate')
        plt.xlabel('False Reject Rate (Log@10)')
        plt.savefig(os.path.join(self.work_dir, roc_name))

    def _get_thr(self, preds, labels):
        """get best thr"""
        thrs = list()
        dists = list()
        acers = list()
        apcers = list()
        bpcers = list()
        for thr in np.arange(1.00000, 0.0, -1e-4):
            acer, apcer, bpcer = self._eval_acer(preds, labels, thr)
            apcers.append(apcer)
            bpcers.append(bpcer)
            acers.append(acer)
            dists.append(np.abs(apcer - bpcer))
            thrs.append(thr)
        print(f'length of dists: {len(dists)}')
        min_dist_ind = int(np.argmin(np.array(dists)))
        min_acer_ind = int(np.argmin(np.array(acers)))

        print(f'thr: {thrs[min_dist_ind]:.6f}, acer: {acers[min_dist_ind]}, dist: {dists[min_dist_ind]:.6f}, acper: {apcers[min_dist_ind]:.6f}, bpcer: {bpcers[min_dist_ind]:.6f}')
        return thrs[min_acer_ind], thrs[min_dist_ind]

    def _log_infos(self, dataname=None):
        """log eval info"""
        info = '\n' + '*' * 99 + '\n'
        if dataname is not None:
            info += ('* >>' + ' {:^29s}'.format(dataname)) 
            info += ' '*60 +' << *\n'
        trrlist = [f'trr@{t:.0e}'.replace('e-0', 'e-') for t in self.trr_list]
        info += ('* >>' + ' {:^9s}' * 9 + ' << *\n').format(
            'auc', 'acc', 'acer', 'apcer', 'bpcer', *trrlist)
        info += ('* >>' + ' {:^9.5f}' * 9 + ' << *\n').format(
            self._auc, self._acc, self._acer, self._apcer, self._bpcer,
            self._trr[0], self._trr[1], self._trr[2], self._trr[3])
        info += '*'+ '-' * 97 + '*\n'
        info += ('* >> {:^9s}' + ' {:^9.6f}' * 8 + ' << *\n').format('thrs:',*self._thr)
        info += '*' * 99
        if self.logger is None:
            print(info)
        else:
            self.logger.info(info)

    @property
    def score(self):
        """return score, larger is better"""
        score_dict = dict(
            acc=self._acc,
            auc=self._auc,
            trr=self._trr[1],
            trre3=self._trr[1],
            trre2=self._trr[2],
            trre1=self._trr[3],
            trr1 = self._trr[0],
            trr2 = self._trr[1],
            trr3 = self._trr[2],
            trr4 = self._trr[3],
            acer=1 - self._acer,
            apcer=1 - self._apcer,
            bpcer=1 - self._bpcer)
        eval_dict = {
            'acc': self._acc,
            'auc': self._auc,
            'acer': self._acer,
            'apcer': self._apcer,
            'bpcer': self._bpcer}
        for i, t in enumerate(self.trr_list):
            eval_dict[f'trr@{t:.0e}'.replace("e-0","e-")] = self._trr[i]
        return score_dict[self.score_type], eval_dict
    
    def eval_and_infer(self, preds, labels, paths=None, thr=None, filename=None, dataname=None, log_info=True, val_num=5396):
        """metric function"""
        assert len(preds) == len(labels)
        # 2 classification
        # multi-classification not supported
        labels = np.where(labels>0, 0, 1)
        self._thr = []
        if thr is None or thr == []:
            thr1, thr = self._get_thr(preds[:val_num], labels[:val_num])

        if isinstance(thr, list):
            print('thr  acer    trr   tar')
            for t in thr:
                self._acer, self._apcer, self._bpcer = self._eval_acer(preds, labels, t)
                print(t, self._acer, f'{100*(1-self._apcer):.3f}%', f'{100*(1-self._bpcer):.3f}%')
            thr = thr[-1]
        else:
            self._acer, self._apcer, self._bpcer = self._eval_acer(preds[val_num:], labels[val_num:], thr)
        self._acc = ((preds < thr) == labels).sum() / len(preds)
        self._thr.extend([thr]*4)

        pos_num = (labels == 0).sum()
        if pos_num != len(labels) and pos_num != 0:
            self._auc = roc_auc_score((labels == 0).astype(labels.dtype), preds)

            frrs, trrs, thrs = self._eval_trrs(preds, labels)
            for i, t in enumerate(self.trr_list):
                self._trr[i] = trrs[int(t/1e-4)]
                self._thr.append(thrs[int(t/1e-4)])
            if filename is not None:
                self._plot_roc(np.log10(frrs[1:]), trrs[1:], thrs[1:], dataname)
        else:
            self._thr.extend([0,0,0,0])

        if filename is not None:
            result = np.hstack((preds[:, np.newaxis], labels[:, np.newaxis], paths[:, np.newaxis]))
            dump(result, os.path.join(self.work_dir, filename))

        if log_info:
            self._log_infos(dataname)

        return self.score

    def __call__(self, preds, labels, paths=None, thr=None, filename=None, dataname=None, log_info=True):
        """metric function"""
        assert len(preds) == len(labels)
        # 2 classification
        # multi-classification not supported
        labels = np.where(labels>0, 1, 0)
        self._thr = []
        if thr is None or thr == []:
            thr, thr2 = self._get_thr(preds, labels)

        if isinstance(thr, list):
            print('thr  acer    trr   tar')
            for t in thr:
                self._acer, self._apcer, self._bpcer = self._eval_acer(preds, labels, t)
                print(t, self._acer, f'{100*(1-self._apcer):.3f}%', f'{100*(1-self._bpcer):.3f}%')
            thr = thr[-1]
        else:
            self._acer, self._apcer, self._bpcer = self._eval_acer(preds, labels, thr)
        self._acc = ((preds < thr) == labels).sum() / len(preds)
        self._thr.extend([thr]*4)

        pos_num = (labels == 0).sum()
        if pos_num != len(labels) and pos_num != 0:
            self._auc = roc_auc_score((labels == 0).astype(labels.dtype), preds)

            frrs, trrs, thrs = self._eval_trrs(preds, labels)
            for i, t in enumerate(self.trr_list):
                self._trr[i] = trrs[int(t/1e-4)]
                self._thr.append(thrs[int(t/1e-4)])
            if filename is not None:
                self._plot_roc(np.log10(frrs[1:]), trrs[1:], thrs[1:], dataname)
        else:
            self._thr.extend([0,0,0,0])

        if filename is not None:
            result = np.hstack((preds[:, np.newaxis], labels[:, np.newaxis], paths[:, np.newaxis]))
            dump(result, os.path.join(self.work_dir, filename))

        if log_info:
            self._log_infos(dataname)

        return self.score


if __name__ == '__main__':
    import numpy as np
    import pandas

    import argparse
    parser = argparse.ArgumentParser(description='evaluator')
    parser.add_argument('result', type=str, default='', help='result')
    parser.add_argument('--thr', type=float,  nargs='+', default=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.9,0.99,0.999,0.9999,0.99999,0.999999], help='thr')
    # parser.add_argument('--thr', type=float,  nargs='+', default=[0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1], help='thr')
    parser.add_argument('--trrlist', type=float, nargs='+', default=[2e-3, 5e-3, 2e-2, 5e-2], help='trr')

    args = parser.parse_args()
    eval_cfg = dict(interval=500, score_type='trr', trr=args.trrlist)

    metric = Metric(None, 'output', eval_cfg)
    data = pandas.read_csv(args.result, sep=' ', header=None)
    print(data[1].value_counts())
    data = np.array(data)
    print(metric(data[:, 0], data[:, 1], thr=args.thr))