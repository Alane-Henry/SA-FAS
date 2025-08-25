import os
import re
import time
import torch
import numpy as np
from tqdm import tqdm
from visualdl import LogWriter
from apis.evaluator import Metric
from utils.dist import is_main_process
from utils.fileio import load
from utils.load_model import load_model
from apis.visualizer import VisualizeLog, VisualizeTSNE
from apis.builder import build_optimizers, build_schedulers, build_dataloaders
from models.losses import  ContrastiveLoss
import torch.nn as nn
from utils.fileio import load, dump



class Runner(object):
    """training or testing system.

    Args:
        model (nn.Module): the model to train/test.
        logger (logger): the logger for train/test.
        work_dir (str): work dir to save checkpoint or log file.
        eval_cfg (dict): the config dict of evaluation.
        optim_cfg (dict): the config dict of optimizer.
        sched_cfg (dict): the config dict of learning rate schedule.
        check_cfg (dict): the config dict of checkpoint.
    """
    def __init__(self,
                 model,
                 logger,
                 work_dir,
                 log_cfg=None,
                 eval_cfg=None,
                 optim_cfg=None,
                 sched_cfg=None,
                 check_cfg=None,
                 freeze_cfg=None):
        assert isinstance(work_dir, str)
        self.model = model
        self.logger = logger
        self.log_cfg = log_cfg
        self.eval_cfg = eval_cfg
        self.optim_cfg = optim_cfg
        self.sched_cfg = sched_cfg
        self.check_cfg = check_cfg
        self.freeze_cfg = freeze_cfg
        self.work_dir = os.path.abspath(work_dir)
        self._total_epoch = 0
        self._total_iter = 0
        self._iter_time = 0
        self._epoch = 0
        self._iter = 0
        self._eta = 0
        self._lr = 0

        os.makedirs(os.path.expanduser(self.work_dir), exist_ok=True)

        if self.freeze_cfg is not None:
            for k, v in self.model.named_parameters():
                for k_freeze in self.freeze_cfg:
                    if k.startswith("module."+k_freeze):
                        v.requires_grad = False
                        logger.info(f"{k} requires_grad be set False!")

        if self.optim_cfg is not None:
            self._warmup = self.sched_cfg.pop('warmup', 0)
            self._every_iter = self.sched_cfg.pop('every_iter', None)
            self.optimizer = build_optimizers(self.optim_cfg, self.model)
            self.scheduler = build_schedulers(self.sched_cfg, self.optimizer)
            if self.log_cfg.plog_cfg is not None and is_main_process():
                self.vis_log = VisualizeLog(self.work_dir, self.log_cfg.plog_cfg)
                self.writer_log = LogWriter(logdir=self.work_dir)
                hparm_cfg = self.log_cfg.plog_cfg.pop('hparm_cfg', None)
                if hparm_cfg is not None:
                    self.writer_log.add_hparams(
                        hparams_dict=dict(hparm_cfg),
                        metrics_list=self.log_cfg.plog_cfg.eval_types + self.log_cfg.plog_cfg.loss_types)

        self._score = np.zeros((self.check_cfg.pop('save_topk', 1),), dtype=np.float32)
        self._init_model(self.check_cfg.resume_from, self.check_cfg.load_from, 
                         self.check_cfg.pretrain_from)
        self.metric = Metric(logger, self.work_dir, eval_cfg)
        self.eval_thr = eval_cfg.pop('thr', None)
        if self.eval_cfg.tsne_cfg is not None:
            self.vis_tsne = VisualizeTSNE(self.work_dir, self.log_cfg.filename, self.eval_cfg.tsne_cfg.copy())

        self.dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self._train_swad = False

    def _lr_step(self):
        """update learning rate"""
        if self._warmup > self._iter:
            init_lrs = [v['initial_lr'] for v in self.optimizer.param_groups]
            for param_group, lr in zip(self.optimizer.param_groups, init_lrs):
                param_group['lr'] = lr / self._warmup * self._iter
        self._lr = self.optimizer.param_groups[0]['lr']

    def _log_infos(self, output):
        """log training info"""
        eta = (self._total_iter - self._iter) * self._iter_time / self.log_cfg.interval
        mins = '{:2d}'.format(int((eta % 3600) / 60))
        hours = '{:2d}'.format(int(eta / 3600))
        lr = '{:.3e}'.format(self._lr)
        self._iter_time = 0
        info = f'Epoch: {self._epoch}, Iter: {self._iter}, ETA: {hours}h{mins}min, Lr: {lr},'
        self.writer_log.add_scalar(tag='lr', step=self._iter, value=self._lr)
        for k, v in output.items():
            if self.log_cfg.plog_cfg is not None and k in self.log_cfg.plog_cfg.loss_types:
                self.writer_log.add_scalar(tag=k, step=self._iter, value=v.mean().detach().item())
            if k == 'loss':
                continue
            loss = '{:.5f}'.format(v.mean().detach().item())
            info += f' {k}: {loss},'
        info += ' loss: {:.5f}'.format(output['loss'].mean().detach().item())
        self.logger.info(info)

    def _init_model(self, resume_from=None, load_from=None, pretrain_from=None):
        """initialize model"""
        if load_from is not None:
            try:
                checkpoint = torch.load(load_from)
            except:
                checkpoint = torch.load(load_from, map_location='cpu')
            self._iter = checkpoint.get('iter', 0)
            self._epoch = checkpoint.get('epoch', 0)
            self.model = load_model(load_from, self.model, allow_size_mismatch=True)
            self.logger.info(f'Load from {load_from}, {self._epoch} epoch, {self._iter} iter')
        elif resume_from is not None:
            try:
                checkpoint = torch.load(resume_from)
            except:
                checkpoint = torch.load(resume_from, map_location='cpu')
            self._iter = checkpoint['iter']
            self._epoch = checkpoint['epoch']
            self._score = checkpoint['score']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info(f'Resume from {resume_from}, {self._epoch} epoch, {self._iter} iter')
        elif pretrain_from is not None:
            # try:
            #     checkpoint = torch.load(pretrain_from)
            # except:
            #     checkpoint = torch.load(pretrain_from, map_location='cpu')
            self.model = load_model(pretrain_from, self.model, allow_size_mismatch=True)
            self.logger.info(f'Pretrain from {pretrain_from}, {self._epoch} epoch, {self._iter} iter')

    def _save_model(self, score=None, filename=None):
        """save model"""
        if score is not None:
            if score < self._score[-1]:
                return
            for k in range(len(self._score)-1, 0, -1): #原先方法不会平移尾部模型
                if score >= self._score[k-1]:
                    filename = os.path.join(self.work_dir, f'top{k}_model.pth')
                    if os.path.exists(filename):
                        self._score[k] = self._score[k-1]
                        filename_next = os.path.join(self.work_dir, f'top{k+1}_model.pth')
                        os.system(f'mv {filename} {filename_next}')
                    self._score[k-1] = score
                else:
                    filename = os.path.join(self.work_dir, f'top{k+1}_model.pth')
                    break
                 
        elif filename is None:
            filename = os.path.join(self.work_dir, f'epoch{self._epoch}_iter{self._iter}.pth')
        else:
            filename = os.path.join(self.work_dir, filename)
        checkpoint = dict(
            iter=self._iter,
            epoch=self._epoch,
            score=self._score,
            state_dict=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict())
        self.logger.info(f'save model: {filename}')
        torch.save(checkpoint, filename)

    def _save_avg_model(self, val_loss, val_score):
        os.makedirs(os.path.join(self.work_dir, 'avg_model'), exist_ok=True)
        filename = os.path.join(self.work_dir, 'avg_model', f'{self._iter}_{val_loss}_{val_score}.pth')
        checkpoint = dict(
            iter=self._iter,
            epoch=self._epoch,
            val_loss=val_loss,
            val_score=val_score,
            state_dict=self.avgmodel.state_dict())
        torch.save(checkpoint, filename)
        self.logger.info(f'save avg model: {filename}')

    def _get_total_iter(self, dataloader, step_cfg=None):
        """get total iter"""
        if step_cfg is None:
            return self._total_epoch * len(dataloader)
        else:
            ir, hr, dr = step_cfg.init_rate, step_cfg.hem_rate, step_cfg.decay_rate
            train_len = int(ir * len(dataloader))
            test_len = len(dataloader) - train_len
            total_iter = 0
            if self._total_epoch < step_cfg.interval:
                self._total_iter = self._total_epoch * len(dataloader)
            for i in range(int(self._total_epoch / step_cfg.interval)):
                total_iter += train_len * step_cfg.interval
                train_len += test_len * hr
                test_len *= (1 - hr)
                hr *= dr
        self._total_iter = int(total_iter)
        return int(total_iter)

    def _train_epoch(self, dataloader):
        """train one epoch"""
        start_time = time.time()
        for i, data in enumerate(dataloader):
            # print(f"Epoch: {self._epoch}, Iter: {i+1}/{len(dataloader)}")
            self._iter += 1
            self.optimizer.zero_grad()

            data.pop('path', 'unknow')
            output = self.model(**data)
            output['loss'].mean().backward()

            self.optimizer.step()
            self._lr_step()

            self._iter_time += time.time() - start_time
            start_time = time.time()

            if self._train_swad:
                self.avgmodel.update_parameters(self.model, step=self._iter)

            if self._iter % self.log_cfg.interval == 0 and is_main_process():
                self._log_infos(output)

            if self._iter % self.eval_cfg.interval == 0 and is_main_process():

                val_score, val_loss = self.val_test(self.model)
                self._save_model(val_score)
                self._save_model(filename='latest.pth' if not self.eval_cfg.get('save_all_pth', False) else None)

                if self._train_swad:
                    self._save_avg_model(val_loss, val_score)
                    self.avgmodel = AveragedModel(self.model)  # reset

            if self._iter % self.check_cfg.interval == 0 and is_main_process():
                self._save_model()
            
            if self._every_iter:
                self.scheduler.step()

    @torch.no_grad()
    def val(self, model, showtitle='Val Loss'):
        """val method"""
        feats = list()
        preds = list()
        labels = list()
        paths = list()
        loss_sum = total = 0
        model.eval()
        for data in tqdm(self.val_dataloader):
            paths.extend(data.pop('path', 'unknow'))
            output = model(**data)
            preds.append(output['pred'].detach().cpu().numpy())
            labels.append(output['label'].detach().cpu().numpy()[:, 0])
            loss_sum += output['loss'].mean().item()*len(output['pred'])
            total += len(output['pred'])
            if output.get('feat') is not None:
                feats.append(output['feat'].detach().cpu().numpy())
        model.train()
        score, eval_dict = self.metric(np.concatenate(preds), np.concatenate(labels), paths=np.array(paths), thr=self.eval_thr)
        self.logger.info(f'{showtitle}: {loss_sum / total} ({total})')
        # if len(feats) > 0:
        #     feats = np.concatenate(feats)
        #     self.writer_log.add_embeddings(tag='feature', mat=feats, metadata=labels.astype(str))
        if self.log_cfg.plog_cfg is not None:
            for k, v in eval_dict.items():
                if k in self.log_cfg.plog_cfg.eval_types:
                    self.writer_log.add_scalar(tag=k, step=self._iter, value=v)
        return score, loss_sum / total

    @torch.no_grad()
    def test(self, dataloader, filename=None, ann_nums=None, log_info=True):
        """test method"""
        feats = list()
        preds = list()
        labels = list()
        paths = list()
        pred_cls = list()
        self.model.eval()
        for data in tqdm(dataloader):
            paths.extend(data.pop('path', 'unknow'))
            output = self.model(**data)
            preds.append(output['pred'].detach().cpu().numpy())
            labels.append(output['label'].detach().cpu().numpy()[:, 0])
            if output.get('feat') is not None:
                feats.append(output['feat'].detach().cpu().numpy())
            if output.get('pred_cls') is not None:
                pred_cls.append(output['pred_cls'].detach().cpu().numpy())
        self.model.train()
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        # pred_cls = np.concatenate(pred_cls)
        paths = np.array(paths)

        # self.metric(preds, labels, paths=paths, thr=self.eval_thr, filename=filename, log_info=log_info)
        result = np.hstack(( paths[:, np.newaxis], preds[:, np.newaxis]))
        dump(result, os.path.join(self.work_dir, filename))

        if isinstance(ann_nums, dict) and len(ann_nums) >= 2:
            ind = 0
            for ann, num in ann_nums.items():
                self.logger.info(f"{ann}: {num}")
                self.metric(preds[ind:ind+num], labels[ind:ind+num], dataname=ann, log_info=log_info)
                ind += num
        if len(feats) > 0:
            feats = np.concatenate(feats)
            if self.eval_cfg.tsne_cfg is not None:
                self.logger.info('vis tsne...')
                self.vis_tsne(feats, labels, paths)
        self.logger.info('End of testing!')
        return preds, labels

    def train(self, dataloader, cfg):
        """train method"""
        self._total_epoch = cfg.total_epochs
        self._total_iter = self._get_total_iter(dataloader)
        self._train_swad = cfg.get('train_swad')
        if self._train_swad:
            self._train_swad['freeze_bn'] = cfg.model.train_cfg.get('freeze_bn', False)
            self.avgmodel = AveragedModel(self.model)

        self.logger.info(f'Start training from the {self._epoch} Epoch, {self._iter} Iter.')
        self.logger.info(f'Total {self._total_epoch} Epochs, {self._total_iter} Iters, {len(dataloader)} Iter/Epoch.')

        self.model.train()
        start_epoch = self._total_epoch if self._train_swad and self._train_swad.get('only_swad') else self._epoch 
        for epoch in range(start_epoch, self._total_epoch):

            self._train_epoch(dataloader)
            
            if self.log_cfg.plog_cfg is not None and is_main_process():
                self.vis_log(self.log_cfg.filename)

            self._epoch += 1
            if not self._every_iter:
                self.scheduler.step()
            self._save_model(filename='latest.pth')

        if self._train_swad and is_main_process():
            export_swad_model(self.avgmodel, dataloader, self._train_swad, self.work_dir, self.logger)
            if self.log_cfg.plog_cfg is not None:
                self.vis_log(self.log_cfg.filename)

        self.logger.info('End of training!')

    @torch.no_grad()
    def val_step(self, dataloader):
        """val method, for step_learning"""
        # feats = list()
        preds = list()
        self.model.eval()
        for data in tqdm(dataloader):
            data.pop('path', 'unknow')
            output = self.model(**data)
            preds.append(output['pred'].detach().cpu().numpy())  # binary
            # if len(output) > 2:
            #     feats.append(output[2].detach().cpu().numpy())
        # self.model.train()
        preds = np.concatenate(preds)
        return preds
