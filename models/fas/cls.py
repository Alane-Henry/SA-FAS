# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现 CLS 分类方法

"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.backbones import encoders
from models.losses import OneCenterLoss

def freeze(model, layers):
    if layers is None: # 无参数不更改
        return

    for param in model.parameters(): # 冻结所有
        param.requires_grad = False

    if layers != 'None': # 如果是 'None' 不解冻，全冻结
        for name, param in model.named_parameters():
            if any(name.startswith(layername) for layername in layers):
                param.requires_grad = True
            
    for name, param in model.named_parameters():
        print(name, ':', param.requires_grad)
        

def weights_init(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_running_stats()
        if m.affine:
            init.uniform_(m.weight)
            init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(m.bias, -bound, bound)

class CLS(nn.Module):
    """General classification network.

    Args:
        Args:
        encoder (dict): the config dict of encoder network.
        feat_dim (tuple): in_channels and out_channels of fc layer,
            (in_channels, out_channels).
        test_cfg (dict): the config dict of testing setting.
        train_cfg (dict): the config dict of training setting, including
            some hyperparameters of loss.

    """
    def __init__(self,
                 encoder,
                 feat_dim=(2048, 1),
                 test_cfg=None,
                 train_cfg=None):
        super(CLS, self).__init__()
        assert isinstance(encoder, dict)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.feat_dim = feat_dim

        pretrained = encoder.get('pretrained', None)
        self.encoder = encoders(encoder)
        freeze(self.encoder, train_cfg.get('unfreeze'))
        if not pretrained:
            self.encoder.apply(weights_init)

        self.classifier = []
        for i in range(1, len(feat_dim)):
            self.classifier.append(nn.Sequential(
                nn.Linear(feat_dim[i-1], feat_dim[i], bias=True),
                nn.ReLU() if i != len(feat_dim)-1 else nn.Identity())
            )
        self.classifier = nn.Sequential(*self.classifier)
            
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        # self.oc_loss = OneCenterLoss()

    def _get_losses(self, logit, feat, label, domain):
        """calculate training losses"""
        if label is None:
            return {}        
        if 'w_cls' in self.train_cfg:
            # loss_oc = self.oc_loss(feat, label[:, 0]).unsqueeze(0) * self.train_cfg['w_oc']
            loss_cls = self.cls_loss(logit, label[:, 0]).unsqueeze(0) * self.train_cfg['w_cls']
            # loss = loss_oc + loss_cls
            loss = loss_cls
            # return dict(loss_cls=loss_cls, loss_oc=loss_oc, loss=loss)
            return dict(loss_cls=loss_cls, loss=loss)
        if 'w_bce' in self.train_cfg:
            loss_bce = self.bce_loss(logit[:, 0], 1 - label[:, 0].float()).unsqueeze(0) * self.train_cfg['w_bce']
            return dict(loss_bce=loss_bce, loss=loss_bce)

    def forward(self, img, label=None, domain=None):
        """forward"""
        feat = self.encoder(img)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if len(feat.shape) == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).squeeze(3).squeeze(2)
        logit = self.classifier(feat)
        output = self._get_losses(logit, feat, label, domain)
        if self.training:
            return output
        else:
            output['pred'] = torch.sigmoid(logit/5)[:, 0] if self.feat_dim[-1] == 1 else F.softmax(logit, dim=1)[:, 0]
            if self.test_cfg.get('return_label', True):
                output['label'] = label
            if self.test_cfg.get('return_feature', False):
                output['feat'] = feat
            return output['pred'].unsqueeze(1) if torch.onnx.is_in_onnx_export() else output
        

