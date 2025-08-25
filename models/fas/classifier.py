# -*- coding: UTF-8 -*-
# !/usr/bin/env python3


"""
本文件实现 Classifier 普通分类方法

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import encoders

def set1channel(model, train_gray):
    if not train_gray:
        return
    for name, module in model.stem.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_module = nn.Conv2d(1, module.out_channels, kernel_size=module.kernel_size,   
                                       stride=module.stride, padding=module.padding, bias=module.bias is not None)
            new_module.weight.data[:, :1, :, :] = module.weight.data.mean(dim=1, keepdim=True)
            if module.bias is not None:  
                new_module.bias.data = module.bias.data 
            setattr(model.stem, name, new_module)  
            print(f'set1channel: stem.{name}')
            break

def freeze(model, layers):
    if layers is None: # 无参数不冻结
        return

    for param in model.parameters():
        param.requires_grad = False

    if layers != 'None': # 如果是 'None' 不解冻，全冻结
        for name, param in model.named_parameters():
            if any(layername in name for layername in layers):
                param.requires_grad = True
            
    for name, param in model.named_parameters():
        print(name, ':', param.requires_grad)

class Classifier(nn.Module):
    """General classification network.

    Args:
        Args:
        encoder (dict): the config dict of encoder network.
        test_cfg (dict): the config dict of testing setting.
        train_cfg (dict): the config dict of training setting, including
            some hyperparameters of loss.

    """
    def __init__(self,
                 encoder,
                 test_cfg=None,
                 train_cfg=None):
        super(Classifier, self).__init__()
        assert isinstance(encoder, dict)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.encoder = encoders(encoder)

        set1channel(self.encoder, train_cfg.get('gray'))
        freeze(self.encoder, train_cfg.get('unfreeze'))

        self.freeze_bn()

        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=train_cfg.pop('label_smoothing', 0))

    def _get_losses(self, feats, label):
        """calculate training losses"""
        if label is None:
            return {}
        if 'w_cls' in self.train_cfg:
            loss_cls = self.cls_loss(feats, label[:, 0]).unsqueeze(0) * self.train_cfg['w_cls']
            return dict(loss_cls=loss_cls, loss=loss_cls)

    def forward(self, img, label=None, domain=None):
        """forward"""
        feat = self.encoder(img)
        output = self._get_losses(feat, label)
        if self.training:
            return output
        else:
            output['pred'] = F.softmax(feat, dim=1)[:, 0]
            if self.test_cfg.get('return_label', True):
                output['label'] = label
            if self.test_cfg.get('return_feature', False):
                output['feat'] = feat
            return output['pred'].unsqueeze(1) if torch.onnx.is_in_onnx_export() else output

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        if not self.train_cfg.get('freeze_bn', False):
            return
        print('freeze_bn')
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()