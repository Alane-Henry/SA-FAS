# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现 CLSLora/ CLS 分类方法

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.backbones import encoders
from utils.load_model import load_model
from models.losses import FocalLoss, ContrastiveLoss, SoftMultiContrastiveLoss,ContrastiveALLgatherLoss
import random
from collections import Counter

lable_dict = {'0_0_0': 'Live Face',
                '1_0_0': 'Print',
                '1_0_1': 'Replay',
                '1_0_2': 'Cutouts',
                '1_1_0': 'Transparent',
                '1_1_1': 'Plaster',
                '1_1_2': 'Resin',
                '2_0_0': 'Attribute-Edit',
                '2_0_1': 'Face-Swap',
                '2_0_2': 'Video-Driven',
                '2_1_0': 'Pixel-Level',
                '2_1_1': 'Semantic-Level',
                '2_2_0': 'ID_Consisnt',
                '2_2_1': 'Style',
                '2_2_2': 'Prompt',
                '3_0_0': 'Fake'}
label2class_dict = {
        'Live Face': 0,
        'Print': 1,
        'Replay': 2,
        'Cutouts': 3,
        'Transparent': 4,
        'Plaster': 5,
        'Resin': 6,
        'Attribute-Edit': 7,
        'Face-Swap': 8,
        'Video-Driven': 9,
        'Pixel-Level': 10,
        'Semantic-Level': 11,
        'ID_Consisnt': 12,
        'Style': 13,
        'Prompt': 14,
        'Fake': 15
}
class2label = {v: k for k, v in label2class_dict.items()}
def get_labels(files):
    labels = []
    import os
    for file in files:
        file = os.path.join('/workspace/iccv2025_face_antispoofing/', file)
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            label = label2class_dict[lable_dict[line[-1]]]
            labels.append(label)
    return np.array(labels)

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
            
class CLSLoraTextAnchor(nn.Module):
    """General classification network.

    Args:
        Args:
        encoder (dict): the config dict of encoder network.
        projection_dim (tuple): in_channels and out_channels of projection layer,
            (in_channels, out_channels).
        out_channels (int): the number of output channels.
        test_cfg (dict): the config dict of testing setting.
        train_cfg (dict): the config dict of training setting, including
            some hyperparameters of loss.

    """
    def __init__(self,
                 encoder,
                 feat_dim=(2048, 1),
                 test_cfg=None,
                 train_cfg=None,
                 text_feat_pth='extracted_text_features.pth'):
        super(CLSLoraTextAnchor, self).__init__()
        assert isinstance(encoder, dict)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.feat_dim = feat_dim
        self.prompts_num_per_class = 5
        self.vote_topk = 10
        
        pretrained = encoder.get('pretrained', None)
        self.encoder = encoders(encoder)
        self.text_features = nn.ParameterDict({
            key: nn.Parameter(torch.randn(5, self.feat_dim[0]), requires_grad=False) for key in label2class_dict.keys()
        })
        if text_feat_pth:
            self.text_features.load_state_dict(torch.load(text_feat_pth))

        self.classifier = []
        for i in range(1, len(feat_dim)):
            self.classifier.append(nn.Sequential(
                nn.Linear(feat_dim[i-1], feat_dim[i], bias=True),
                nn.ReLU() if i != len(feat_dim)-1 else nn.Identity())
            )
        self.classifier = nn.Sequential(*self.classifier)
        if not pretrained:
            self.encoder.apply(weights_init)
        elif isinstance(pretrained, str):
            load_model(pretrained, self)

        # set lora model
        if train_cfg.get('lora_modules'):
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=train_cfg.get('lora_modules'),
                lora_dropout=0.1,
                bias="none",
            )
            print('Lora config:', config)
            self.encoder = get_peft_model(self.encoder, config)
            self.encoder.print_trainable_parameters()
        
        # unfreeze some modules
        freeze(self, train_cfg.get('unfreeze'))
            
        # self.bce_loss = nn.BCEWithLogitsLoss()
        # self.cls_loss = FocalLoss()
        # self.supcon_loss = SupConLoss()
        strategy = train_cfg.get('weight_balance_strategy')
        ann_files = train_cfg.get('ann_files')
        all_labels = get_labels(ann_files)
        num_classes = len(label2class_dict) 
        class_freq = np.bincount(all_labels, minlength=num_classes).astype(np.float32)
        print('class_freq', class_freq)
        class_freq[0] = 16.0
        class_freq[-1] = 08.0
        print('class_freq_modify', class_freq)
        unseen_mask = class_freq == 0

        # Avoid division by zero
        freq = np.clip(class_freq, a_min=1e-6, a_max=None)

        if strategy == 'inv':
            weights = 1.0 / freq
        elif strategy == 'sqrt_inv':
            weights = 1.0 / np.sqrt(freq)
        elif strategy == 'log_inv':
            weights = 1.0 / np.log(1.1 + freq)
        elif strategy == 'squ_inv':
            weights = 1.0 / (freq ** 2)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Set unseen class weights to 0
        weights[unseen_mask] = 0.0
        self.class_weights = torch.tensor(weights, dtype=torch.float32) * 1000

        self.contrastive_loss = ContrastiveLoss(temperature=0.07, class_weights=self.class_weights, need_norm=False)
        self.cls_loss = nn.CrossEntropyLoss(weight=self.class_weights)

    def _get_losses(self, logit, feat, label, domain):
        """calculate training losses"""
        if label is None:
            return {}
        
        text_anchors = [self.text_features[class2label[i]][torch.randint(0, 5, (1,), device=feat.device).item()] for i in range(len(label2class_dict))]
        text_anchors = torch.stack(text_anchors, dim=0).to(feat.device)
        if 'w_cls' in self.train_cfg:
            loss_cls = self.cls_loss(logit, label[:, 0]).unsqueeze(0) * self.train_cfg['w_cls']
        else:
            loss_cls = self.cls_loss(logit, label[:, 0]).unsqueeze(0)

        feat = feat / feat.norm(dim=-1, keepdim=True)
        loss_contrastive = self.contrastive_loss(feat, text_anchors, label[:,0]).unsqueeze(0)

        if 'w_contrastive' in self.train_cfg:
            loss_contrastive = loss_contrastive * self.train_cfg['w_contrastive']
        
        return dict(loss_cls=loss_cls, loss_contrastive=loss_contrastive, loss=loss_cls + loss_contrastive)
        
    def forward(self, img, label=None, domain=None):
        """forward"""
        feat = self.encoder(img)

        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if len(feat.shape) == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).squeeze(3).squeeze(2)
        
        logit = self.classifier(feat)
        if self.training:
            output = self._get_losses(logit, feat, label, domain)
            return output
        else:
            output = {}
            # feat = feat / feat.norm(dim=-1, keepdim=True)
            # all_text_feat = [self.text_features[class2label[i]][:] for i in range(len(label2class_dict))]
            # all_text_feat = torch.stack(all_text_feat, dim=0).view(-1, self.feat_dim[0]).to(feat.device)
            # similarity = (feat @ all_text_feat.t())
            # # print('similarity shape', similarity.shape)
            # top_scores, top_indices = similarity.topk(self.vote_topk)
            # # print('top_scores', top_scores)
            # # print('top_indices', top_indices)
            # top_major_indices = top_indices // self.prompts_num_per_class
            # # print('top_major_indices', top_major_indices)
            # major_class_indices = []

            pred_score = 1.0 - F.softmax(logit, dim=1)[:, 0]
            output['pred'] = pred_score
            # output['pred_cls'] = pred_cls
            output['pred_cls'] = torch.argmax(logit, dim=1)
            
            if self.test_cfg.get('return_label', True):
                output['label'] = label
            if self.test_cfg.get('return_feature', False):
                output['feat'] = feat
            return output['pred'].unsqueeze(1) if torch.onnx.is_in_onnx_export() else output
        

