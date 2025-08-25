# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现数据集基础类'CustomDataset'模板.

"""

import os
import cv2
import copy
import warnings
import numpy as np
from utils.fileio import load
from torch.utils.data import Dataset
from datasets.transform import Transforms


class CustomDataset(Dataset):
    """Custom dataset for FAS.
    The annotation format is show as follows:
        [
            {
                'filename': 'image name',
                'label': 0/1 (0: real, 1: fake)
            }
            ...
        ]
    The dataset format is show as follows:
        —— data_root
            —— ann_file
            —— img_prefix
    Args:
        ann_file (str or list[str]): Annotation file path
        pipeline (dict): Processing pipeline.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``,  if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
    """
    NAME = "CustomDataset"

    def __init__(self,
                 data_root,
                 ann_files,
                 pipeline=None,
                 img_prefix='',
                 test_mode=False):
        super(CustomDataset, self).__init__()
        if os.path.isabs(data_root):
            data_root = os.path.abspath(data_root)
        self.labels = None
        self.filenames = None
        self.data_root = data_root
        self.test_mode = test_mode
        self.ann_files = ann_files if isinstance(ann_files, list) else [ann_files]
        self.img_prefix = img_prefix if isinstance(img_prefix, list) else [img_prefix]

        if len(self.img_prefix) == 1:
            self.img_prefix *= len(self.ann_files)
        elif len(self.img_prefix) != len(self.ann_files):
            raise ValueError("num of img_prefix not equal to ann_files!")

        for i in range(len(self.ann_files)):
            if not os.path.isabs(self.ann_files[i]):
                self.ann_files[i] = os.path.join(self.data_root, self.ann_files[i])
            if not os.path.isabs(self.img_prefix[i]):
                self.img_prefix[i] = os.path.join(self.data_root, self.img_prefix[i])

        self.pipeline = Transforms(pipeline)
        self.load_annotations(self.ann_files)
        self.set_group_flag()

    def load_annotations(self, ann_files):
        """load annotation information"""
        _labels = list()
        ann_nums = dict()
        _filenames = list()
        for ann_file in ann_files:
            ann_infos = load(ann_file)
            ann_nums[os.path.splitext(os.path.basename(ann_file))[0]] = len(ann_infos)
            for ann_info in ann_infos:
                _filenames.append(os.path.join(self.img_prefix, ann_info['filename']))
                _labels.append(ann_info['label'])

        assert len(_filenames) == len(_labels),"num of files not equal to labels!"
        self.filenames = np.array(_filenames, dtype=np.string_)
        self.labels = np.array(_labels)

    def set_group_flag(self):
        """Set flag according to label"""
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = self.labels[i]
        self.groups = np.bincount(self.flag)

    def _rand_another(self, idx):
        """random select another index"""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _get_ann_info(self, filename, label):
        """read images and annotations"""
        data = dict(
            img=cv2.imread(filename),
            label=np.ones((1,)).astype(np.int64) * label)
        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        while True:
            try:
                filename = copy.deepcopy(self.filenames[idx])
                label = copy.deepcopy(self.labels[idx])
                data = self._get_ann_info(filename, label)
            except:
                warnings.warn('Fail to read image: {}'.format(filename))
                idx = self._rand_another(idx)
                continue
            break
        data = self.pipeline(data)
        return data
