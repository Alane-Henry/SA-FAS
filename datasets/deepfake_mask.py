# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现业务上活体检测数据集类'DFDDataset'.

"""

import os
import cv2
import copy
import random
import warnings
import numpy as np
import albumentations as alb
from utils.fileio import load
from utils.cv_util import distance_pt
from datasets.custom import CustomDataset


Interpolation = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT, cv2.INTER_NEAREST_EXACT]

class DFDMDataset(CustomDataset):
    """YeWu dataset for FAS.
    The annotation format is show as follows:
        -- annotation.txt
            ...
            img_path/img_name.jpg x_1 y_1 ... x_72 y_72 label
            ...
    Args:
        ann_file (str or list[str]): Annotation file path
        pipeline (dict): Processing pipeline.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``mask_prefix``,  if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        enlarge (float): enlarge face to crop according to landmarks.
    """
    NAME = "DFDMDataset"

    def __init__(self,
                 data_root,
                 ann_files,
                 pipeline=None,
                 img_prefix='',
                 test_mode=False,
                 enlarge=3.0,
                 img_size=(256,256),
                 selfblended_prob=0.0,
                 compression=0,
                 crop_method="all_lmks_of_72",
                 lamk_jit={"is_used":False},
                 ct_jit={"is_used":False},
                 crop_jit={"is_used":False}):
        self.enlarge = enlarge
        self.img_size = img_size
        self.lamk_jit = lamk_jit
        self.ct_jit = ct_jit
        self.crop_jit = crop_jit
        self.lmks = None
        self.jpeg_compression = alb.Compose([alb.ImageCompression(quality_lower=compression[0], quality_upper=compression[1], p=1.0)], 
                                            additional_targets={'mask': 'image'}) if compression else None
        self.jpeg_compression95 = alb.Compose([alb.ImageCompression(quality_lower=95, quality_upper=95, p=1.0)], additional_targets={'mask': 'image'})
        # crop index[[eyes indx], [crop face index]]
        _crop_inds = {"4_lmks_of_72": [[21,38], [21, 38, 58, 62], 144],
                        "4_lmks_of_5": [[0,1], [0, 1, 3, 4], 10],
                        "4_lmks_of_106": [[61, 71],[61, 71, 88, 94], 212],
                        "2eyes_lmks_of_72":[[21,38], [21,38], 144],
                        "all_lmks_of_72": [[21,38], [], 144]}
        self.lamk_inds = _crop_inds[crop_method]
        self.lm_num = self.lamk_inds[2]

        super(DFDMDataset, self).__init__(
            data_root,
            ann_files,
            pipeline,
            img_prefix,
            test_mode)

    def load_annotations(self, ann_files):
        """load annotation information"""
        ann_nums = dict()
        _lmks = list()
        _labels = list()
        _filenames = list()
        _domain = list()
        for i, ann_file in enumerate(ann_files):
            ann_file_name = os.path.splitext(os.path.basename(ann_file))[0]
            lines, thr = self._parse_thr_from_filename(ann_file)
            count = 0
            for line in lines:
                if thr[0] < 0:
                    label = int(line[-1]) if len(line)>1 else -1
                else:
                    if (line[-2] >= thr[0]) and (line[-2] <= thr[1]):
                        continue
                    if line[-2] < thr[0] and line[-1] == 1:
                        label = 1
                    else:
                        label = 0
                # warning: multi_label be forbiden
                label = 1 if label >=1 else 0
                try:
                    float(line[1])
                    _lmks.append(np.array(line[1:1+self.lm_num],dtype=np.float32).reshape(1, -1) if self.enlarge!=0 and len(line)>self.lm_num else np.zeros((1,self.lm_num)))
                    _filenames.append([os.path.join(self.img_prefix[i], line[0])])
                except:
                    _lmks.append(np.array(line[2:2+self.lm_num],dtype=np.float32).reshape(1, -1) if self.enlarge!=0 and len(line)>self.lm_num else np.zeros((1,self.lm_num)))
                    _filenames.append([os.path.join(self.img_prefix[i], line[0]), os.path.join(self.img_prefix[i], line[1])])
                    label = 1
                count += 1
                _labels.append(label)
                _domain.append(ann_file_name)
            ann_nums[ann_file_name] = count
 
        self.labels = np.array(_labels)
        self.lmks = np.concatenate(_lmks, axis=0)
        self.filenames = np.array(_filenames)
        self.domain = np.array(_domain)
        self.ann_nums = ann_nums

    def _parse_thr_from_filename(self, ann_file):
        thr = [-1, -1]
        if 'pseudo_' in ann_file:
            thr = [0.1, 0.9]
            data = os.path.splitext(os.path.basename(ann_file))[0].split('_')
            if len(data) == 4:
                ann_file = os.path.join(os.path.dirname(ann_file), f'pseudo_{data[1]}.txt')
                thr = [float(f'0.{data[2]}'), float(f'0.{data[3]}')]
            print('{} Neg thr: {}, Pos thr: {}'.format(os.path.basename(ann_file), thr[0], thr[1]))
        lines = load(ann_file)

        return lines, thr

    def _landmark_jitter(self, lamk, jitter_range=0.01, freeze_ratio=0.3, lamk_num=72, is_used=False):
        if is_used:
            """jitter face landmark"""
            jit = np.random.normal(1.0, jitter_range, lamk_num*2).reshape(lamk_num, 2)
            freeze_ind = np.random.randint(0, lamk_num, int(lamk_num*freeze_ratio))
            jit[freeze_ind] = 1.0
            lamk *= jit
        else:
            pass
        return lamk

    def _crop_jitter(self, wh, mean=1.0, std=0.1, is_used=False):
        if is_used:
            wh *= np.random.normal(mean, std)
        else:
            pass
        return float(wh)

    def _ct_jitter(self, ct, w, h , mean=[0,0], std=[0.07, 0.07], is_used=False):
        if is_used:
            _x, _y = np.random.normal(mean, std, 2)
            ct[0] = ct[0] + w * _x
            ct[1] = ct[1] + h * _y
        else:
            pass
        return ct

    def _crop_warp(self, img, mask, lamk):
        """crop image according ot landmark"""
        assert lamk.shape[1] == 2,"lamk must be [N,2] shape!"
        # set left, right eyes index
        idx1, idx2 = self.lamk_inds[0][0], self.lamk_inds[0][1]

        lamk = self._landmark_jitter(lamk, **self.lamk_jit, lamk_num=len(lamk))

        angle = 0
        if (lamk[idx1] > 0).all() and (lamk[idx2] > 0).all():
            angle += np.arctan2(lamk[idx2, 1] - lamk[idx1, 1], lamk[idx2, 0] - lamk[idx1, 0]) * 180.0 / np.pi

        if len(self.lamk_inds[1]) == 4:
            ct = np.mean(lamk[self.lamk_inds[1], :], axis=0)

            # lamk_inds should be lt, rtm ld, rd
            _w = 0.5*distance_pt(lamk[self.lamk_inds[1][0]], lamk[self.lamk_inds[1][1]]) + \
                    0.5*distance_pt(lamk[self.lamk_inds[1][2]], lamk[self.lamk_inds[1][3]])
            _h = 0.5*distance_pt(lamk[self.lamk_inds[1][0]], lamk[self.lamk_inds[1][2]]) + \
                    0.5*distance_pt(lamk[self.lamk_inds[1][1]], lamk[self.lamk_inds[1][3]])
            wh = np.sqrt(_w*_h)
            wh = wh if wh != 0 else 10

            wh = self._crop_jitter(wh, **self.crop_jit)
            ct = self._ct_jitter(ct, wh, wh, **self.ct_jit)

        elif len(self.lamk_inds[1]) == 2:
            ct = np.mean(lamk[self.lamk_inds[1], :], axis=0)
            wh = distance_pt(lamk[self.lamk_inds[1][0]], lamk[self.lamk_inds[1][1]])
            
            wh = self._crop_jitter(wh, **self.crop_jit)
            ct = self._ct_jitter(ct, wh, wh, **self.ct_jit)

        else:
            rd = np.max(lamk, axis=0)
            lt = np.min(lamk, axis=0)
            ct, wh = (rd + lt) / 2, (rd[0] - lt[0])

            # if self.crop_jit:
            #     lt, rd = self._crop_jitter(lt, rd)
            ct, wh = (rd + lt) / 2, (rd[0] - lt[0])
            ct = self._ct_jitter(ct, wh, wh, **self.ct_jit)

        enlarge = np.random.rand() * (self.enlarge[1] - self.enlarge[0]) + self.enlarge[0] if isinstance(self.enlarge, (list, tuple)) else self.enlarge 
        img_size = self.img_size
        # if img_size[0] > (wh * enlarge):
        #     size = max(int(wh * enlarge), 112)
        #     img_size = (size, size)
        M = cv2.getRotationMatrix2D((ct[0], ct[1]), angle, img_size[0] / (wh * enlarge))
        M[0, 2] = M[0, 2] - (ct[0] - img_size[0] / 2.0)
        M[1, 2] = M[1, 2] - (ct[1] - img_size[1] / 2.0)

        img = cv2.warpAffine(img, M, img_size)
        mask = cv2.warpAffine(mask, M, img_size)

        return img, mask, M

    def _get_ann_info(self, filename, lmk, label, domain):
        """read images and annotations"""
        img = cv2.imread(filename[0])
        # if 'png' in filename[0]:
        #     img = self.jpeg_compression95(image=img)['image']
        if len(filename)==2:
            img_real = cv2.imread(filename[1], cv2.IMREAD_GRAYSCALE)
            img_fake = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_real = cv2.resize(img_real, img_fake.shape[::-1])
            resimg = cv2.absdiff(img_fake,img_real)
            ret, mask = cv2.threshold(resimg, thresh=10, maxval=1, type=cv2.THRESH_BINARY)
            if mask.mean() <= 0:
                print(filename[0], mask.mean())
        else:
            mask = np.zeros(img.shape[:2])
            # mask = np.zeros(img.shape[:2]) if label == 0 else np.ones(img.shape[:2])*-1
        if self.jpeg_compression is not None and not self.test_mode and np.random.randint(2):
            flag=random.random()
            if flag<0.33:
                img = cv2.GaussianBlur(img, (5, 5), 0)
            elif flag<0.66:
                first, second = (0.5, 2) if np.random.randint(2) else (2, 0.5)
                img = cv2.resize(img, None, fx=first, fy=first, interpolation=random.choice(Interpolation))
                img = cv2.resize(img, None, fx=second, fy=second, interpolation=random.choice(Interpolation))
            else:
                augmented = self.jpeg_compression(image=img)
                img = augmented['image']
        if lmk.sum()==0:
            img = cv2.resize(img, self.img_size)
            mask = cv2.resize(mask, self.img_size)
        else:   
            img, mask, _ = self._crop_warp(img, mask, lmk)
        data = dict(
            img=img,
            mask=mask,
            label=np.ones((1,)).astype(np.int64) * label,
            path=os.path.relpath(filename[0], self.img_prefix[0]) if len(set(self.img_prefix))==1 else filename[0],
            domain=domain
            )
        return data
    
    def __getitem__(self, idx):
        while True:
            label = self.labels[idx]
            lmk = self.lmks[idx].reshape(-1,2)
            filename = self.filenames[idx]
            domain = self.domain[idx]
            try:
                data = self._get_ann_info(filename, lmk, label, domain)
            except:
                warnings.warn('Fail to read image: {} {}'.format(filename, label))
                idx = self._rand_another(idx)
                continue
            break
        data = self.pipeline(data)
        return data
    