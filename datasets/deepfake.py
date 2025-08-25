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
from utils.cv_util import distance_pt, expand2square_cv2
from datasets.custom import CustomDataset


Interpolation = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT, cv2.INTER_NEAREST_EXACT]

class DFDDataset(CustomDataset):
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
    NAME = "DFDDataset"

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
        self.jpeg_compression50 = alb.Compose([alb.ImageCompression(quality_lower=50, quality_upper=50, p=1.0)], additional_targets={'mask': 'image'})
        # crop index[[eyes indx], [crop face index]]
        _crop_inds = {"4_lmks_of_72": [[21,38], [21, 38, 58, 62], 144],
                        "4_lmks_of_5": [[0,1], [0, 1, 3, 4], 10],
                        "4_lmks_of_106": [[61, 71],[61, 71, 88, 94], 212],
                        "2eyes_lmks_of_72":[[21,38], [21,38], 144],
                        "all_lmks_of_72": [[21,38], [], 144]}
        self.lamk_inds = _crop_inds[crop_method]
        self.lm_num = self.lamk_inds[2]

        self.selfblended_prob = selfblended_prob
        self.transforms=self.get_transforms()
        self.source_transforms = self.get_source_transforms()

        super(DFDDataset, self).__init__(
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
        fiter_num = 0
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
                    _lmks.append(np.zeros((1,self.lm_num//2, 2)) if isinstance(self.enlarge, str) or self.enlarge==0 or len(line)<self.lm_num else np.array(line[1:1+self.lm_num],dtype=np.float32).reshape(1,-1,2))
                    count += 1
                except:
                    # if len(line) == 4:
                    #     if int(line[-2]) < 10000 or ('PS' not in ann_file and int(line[-2]) < 10000):
                    #         fiter_num += 1
                    #         continue
                    lmk1 = np.zeros((1,self.lm_num//2, 2)) if isinstance(self.enlarge, str) or self.enlarge==0 or len(line)<self.lm_num else np.array(line[2:2+self.lm_num],dtype=np.float32).reshape(1,-1,2)
                    lmk0 = lmk1 * (float(line[-3]), float(line[-2])) if len(line) > self.lm_num and '.' in line[-3] else lmk1 # 成对数据可能分辨率和lmk不一致
                    _lmks.append(lmk1)
                    _lmks.append(lmk0)
                    _filenames.append(os.path.join(self.img_prefix[i], line[1]))
                    _labels.append(0)
                    _domain.append(ann_file_name)
                    label = 1
                    count += 2
                _labels.append(label)
                _filenames.append(os.path.join(self.img_prefix[i], line[0]))
                _domain.append(ann_file_name)
            ann_nums[ann_file_name] = count
        print('fiter_num', fiter_num)
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

    def _crop_warp(self, img, lamk):
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

        return img, M
    
    def _selfblending(self, img, lamk):
        # crop and warp
        img, M = self._crop_warp(img, lamk)
        M_row3 = np.array([0,0,1])
        M3 = np.insert(M, 2, M_row3, axis=0)

        lamk_col = np.ones_like(lamk[:,0])
        lamk = np.column_stack((lamk, lamk_col))
    
        lamk_warp = lamk @ M3.T
        lamk = lamk_warp[:,:2]

        # selfblending
        lamk = lamk.astype(np.int32)
        H,W=len(img),len(img[0])
        # if exist_bi:
        #     logging.disable(logging.FATAL)
        #     mask=random_get_hull(landmark,img)[:,:,0]
        #     logging.disable(logging.NOTSET)
        mask=np.zeros_like(img[:,:,0])
        cv2.fillConvexPoly(mask, cv2.convexHull(lamk), 1.)

        source = img.copy()
        if np.random.rand()<0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source,mask)

        img_blended,mask=dynamic_blend(source,img,mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img_blended

    def _random_scale(self, img, lmk):
        # 随机非等比例resize
        h, w = img.shape[:2]
        if self.test_mode:
            aspect_ratio = w / h 
        elif h == w:
            aspect_ratio = random.choice([1,1,1,1,1,1, 2/3, 3/4, 4/5, 3/2, 4/3, 5/4])
        elif h > w:
            aspect_ratio = random.choice([1,1,1,1,1,1,1, 1/2, 2/3, 3/4, 4/5, 9/16, 10/16, 9/21])
        else:
            aspect_ratio = random.choice([1,1,1,1,1,1,1, 2/1, 3/2, 4/3, 5/4, 16/9, 16/10, 21/9])
        new_h = 640
        new_w = int(aspect_ratio * new_h)
        img = cv2.resize(img, (new_w, new_h), interpolation=random.choice(Interpolation))
        lmk = lmk * [new_w/w, new_h/h]
        return img, lmk
    
    def _scale_jitter(self, img, lmk, jitter_range=0.2):
        # 随机非等比例resize
        h, w = img.shape[:2]
        aspect_ratio = w / h 
        if not self.test_mode:
            aspect_ratio *= 1+random.uniform(-jitter_range, jitter_range)
        new_h = min(640, h)
        new_w = int(aspect_ratio * new_h)
        img = cv2.resize(img, (new_w, new_h), interpolation=random.choice(Interpolation))
        lmk = lmk * [new_w/w, new_h/h]
        return img, lmk

    def _get_ann_info(self, filename, lmk, label, domain):
        """read images and annotations"""
        img = cv2.imread(filename)
        # if img.shape[0] > 1600:
        #     scale = 1.0/random.uniform(2, 3) if not self.test_mode else 0.4
        #     img = cv2.resize(img, None, fx=scale, fy=scale)
        #     lmk = lmk * scale
        img, lmk = self._scale_jitter(img, lmk)
        # if os.path.splitext(filename)[-1].lower() not in ['.jpg', '.jpeg']:
        img = self.jpeg_compression95(image=img)['image'] # 放后面可以全都jpg
        # if self.jpeg_compression is not None and not self.test_mode and random.random()<0.5:
        #     # for i in range(random.choice([0,1,2,3])):
        #     augmented = self.jpeg_compression(image=img)
        #     img = augmented['image']
        # if self.jpeg_compression is not None and not self.test_mode and np.random.randint(2):
        #     flag=random.random()
        #     if flag<0.33:
        #         img = cv2.GaussianBlur(img, (5, 5), 0)
        #     elif flag<0.66:
        #         first, second = (0.5, 2) if np.random.randint(2) else (2, 0.5)
        #         img = cv2.resize(img, None, fx=first, fy=first, interpolation=random.choice(Interpolation))
        #         img = cv2.resize(img, None, fx=second, fy=second, interpolation=random.choice(Interpolation))
        #     else:
        #         augmented = self.jpeg_compression(image=img)
        #         img = augmented['image']
        if lmk.sum()==0:
            if self.enlarge == 'pad':
                img = expand2square_cv2(img, tuple(int(x*255) for x in (0.40821073, 0.4578275, 0.48145466)))
            elif self.enlarge == 'pad0':
                img = expand2square_cv2(img, (0, 0, 0))
            img = cv2.resize(img, self.img_size)
        elif self.test_mode or np.random.uniform(0, 1) >= self.selfblended_prob:#or label == 1:
            img, _ = self._crop_warp(img, lmk)
        else:
            img = self._selfblending(img, lmk)
            label = 1
        data = dict(
            img=img,
            label=np.ones((1,)).astype(np.int64) * label,
            path=os.path.relpath(filename, self.img_prefix[0]) if len(set(self.img_prefix))==1 else filename,
            domain=domain
            )
        return data
    
    def __getitem__(self, idx):
        while True:
            label = self.labels[idx]
            lmk = self.lmks[idx]
            filename = self.filenames[idx]
            domain = self.domain[idx]
            # data = self._get_ann_info(filename, lmk, label, domain)
            try:
                data = self._get_ann_info(filename, lmk, label, domain)
            except:
                warnings.warn('Fail to read image: {} {}'.format(filename, label))
                idx = self._rand_another(idx)
                continue
            break
        data = self.pipeline(data)
        return data
    
    def get_source_transforms(self):
        return alb.Compose([
                alb.Compose([
                        alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
                        alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                        alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
                    ],p=1),
    
                alb.OneOf([
                    RandomDownScale(p=1),
                    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                ],p=1),
                
            ], p=1.)

    def get_transforms(self):
        return alb.Compose([
            
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
            
        ], 
        additional_targets={f'image1': 'image'},
        p=1.)

    def randaffine(self,img,mask):
        f=alb.Affine(
                translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
                scale=[0.95,1/0.95],
                fit_output=False,
                p=1)
            
        g=alb.ElasticTransform(
                alpha=50,
                sigma=7,
                alpha_affine=0,
                p=1,
            )

        transformed=f(image=img,mask=mask)
        img=transformed['image']
        
        mask=transformed['mask']
        transformed=g(image=img,mask=mask)
        mask=transformed['mask']
        return img,mask 


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self,img,**params):
        return self.randomdownscale(img)

    def randomdownscale(self,img):
        keep_ratio=True
        keep_input_shape=True
        H,W,C=img.shape
        ratio_list=[2,4]
        r=ratio_list[np.random.randint(len(ratio_list))]
        img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

        return img_ds

def dynamic_blend(source,target,mask):
    mask_blured = get_blend_mask(mask)
    blend_list=[0.25,0.5,0.75,1,1,1]
    blend_ratio = blend_list[np.random.randint(len(blend_list))]
    mask_blured*=blend_ratio
    img_blended=(mask_blured * source + (1 - mask_blured) * target)
    return img_blended,mask_blured

def get_blend_mask(mask):
    H,W=mask.shape
    size_h=np.random.randint(192,257)
    size_w=np.random.randint(192,257)
    mask=cv2.resize(mask,(size_w,size_h))
    kernel_1=random.randrange(5,26,2)
    kernel_1=(kernel_1,kernel_1)
    kernel_2=random.randrange(5,26,2)
    kernel_2=(kernel_2,kernel_2)
    
    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured[mask_blured<1]=0
    
    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured = cv2.resize(mask_blured,(W,H))
    return mask_blured.reshape((mask_blured.shape+(1,)))