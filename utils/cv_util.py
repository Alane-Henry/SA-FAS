# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现了cv相关的一些图像处理函数

"""

import cv2
import numpy as np
import random

def distance_pt(pt1, pt2):
    return ((pt2[1] - pt1[1])**2 +  (pt2[0] - pt1[0])**2)**0.5

def expand2square_cv2(cv2_img, background_color):
    # 获取原始宽度和高度
    height, width = cv2_img.shape[:2]

    if width == height:
        return cv2_img
    elif width > height:
        # 宽大于高，创建一个新的方形背景
        result = np.full((width, width, 3), background_color, dtype=cv2_img.dtype)
        # 将原始图像粘贴到背景的中央
        result[(width - height) // 2:(width - height) // 2 + height, 0:width] = cv2_img
    else:
        # 高大于宽，创建一个新的方形背景
        result = np.full((height, height, 3), background_color, dtype=cv2_img.dtype)
        # 将原始图像粘贴到背景的中央
        result[0:height, (height - width) // 2:(height - width) // 2 + width] = cv2_img
    return result

def AffineRect(img, center, size, angle, shear_x, shear_y):
    # 计算旋转矩形的四个角点  
    # 创建一个(0, 0)为中心，宽高为size的矩形，并找到其四个角点  
    width_half = size[0] / 2.0  
    height_half = size[1] / 2.0  
        
    # 原始矩形的四个角点（逆时针顺序）  
    points = np.array([[ 
        [-width_half, -height_half],  
        [width_half, -height_half],  
        [width_half, height_half],  
        [-width_half, height_half]  
    ]], dtype=np.float32)  
    points += center
    # 错切矩阵
    shear_mat = np.array([
        [1, np.tan(np.radians(shear_x)), 0],
        [0, 1, 0],
        # [0,0,1]
        ])
    # 旋转矩阵  
    # rotation_mat = cv2.getRotationMatrix2D((0, 0), angle, 1)  
    # rotation_mat = np.insert(rotation_mat, 2, [0,0,1], axis=0)
    
    # Mat = np.dot(rotation_mat, shear_mat)

    # affine_img = cv2.warpAffine(img, Mat[:2], img.shape[:2])
    # affine_points = np.int0(cv2.transform(points, Mat))[:,:,:2]
    affine_img = cv2.warpAffine(img, shear_mat, img.shape[:2])
    affine_points = np.int0(cv2.transform(points, shear_mat))

    return affine_points, affine_img


def perspective(img, M3, img_size, center, rect_size, var=20):
    width_half = rect_size[0] / 2.0  
    height_half = rect_size[1] / 2.0  
        
    # 原始矩形的四个角点（顺时针顺序）  
    points = np.array([
        [-width_half, -height_half],  
        [width_half, -height_half],  
        [width_half, height_half],  
        [-width_half, height_half]  
    ], dtype=np.float32)  
    points += center

    # 生成一个均值为0，方差为var的正态分布随机数数组
    hShift = np.random.normal(0, var, size=(4,1))
    vShift = np.random.normal(0, var, size=(4,1))
    # 保持水平
    vShift[0] = -vShift[3] 
    vShift[1] = -vShift[2]
    # new_points = points - np.hstack((hShift, [[0]]*4)).astype(np.float32)
    # new_points = points - np.hstack(([[0]]*4, vShift)).astype(np.float32)
    new_points = points - np.hstack((hShift, vShift)).astype(np.float32)
    new_points = new_points.round()

    pM = cv2.getPerspectiveTransform(points, new_points)
    M = np.dot(pM, M3) # 基础变换M3+透视变换
    # M = M3 # 基础变换M3+透视变换
    perspective_img = cv2.warpPerspective(img, M, img_size)

    return np.int0(new_points), perspective_img


def draw_AffineRect(img, affine_args, thickness_var=20, mask=None):
    # 将角度转换为弧度  
    
    # 计算矩形的角点坐标（在旋转之前）  
    # box_points, img = AffineRect(img, **affine_args)  
    box_points, perspective_img = perspective(img, **affine_args)  

    if mask is not None:
        cv2.fillConvexPoly(mask, box_points, (255,255,255), cv2.LINE_AA)
    
    # 使用cv2.polylines函数绘制矩形的四条边  
    # cv2.polylines(img, [box_points], True, color, thickness)
    # center = lamk_warp[np.random.choice(range(len(lamk_warp)))]
    def generate_random_color(ran):
        return (random.choice(ran), random.choice(ran), random.choice(ran))
    
    black = range(30)
    white = range(230,256)
    color=generate_random_color(black if np.random.uniform(0, 1) < 0.75 else white)
    border = [round(np.abs(np.random.normal(0, thickness_var)))+thickness_var, round(np.abs(np.random.normal(0, thickness_var)))+thickness_var]
    for i in range(4):
        cv2.line(perspective_img, box_points[i], box_points[i-1], color, border[i%2], cv2.LINE_AA) # cv2.LINE_AA抗锯齿

    if np.random.uniform(0, 1) < 0.8: # double border
        # grayk = random.choice(range(0,256))
        # colormarge = 10
        # color=generate_random_color(range(max(0, grayk-colormarge), min(256, grayk+colormarge)))
        black = range(40)
        white = range(220,256)
        color=generate_random_color(black if np.random.uniform(0, 1) < 0.5 else white)
        border = np.array(border) - np.random.randint(min(thickness_var, min(border)//2))
        for i in range(4):
            cv2.line(perspective_img, box_points[i], box_points[i-1], color, border[i%2], cv2.LINE_AA) # cv2.LINE_AA抗锯齿
    
    mask_black = np.ones_like(perspective_img, dtype=np.uint8) 
    if np.random.uniform(0, 1) < 1:
        if box_points[0, 0] < box_points[3, 0]:
            x1 = random.randint(box_points[0, 0], box_points[3, 0])
        else:
            x1 = random.randint(box_points[3, 0], box_points[0, 0])
        if box_points[1, 0] < box_points[2, 0]:
            x2 = random.randint(box_points[1, 0], box_points[2, 0])
        else:
            x2 = random.randint(box_points[2, 0], box_points[1, 0])
        mask_black[:, :max(0,x1)] = 0
        mask_black[:, x2:] = 0

    return perspective_img, mask_black

  