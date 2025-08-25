import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from copy import deepcopy
from typing import List, Tuple


def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]
    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        else:
            return module
    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)
            else:
                module2 = module2[int(l)]
            lst_index += 1
    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)
        else:
            module = module[int(l)]
    l = layer[lst_index]
    setattr(module, l, val)

# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)  

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    

class Linear(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Wraps torch.nn.Linear to support AMP + torchscript usage by manually casting
    weight & bias to input.dtype to work around an issue w/ torch.addmm in this use case.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)
        

def modification(parent_module, pruned_txt=None, old_txt=None):
    if pruned_txt is None:
        return parent_module
    state_dict = {}
    if old_txt:
        old_txt = open(old_txt, 'w')
    else:
        model_string = open(pruned_txt, 'r').readlines()
        for k in model_string:
            k = k.split(' ')
            state_dict[k[0]] = [int(i) for i in k[1:]]

    new_module = deepcopy(parent_module)
    for n, m in parent_module.named_modules():
        old_module = extract_layer(parent_module, n)
        if isinstance(old_module, nn.Conv2d) or isinstance(old_module, Conv2dSame):
            if old_txt:
                old_txt.write(f'{n} {old_module.in_channels} {old_module.out_channels} {old_module.groups}\n')
            else:
                if isinstance(old_module, Conv2dSame):
                    conv = Conv2dSame
                else:
                    conv = nn.Conv2d
                new_conv = conv(
                    in_channels=state_dict[n][0], out_channels=state_dict[n][1], kernel_size=old_module.kernel_size,
                    bias=old_module.bias is not None, padding=old_module.padding, dilation=old_module.dilation,
                    groups=state_dict[n][2], stride=old_module.stride)
                set_layer(new_module, n, new_conv)
        if isinstance(old_module, nn.BatchNorm2d):
            if old_txt:
                old_txt.write(f'{n} {old_module.num_features}\n')
            else:
                new_bn = nn.BatchNorm2d(
                    num_features=state_dict[n][0], eps=old_module.eps, momentum=old_module.momentum,
                    affine=old_module.affine, track_running_stats=True)
                set_layer(new_module, n, new_bn)
        if isinstance(old_module, nn.Linear):
            if old_txt:
                old_txt.write(f'{n} {old_module.in_features} {old_module.out_features}\n')
            else:
                new_fc = Linear(
                    in_features=state_dict[n][0], out_features=state_dict[n][1], bias=old_module.bias is not None)
                set_layer(new_module, n, new_fc)
                if hasattr(new_module, 'num_features'):
                    new_module.num_features = state_dict[n][0]
    if old_txt:
        old_txt.close()
    new_module.eval()
    return new_module

if __name__=='__main__':
    import timm.models as tm
    print(modification(tm.resnest14d(), pruned_txt='old.txt', old_txt='old.txt'))