# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

"""
本文件实现了Torch函数库字典形式调用。

"""

import warnings
import torch.nn as nn
from .cdcc import CDCConv2d
from .initliazer import kaiming_init, constant_init


CUSTOMIZE = ['CDCConv2d']


class ConvNormAct(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 order=('conv', 'norm', 'act')):
        super(ConvNormAct, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm = build_norm_layer(norm_cfg, norm_channels)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        self.init_weights()

    def init_weights(self):
        """initialize function"""
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        """forward"""
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x


def build_activation_layer(cfg, *args, **kwargs):
    """Build activation layer.

    Args:
        cfg (None or dict): The activation layer config, which should contain:
            - type (str): Layer type Default: ``ReLU``.
            - layer args: Args needed to instantiate a activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    if cfg is not None and not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if cfg is None:
        cfg = dict(type='ReLU')

    cfg_ = cfg.copy()
    activation_type = cfg_.pop('type')
    if activation_type in CUSTOMIZE:
        activation_layer = eval(f'{activation_type}')
    else:
        activation_layer = eval(f'nn.{activation_type}')
    layer = activation_layer(*args, **kwargs, **cfg_)

    return layer


def build_padding_layer(cfg, *args, **kwargs):
    """Build padding layer.

    Args:
        cfg (None or dict): The padding layer config, which should contain:
            - type (str): Layer type Default: ``ZeroPad2d``.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if cfg is not None and not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if cfg is None:
        cfg = dict(type='ZeroPad2d')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type in CUSTOMIZE:
        padding_layer = eval(f'{padding_type}')
    else:
        padding_layer = eval(f'nn.{padding_type}')
    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer


def build_norm_layer(cfg, *args, **kwargs):
    """Build normalization layer.

    Args:
        cfg (None or dict): The padding layer config, which should contain:
            - type (str): Layer type Default: ``BatchNorm2d``.
            - layer args: Args needed to instantiate a normalization layer.

    Returns:
        nn.Module: Created normalization layer.
    """
    if cfg is not None and not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if cfg is None:
        cfg = dict(type='BatchNorm2d')

    cfg_ = cfg.copy()
    norm_type = cfg_.pop('type')
    if norm_type in CUSTOMIZE:
        norm_layer = eval(f'{norm_type}')
    else:
        norm_layer = eval(f'nn.{norm_type}')
    layer = norm_layer(*args, **kwargs, **cfg_)

    return layer


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The padding layer config, which should contain:
            - type (str): Layer type Default: ``Conv2d``.
            - layer args: Args needed to instantiate a convolution layer.

    Returns:
        nn.Module: Created convolution layer.
    """
    if cfg is not None and not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if cfg is None:
        cfg = dict(type='Conv2d')

    cfg_ = cfg.copy()
    conv_type = cfg_.pop('type')
    if conv_type in CUSTOMIZE:
        conv_layer = eval(f'{conv_type}')
    else:
        conv_layer = eval(f'nn.{conv_type}')
    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer

