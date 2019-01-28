#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair

from ..functions.sparse_conv2d_func import SparseConv2dFunction


class SparseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, num_pts=None, im2col_step=64, bias=True):
        super(SparseConv2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.num_pts = self.kernel_size[0] * self.kernel_size[1] if num_pts is None else num_pts
        self.im2col_step = im2col_step
        self.use_bias = bias
        
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels//groups, self.num_pts))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False
            self.bias.data.zero_()

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        assert 2 * self.deformable_groups * self.num_pts == \
            offset.shape[1]
        return SparseConv2dFunction.apply(input, offset,
                                          self.weight,
                                          self.bias,
                                          self.kernel_size,
                                          self.stride,
                                          self.padding,
                                          self.dilation,
                                          self.groups,
                                          self.deformable_groups,
                                          self.num_pts,
                                          self.im2col_step)


_SparseConv2d = SparseConv2dFunction.apply


class SparseConv2dPack(SparseConv2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, num_pts=None, im2col_step=64, bias=True, lr_mult=0.1):
        super(SparseConv2dPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, num_pts, im2col_step, bias)

        out_channels = self.deformable_groups * 2 * self.num_pts
        self.conv_offset = nn.Conv2d(self.in_channels,
                                     out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.conv_offset.inited = True
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        # self.conv_offset.bias.data.zero_()
        bound = (self.kernel_size[0] + self.kernel_size[1])//4
        init.uniform_(self.conv_offset.bias, -bound, bound)

    def forward(self, input):
        offset = self.conv_offset(input)
        return SparseConv2dFunction.apply(input, offset,
                                          self.weight, 
                                          self.bias,
                                          self.kernel_size,
                                          self.stride, 
                                          self.padding, 
                                          self.dilation, 
                                          self.groups,
                                          self.deformable_groups,
                                          self.num_pts,
                                          self.im2col_step)

