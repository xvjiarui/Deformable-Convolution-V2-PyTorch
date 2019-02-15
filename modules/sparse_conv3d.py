#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _triple
import numpy as np

from ..functions.sparse_conv3d_func import SparseConv3dFunction

class SparseConv3d(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, num_pts=None, im2col_step=64, bias=True):
        super(SparseConv3d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.num_pts = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] if num_pts is None else num_pts
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
        assert 3 * self.deformable_groups * self.num_pts == \
            offset.shape[1], 'offset chanel num incorrect: 3x{}x{}={}'.format(self.deformable_groups,
              self.num_pts, offset.shape[1])
        return SparseConv3dFunction.apply(input, offset,
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

_SparseConv3d = SparseConv3dFunction.apply

class SparseConv3dPack(SparseConv3d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, num_pts=None, im2col_step=64, bias=True, lr_mult=0.1, dim_mask=(1, 1, 1), offset_kernel_size=None):
        super(SparseConv3dPack, self).__init__(in_channels, out_channels,
                                               kernel_size, stride, padding, dilation, groups, deformable_groups, num_pts, im2col_step, bias)

        out_channels = self.deformable_groups * 3 * self.num_pts
        if offset_kernel_size is None: 
            offset_kernel_size = self.kernel_size
            offset_stride = self.stride
            offset_padding = self.padding
        else:
            offset_stride = (1, 1, 1)
            offset_padding = ((offset_kernel_size[0]-1)//2, (offset_kernel_size[1]-1)//2, (offset_kernel_size[2]-1)//2)
        self.conv_offset = nn.Conv3d(self.in_channels,
                                          out_channels,
                                          kernel_size=offset_kernel_size,
                                          stride=offset_stride,
                                          padding=offset_padding,
                                          bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.conv_offset.inited = True
        self.init_offset()
        self.dim_mask = np.tile(np.repeat(np.array(dim_mask), self.num_pts), self.deformable_groups)
        self.dim_mask = torch.from_numpy(self.dim_mask).view(1, out_channels, 1, 1, 1).float()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        # self.conv_offset.bias.data.zero_()
        bound = (self.kernel_size[0] + self.kernel_size[1] + self.kernel_size[2])//6
        init.uniform_(self.conv_offset.bias, -bound, bound)

    def forward(self, input):
        offset = self.conv_offset(input)
        offset = offset * self.dim_mask.cuda(input.get_device(), non_blocking=True)
        return SparseConv3dFunction.apply(input, offset,
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


class SparseConv3dPackMore(SparseConv3d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, num_pts=None, im2col_step=64, bias=True, lr_mult=0.1, dim_mask=(1, 1, 1), offset_kernel_size=None):
        super(SparseConv3dPackMore, self).__init__(in_channels, out_channels,
                                               kernel_size, stride, padding, dilation, groups, deformable_groups, num_pts, im2col_step, bias)

        out_channels = self.deformable_groups * 3 * self.num_pts
        if offset_kernel_size is None:
            offset_kernel_size = self.kernel_size
            offset_stride = self.stride
            offset_padding = self.padding
        else:
            offset_stride = (1, 1, 1)
            offset_padding = ((offset_kernel_size[0]-1)//2, (offset_kernel_size[1]-1)//2, (offset_kernel_size[2]-1)//2)
        self.conv_offset = nn.Sequential(
            nn.Conv3d(self.in_channels, self.in_channels//4,
                      kernel_size=(offset_kernel_size[0], 1, 1),
                      stride=(offset_stride[0], 1, 1),
                      padding=(offset_padding[0], 0, 0),
                      bias=False),
            nn.BatchNorm3d(self.in_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.in_channels//4, out_channels,
                      kernel_size=(1, *offset_kernel_size[1:]),
                      stride=(1, *offset_stride[1:]),
                      padding=(0, *offset_padding[1:]),
                      bias=True)
        )
        self.conv_offset[-1].lr_mult = lr_mult
        self.conv_offset[-1].inited = True
        self.init_offset()
        self.dim_mask = np.tile(np.repeat(np.array(dim_mask), self.num_pts), self.deformable_groups)
        self.dim_mask = torch.from_numpy(self.dim_mask).view(1, out_channels, 1, 1, 1).float()

    def init_offset(self):
        self.conv_offset[-1].weight.data.zero_()
        # self.conv_offset[-1].bias.data.zero_()
        bound = (self.kernel_size[0] + self.kernel_size[1] + self.kernel_size[2])//6
        init.uniform_(self.conv_offset[-1].bias, -bound, bound)

    def forward(self, input):
        offset = self.conv_offset(input)
        offset = offset * self.dim_mask.cuda(input.get_device(), non_blocking=True)
        return SparseConv3dFunction.apply(input, offset,
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
