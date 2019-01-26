#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import DCN

class SparseConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias, kernel_size,
                stride, padding, dilation, group, deformable_groups, num_pts, im2col_step):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(kernel_size)
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.num_pts = num_pts
        ctx.im2col_step = im2col_step
        output = DCN.sparse_conv2d_forward(input, weight, bias,
                                           offset,
                                           ctx.kernel_size[0], ctx.kernel_size[1],
                                           ctx.stride[0], ctx.stride[1],
                                           ctx.padding[0], ctx.padding[1],
                                           ctx.dilation[0], ctx.dilation[1],
                                           ctx.group,
                                           ctx.deformable_groups,
                                           ctx.num_pts,
                                           ctx.im2col_step)
        ctx.save_for_backward(input, offset, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_weight, grad_bias = \
            DCN.sparse_conv2d_backward(input, weight,
                                       bias,
                                       offset,
                                       grad_output,
                                       ctx.kernel_size[0], ctx.kernel_size[1],
                                       ctx.stride[0], ctx.stride[1],
                                       ctx.padding[0], ctx.padding[1],
                                       ctx.dilation[0], ctx.dilation[1],
                                       ctx.group,
                                       ctx.deformable_groups,
                                       ctx.num_pts,
                                       ctx.im2col_step)

        return grad_input, grad_offset, grad_weight, grad_bias,\
            None, None, None, None, None, None, None, None
