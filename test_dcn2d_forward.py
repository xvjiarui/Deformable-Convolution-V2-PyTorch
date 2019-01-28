#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn

from dcn.modules.deform_conv2d import DeformConv2d, _DeformConv2d, DeformConv2dPack
from dcn.deform_conv2d_naive import deform_conv2d_naive

deformable_groups = 1
N, inC, inH, inW = 1, 4, 16, 16
outC = 2
kH, kW = 3, 3
stride = 1
groups = 1
dilation = 1
padding = 1

torch.manual_seed(3)


def check_dconv_zero_offset():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            bias=True).cuda()

    dcn = DeformConv2d(inC, outC, (kH, kW),
                   stride=stride, padding=padding, dilation=dilation,
                   groups=groups,
                   deformable_groups=deformable_groups, im2col_step=1).cuda()
    pcn = nn.Conv2d(inC, outC, (kH, kW), stride=stride, padding=padding, dilation=dilation, groups=groups).cuda()
    pcn.weight = dcn.weight
    pcn.bias = dcn.bias
    print((pcn.weight.data - dcn.weight.data).abs().max())

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    # conv_identify(dcn.weight, dcn.bias)

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    output_d = dcn(input, offset)
    output_p = pcn(input)
    d = (output_d - output_p).abs().max()
    if d < 1e-5:
        print('dconv zero offset passed with {}'.format(d))
    else:
        print('dconv zero offset failed with {}'.format(d))
        # print(output_p)
        # print(output_d)
        print((output_d - output_p).abs())


def check_dconv_naive_zero_offset():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            bias=True).cuda()

    dcn = deform_conv2d_naive(inC, outC, (kH, kW), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False).cuda()
    pcn = nn.Conv2d(inC, outC, (kH, kW), stride=stride, padding=padding,
                    dilation=dilation, groups=groups, bias=False).cuda()
    pcn.weight = dcn.weight
    # pcn.bias = dcn.bias
    print((pcn.weight.data - dcn.weight.data).abs().max())

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    # conv_identify(dcn.weight, dcn.bias)

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    output_d = dcn(input, offset)
    output_p = pcn(input)
    d = (output_d - output_p).abs().max()
    if d < 1e-5:
        print('dconv zero offset passed with {}'.format(d))
    else:
        print('dconv zero offset failed with {}'.format(d))
        # print(output_p)
        # print(output_d)
        print((output_d - output_p).abs())


def check_forward_dconv():
    conv_offset = nn.Conv2d(inC, 1 * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            bias=True).cuda()
    # conv_offset.weight.data.zero_()
    # conv_offset.bias.data.zero_()

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    dcn = DeformConv2d(inC, outC, (kH, kW),
        stride=stride, padding=padding, dilation=dilation,
        groups=groups, bias=False).cuda()

    dcnn = deform_conv2d_naive(inC, outC, (kH, kW), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False).cuda()
    dcnn.weight = dcn.weight

    output1 = dcn(input, offset)
    output2 = dcnn(input, offset)

    d = (output1 - output2).abs().max()
    if d < 1e-5:
        print('dconv naive forward passed with {}'.format(d))
    else:
        print('dconv naive forward failed with {}'.format(d))
        print(output1)
        print(output2)
        print((output1 - output2).abs())


if __name__ == '__main__':

    check_dconv_naive_zero_offset()
    check_forward_dconv()
    kernel_size_list = [1, 3, 5, 7]
    stride_list = [1, 2]
    padding_list = [0, 1, 2]
    dilation_list = [1, 2]
    for kernel_size in kernel_size_list:
        print('kernel: {}'.format(kernel_size))
        kH = kernel_size
        kW = kernel_size
        for stride_size in stride_list:
            print('stride: {}'.format(stride_size))
            stride = stride_size
            for padding_size in padding_list:
                print('padding: {}'.format(padding_size))
                padding = padding_size
                for dilation_size in dilation_list:
                    print('dilation: {}'.format(dilation_size))
                    dilation = dilation_size
                    check_dconv_naive_zero_offset()
                    check_forward_dconv()

    # """
    # ****** Note: backward is not reentrant error may not be a serious problem,
    # ****** since the max error is less than 1e-7,
    # ****** Still looking for what trigger this problem
    # """
