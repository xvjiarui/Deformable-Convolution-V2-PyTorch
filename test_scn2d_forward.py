#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn

from dcn.modules.sparse_conv2d import SparseConv2d, _SparseConv2d, SparseConv2dPack
from dcn.sparse_conv2d_naive import sparse_conv2d_naive

deformable_groups = 1
N, inC, inH, inW = 1, 4, 16, 16
outC = 2
kH, kW = 3, 3
stride = 1
groups = 1
dilation = 1
padding = 1
num_pts = 9

torch.manual_seed(3)


def check_forward_sconv():
    conv_offset = nn.Conv2d(inC, 1 * 2 * num_pts,
                            kernel_size=(kH, kW),
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            bias=True).cuda()
    # conv_offset.weight.data.zero_()
    # conv_offset.bias.data.zero_()

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    scn = SparseConv2d(inC, outC, (kH, kW),
        stride=stride, padding=padding, dilation=dilation,
        groups=groups, num_pts=num_pts, bias=False).cuda()

    scnn = sparse_conv2d_naive(inC, outC, (kH, kW), stride=stride, padding=padding, dilation=dilation, groups=groups, num_pts=num_pts, bias=False).cuda()
    scnn.weight = scn.weight

    output1 = scn(input, offset)
    output2 = scnn(input, offset)

    d = (output1 - output2).abs().max()
    if d < 1e-5:
        print('sconv naive forward passed with {}'.format(d))
    else:
        print('sconv naive forward failed with {}'.format(d))
        print(output1)
        print(output2)
        print((output1 - output2).abs())


if __name__ == '__main__':

    check_forward_sconv()
    kernel_size_list = [1, 3, 5, 7]
    stride_list = [1, 2]
    padding_list = [0, 1, 2]
    dilation_list = [1, 2]
    for kernel_size in kernel_size_list:
        print('kernel: {}'.format(kernel_size))
        kH = kernel_size
        kW = kernel_size
        for num in range(kW, kW*kH, kW):
            print('num_pts: {}'.format(num))
            num_pts = num
            for stride_size in stride_list:
                print('stride: {}'.format(stride_size))
                stride = stride_size
                for padding_size in padding_list:
                    print('padding: {}'.format(padding_size))
                    padding = padding_size
                    for dilation_size in dilation_list:
                        print('dilation: {}'.format(dilation_size))
                        dilation = dilation_size
                        check_forward_sconv()

    # """
    # ****** Note: backward is not reentrant error may not be a serious problem,
    # ****** since the max error is less than 1e-7,
    # ****** Still looking for what trigger this problem
    # """
