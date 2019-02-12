#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from dcn.gradcheck import gradcheck

# please run test file from parent folder, e.g. scp test.py .. && python ../test.py
from dcn.modules.sparse_conv2d import SparseConv2d, _SparseConv2d, SparseConv2dPack, SparseConv2dPackMore

deformable_groups = 1
N, inC, inH, inW = 2, 4, 4, 4
outC = 4
kH, kW = 3, 3
stride = 1
padding = 1
groups = 1
dilation = 1
im2col_step = 1
num_pts = kH * kW


torch.manual_seed(3)


def check_sconv_im2col_step_forward():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * num_pts,
                            kernel_size=(kH, kW),
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            bias=True).cuda()

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)

    scn1 = SparseConv2d(inC, outC, (kH, kW),
                        stride=stride, padding=padding, dilation=dilation,
                        groups=groups,
                        deformable_groups=deformable_groups,
                        num_pts=num_pts,
                        im2col_step=1).cuda()

    scn2 = SparseConv2d(inC, outC, (kH, kW),
                        stride=stride, padding=padding, dilation=dilation,
                        groups=groups,
                        deformable_groups=deformable_groups,
                        num_pts=num_pts,
                        im2col_step=2).cuda()
    scn1.weight = scn2.weight
    scn1.bias = scn2.bias
    output1 = scn1(input, offset)
    output2 = scn2(input, offset)

    d = (output1 - output2).abs().max()
    if d < 1e-10:
        print('sconv im2col_step forward successfully passed with {}'.format(d))
    else:
        print('sconv im2col_step forward failed with {}'.format(d))
        print(output1)
        print(output2)
        print((output1 - output2).abs())


def check_sconv_im2col_step_backward():
    input = torch.rand(N, inC, inH, inW).cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * num_pts, inH, inW).cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    weight = torch.randn(outC, int(inC//groups), num_pts).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).cuda()
    bias.requires_grad = True

    output1 = _SparseConv2d(input, offset, weight, bias, (kH, kW), stride, padding, dilation, groups, deformable_groups, num_pts, 2)
    targert = torch.rand(*output1.size()).cuda()
    error = (targert - output1).mean()
    error.backward(retain_graph=True)
    input_grad = input.grad.clone()
    offset_grad = offset.grad.clone()
    weight_grad = weight.grad.clone()
    bias_grad = bias.grad.clone()
    output2 = _SparseConv2d(input, offset, weight, bias, (kH, kW), stride, padding, dilation, groups, deformable_groups, num_pts, 1)
    error2 = (targert - output2).mean()
    error2.backward()
    print((output1 - output2).abs().max())
    input_grad_err = (input.grad - 2 * input_grad).abs().max() 
    offset_grad_err = (offset.grad - 2 * offset_grad).abs().max()
    weight_grad_err = (weight.grad - 2 * weight_grad).abs().max()
    bias_grad_err = (bias.grad - 2 * bias_grad).abs().max()
    grad_err = input_grad_err + offset_grad_err + weight_grad_err + bias_grad_err
    if grad_err:
        print("sconv im2col_step backward successfully passed with {} = {}+{}+{}+{}".format(grad_err, input_grad_err, offset_grad_err, weight_grad_err, bias_grad_err))
    else:
        print("sconv im2col_step backward failed with {} = {}+{}+{}+{}".format(grad_err, input_grad_err, offset_grad_err, weight_grad_err, bias_grad_err))


def check_gradient_sconv():

    input = torch.rand(N, inC, inH, inW).double().cuda()
    print('max input:', input.max())
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * num_pts, inH, inW).double().cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    weight = torch.randn(outC, int(inC//groups), num_pts).double().cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).double().cuda()
    bias.requires_grad = True

    # print('check_gradient_dconv: ',
    #       gradcheck(_DeformConv, (input, offset, weight, bias,
    #                 stride, padding, dilation, groups, deformable_groups, im2col_step),
    #                 eps=1e-3, atol=1e-3, rtol=1e-2, raise_exception=True))
    print('check_gradient_sconv: ',
          gradcheck(_SparseConv2d, (input, offset, weight, bias, (kH, kW),
                    stride, padding, dilation, groups, deformable_groups, num_pts, im2col_step)))

def example_sconv():
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    scn = SparseConv2dPack(64, 128, kernel_size=(3, 3), stride=1,
              padding=1, groups=2, deformable_groups=2).cuda()
    # print(scn.weight.shape, input.shape)
    output = scn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print(output.shape)

def example_sconv_more():
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    scn = SparseConv2dPackMore(64, 128, kernel_size=(3, 3), stride=1,
                           padding=1, groups=2, deformable_groups=2).cuda()
    # print(scn.weight.shape, input.shape)
    output = scn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print(output.shape)


if __name__ == '__main__':

    example_sconv()
    example_sconv_more()

    for _num_pts in [4, 9, 25]:
        num_pts = _num_pts
        print('checking num_pts: {}'.format(num_pts if num_pts is not None else "None"))
        check_sconv_im2col_step_forward()
        check_sconv_im2col_step_backward()

        check_gradient_sconv()
    # """
    # ****** Note: backward is not reentrant error may not be a serious problem,
    # ****** since the max error is less than 1e-7,
    # ****** Still looking for what trigger this problem
    # """
