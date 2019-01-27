import torch
import torch.nn as nn
from torch.nn import init
import math
import numpy as np
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class deform_conv2d_naive(Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(deform_conv2d_naive, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.use_bias = bias
        
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels//groups, *self.kernel_size))
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
        N = input.size(0)
        inC = self.in_channels
        outC = self.out_channels
        inH = input.size(2)
        inW = input.size(3)
        outH = offset.size(2)
        outW = offset.size(3)
        kH = self.kernel_size[0]
        kW = self.kernel_size[1]
        # [1, kH * kW, outH, outW, 2]
        self.mesh = self.compute_mesh_grid(inH, inW).cuda(input.get_device())
        offset = offset.view(N, self.deformable_groups, kH, kW, 2, outH, outW)
        # [N * dg * kH * kW, outH, outW, 2]
        offset = offset.permute(0, 1, 2, 3, 5, 6, 4).contiguous().view(N * self.deformable_groups * kH * kW, outH, outW, 2)
        # offset_x_normalize = (offset[:, :, :, 1]) / ((outW - 1) * 1.0 / 2)
        # offset_y_normalize = (offset[:, :, :, 0]) / ((outH - 1) * 1.0 / 2)
        offset_x_normalize = (offset[:, :, :, 1]) / ((inW - 1) * 1.0 / 2)
        offset_y_normalize = (offset[:, :, :, 0]) / ((inH - 1) * 1.0 / 2)
        # [N * dg * kH * kW, outH, outW, 2]
        offset = torch.cat([offset_x_normalize[..., None], offset_y_normalize[..., None]], dim=3)
        # [N * dg * kH * kW, outH, outW, 2]
        grid = self.mesh.expand(N * self.deformable_groups, -1, -1, -1, -1).contiguous().view(-1, outH, outW, 2) + offset
        # [N * kH * kW * dg, inC/dg, inH, inW]
        input = input[:, None, ...].expand(-1, kH * kW, -1, -1, -1).contiguous().view(
            int(N * kH * kW * self.deformable_groups), int(inC / self.deformable_groups),  inH, inW)
        sampled_feat = F.grid_sample(input, grid).view(N, kH * kW, inC, outH, outW).permute(2, 1, 0, 3, 4).contiguous().view(inC * kH * kW, -1)
        output_feat = torch.matmul(self.weight.view(self.weight.size(0), -1), sampled_feat).view(outC, N, outH, outW).permute(1,0,2,3)
        return output_feat
        
    def compute_mesh_grid(self, inH, inW):
        kH = self.kernel_size[0]
        kW = self.kernel_size[1]
        outH = (inH-kH + 2 * self.padding[0])//self.stride[0] + 1
        outW = (inW-kW + 2 * self.padding[1])//self.stride[1] + 1
        # [outH, outW]
        mesh_y, mesh_x = torch.meshgrid(torch.arange(start=0, end=outH*self.stride[0], step=self.stride[0]), torch.arange(start=0, end=outW*self.stride[1], step=self.stride[1]))
        # [1, outH, outW]
        mesh_y = mesh_y.unsqueeze(0).float()
        mesh_x = mesh_x.unsqueeze(0).float()
        # [kH, kW]
        kernel_offset_y, kernel_offset_x = torch.meshgrid(torch.arange(kH), torch.arange(kW))
        # [kH * kW, 1, 1]
        # kernel_offset_y = (kernel_offset_y.float() - (kH - 1)/2.).view(kH * kW, 1, 1)
        # kernel_offset_x = (kernel_offset_x.float() - (kW - 1)/2.).view(kH * kW, 1, 1)
        kernel_offset_y = (kernel_offset_y.float() - self.padding[0]).view(kH * kW, 1, 1)
        kernel_offset_x = (kernel_offset_x.float() - self.padding[1]).view(kH * kW, 1, 1)
        # [kH * kW, outH, outW]
        mesh_y = mesh_y + kernel_offset_y
        mesh_x = mesh_x + kernel_offset_x
        # mesh_x = (mesh_x - (outW - 1)/2.) / ((outW - 1)/2.)
        # mesh_y = (mesh_y - (outH - 1)/2.) / ((outH - 1)/2.)
        mesh_y = (mesh_y - (inH - 1)/2.) / ((inH - 1)/2.)
        mesh_x = (mesh_x - (inW - 1)/2.) / ((inW - 1)/2.)
        mesh = torch.cat([mesh_x[None, ..., None], mesh_y[None, ..., None]], dim=4)
        return mesh
