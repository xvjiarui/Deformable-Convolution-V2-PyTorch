import torch
import torch.nn as nn
from torch.nn import init
import math
import numpy as np
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

class deform_conv3d_naive(Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(deform_conv3d_naive, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
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
        in_channels = self.in_channels
        out_channels = self.out_channels
        in_d = input.size(2)
        in_h = input.size(3)
        in_w = input.size(4)
        out_d = offset.size(2)
        out_h = offset.size(3)
        out_w = offset.size(4)
        kernel_d = self.kernel_size[0]
        kernel_h = self.kernel_size[1]
        kernel_w = self.kernel_size[2]
        # [1, kernel_d * kernel_h * kernel_w, out_d, out_h, out_w, 3]
        mesh = self.compute_mesh_grid(in_d, in_h, in_w).cuda(input.get_device())
        offset = offset.view(N, self.deformable_groups, kernel_d, kernel_h, kernel_w, 3, out_d, out_h, out_w)
        # [N * dg * kernel_d * kernel_h * kernel_w, out_d, out_h, out_w, 3]
        offset = offset.permute(0, 1, 2, 3, 4, 6, 7, 8, 5).contiguous().view(N * self.deformable_groups * kernel_d * kernel_h * kernel_w, out_d, out_h, out_w, 3)
        offset_z_normalize = (offset[..., 0]) / ((in_d - 1) * 1.0 / 2)
        offset_y_normalize = (offset[..., 1]) / ((in_h - 1) * 1.0 / 2)
        offset_x_normalize = (offset[..., 2]) / ((in_w - 1) * 1.0 / 2)
        # [N * dg * kernel_d * kernel_h * kernel_w, out_d, out_h, out_w, 3]
        offset = torch.cat([offset_x_normalize[..., None], offset_y_normalize[..., None], offset_z_normalize[..., None]], dim=4)
        # [N * dg * kernel_d * kernel_h * kernel_w, out_d, out_h, out_w, 3]
        grid = mesh.expand(N * self.deformable_groups, -1, -1, -1, -1, -1).contiguous().view(-1, out_d, out_h, out_w, 3) + offset
        # [N * kernel_d * kernel_h * kernel_w * dg, in_channels/dg, in_d, in_h, in_w]
        input = input[:, None, ...].expand(-1, kernel_d * kernel_h * kernel_w, -1, -1, -1, -1).contiguous().view(
            N * kernel_d * kernel_h * kernel_w * self.deformable_groups, in_channels // self.deformable_groups,  in_d, in_h, in_w)
        sampled_feat = F.grid_sample(input, grid).view(N, kernel_d * kernel_h * kernel_w, in_channels, out_d, out_h, out_w).permute(2, 1, 0, 3, 4, 5).contiguous().view(in_channels * kernel_d * kernel_h * kernel_w, -1)
        output_feat = torch.matmul(self.weight.view(self.weight.size(0), -1), sampled_feat).view(out_channels, N, out_d, out_h, out_w).permute(1, 0, 2, 3, 4)
        return output_feat
        
    def compute_mesh_grid(self, in_d, in_h, in_w):
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        dilation_d, dilation_h, dilation_w = self.dilation
        padding_d, padding_h, padding_w = self.padding
        out_d = (in_d + 2 * padding_d - (dilation_d * (kernel_d - 1) + 1)) // stride_d + 1
        out_h = (in_h + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
        # [out_d, out_h, out_w]
        mesh_z, mesh_y, mesh_x = torch.meshgrid(torch.arange(out_d), torch.arange(out_h), torch.arange(out_w))
        mesh_z = mesh_z * stride_d - padding_d
        mesh_y = mesh_y * stride_h - padding_h
        mesh_x = mesh_x * stride_w - padding_w
        # [1, out_d, out_h, out_w]
        mesh_z = mesh_z.unsqueeze(0).float()
        mesh_y = mesh_y.unsqueeze(0).float()
        mesh_x = mesh_x.unsqueeze(0).float()
        # [kernel_d, kernel_h, kernel_w]
        kernel_offset_z, kernel_offset_y, kernel_offset_x = torch.meshgrid(torch.arange(kernel_d), torch.arange(kernel_h), torch.arange(kernel_w))
        # [kernel_d, kernel_h * kernel_w, 1, 1, 1]
        kernel_offset_z = kernel_offset_z.float().view(kernel_d * kernel_h * kernel_w, 1, 1, 1) * dilation_d
        kernel_offset_y = kernel_offset_y.float().view(kernel_d * kernel_h * kernel_w, 1, 1, 1) * dilation_h
        kernel_offset_x = kernel_offset_x.float().view(kernel_d * kernel_h * kernel_w, 1, 1, 1) * dilation_w
        # [kernel_d * kernel_h * kernel_w, out_D, out_h, out_w]
        mesh_z = mesh_z + kernel_offset_z
        mesh_y = mesh_y + kernel_offset_y
        mesh_x = mesh_x + kernel_offset_x
        mesh_z = (mesh_z - (in_d - 1) / 2.) / ((in_d - 1) / 2.)
        mesh_y = (mesh_y - (in_h - 1) / 2.) / ((in_h - 1) / 2.)
        mesh_x = (mesh_x - (in_w - 1) / 2.) / ((in_w - 1) / 2.)
        mesh = torch.cat([mesh_x[None, ..., None], mesh_y[None, ..., None], mesh_z[None, ..., None]], dim=5)
        return mesh
