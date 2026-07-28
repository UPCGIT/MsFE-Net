import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.ops as ops
"""Multiscale Semantically Modulated Mixed Convolutional Networks for Subpixel Mapping"""


class DeformConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 权重参数
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size
        ))

        # 偏移量预测分支
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,  # x和y方向偏移
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # 初始化权重
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        # 预测偏移量
        offset = self.offset_conv(x)

        # 应用可变形卷积
        x = ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            stride=self.stride,
            padding=self.padding
        )

        return x


class semanticlidar(nn.Module):
    def __init__(self, num_class, out_channels):
        super(semanticlidar, self).__init__()
        self.semantic_conv1 = nn.Conv2d(num_class, out_channels, 1, 1, 0)
        self.semantic_conv2 = nn.Conv2d(num_class, out_channels, 3, 1, 1)
        self.semantic_conv3 = nn.Conv2d(num_class, out_channels, 5, 1, 2)
        self.bian = DeformConv(out_channels, out_channels)

        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, semanticPrior):
        semanticPrior5x5 = self.semantic_conv3(semanticPrior)
        semanticPrior5x5 = torch.tanh(semanticPrior5x5)

        semanticPrior3x3 = self.semantic_conv2(semanticPrior)
        semanticPrior3x3 = torch.tanh(semanticPrior3x3)

        semanticPrior_1 = self.semantic_conv1(semanticPrior)
        semanticPrior_1 = torch.tanh(semanticPrior_1)

        semanticPrior = self.bian(semanticPrior)
        semanticPrior = torch.tanh(semanticPrior)

        out = semanticPrior3x3+semanticPrior_1 + semanticPrior5x5
        out = self.bian(out)
        # out = self.conv(out)
        # out = semanticPrior_1 + semanticPrior3x3
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x + x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.BN = nn.BatchNorm2d(1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 1*H*W
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 1*H*W
        out = torch.cat([avg_out, max_out], dim=1)  # 2*H*W
        out = self.conv1(out)  # 1*H*W
        out = self.BN(out)  # 1*H*W
        return x + self.activation(out) * x  # NC*H*W


class Selayer(nn.Module):
    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out


class MixedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixedConv, self).__init__()
        # 3D卷积
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # 深度卷积
        self.depth_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels,
                                    bias=False)
        # 点卷积
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.point_conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0)
        self.spatial = SpatialAttention()
        self.channel = ChannelAttention(out_channels)
        self.se = Selayer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = x.unsqueeze(2)
        conv1_out = self.conv(x1)
        # conv1_out = self.conv(conv1_out)
        #  conv1_out = torch.tanh(conv1_out)
        conv1_out = conv1_out.squeeze(2)
        conv1_out = self.channel(conv1_out)
        conv1_out = self.spatial(conv1_out)
        # conv1_out=self.spatial(conv1_out)
        depth_conv_out = self.depth_conv(x)
        depth_conv_out = self.relu(depth_conv_out)
        depth_conv_out1 = self.point_conv(depth_conv_out)
        depth_conv_out1 = self.relu(depth_conv_out1)
        depth_conv_out = depth_conv_out + depth_conv_out1

        depth_conv_out = depth_conv_out + self.se(depth_conv_out)

        # 将不同卷积的输出进行拼接
        # out=conv1_out+depth_conv_out+point_conv_out
        out = torch.cat([conv1_out, depth_conv_out], dim=1)
        out = self.point_conv1(out)
        return self.relu(out)


class semantic(nn.Module):
    def __init__(self, num_class, out_channels):
        super(semantic, self).__init__()
        self.semantic_conv1 = nn.Conv2d(num_class, out_channels, 3, 1, 1)
        self.semantic_conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.point_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.point_conv1 = nn.Conv2d(num_class, out_channels, kernel_size=1, padding=0)
        self.point_conv2 = nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)
        self.semantic_conv3 = nn.Conv2d(num_class, out_channels, 5, 1, 2)
        self.semantic_conv4 = nn.Conv2d(out_channels, out_channels, 5, 1, 2)
        self.spatal = SpatialAttention()
        self.conv = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, semanticPrior):
        '''# 5×5********************************************************************************************************************'''
        semanticPrior5x5 = self.semantic_conv3(semanticPrior)
        semanticPrior5x5 = torch.tanh(semanticPrior5x5)
        # semanticPrior_1=semanticPrior3x3
        '''#  semantic1********************************************************************************************************************'''

        semanticPrior5x5 = self.point_conv(semanticPrior5x5)
        semanticPrior5x5 = torch.tanh(semanticPrior5x5)
        '''# 3×3********************************************************************************************************************'''
        semanticPrior3x3 = self.semantic_conv1(semanticPrior)
        semanticPrior3x3 = torch.tanh(semanticPrior3x3)
        # semanticPrior_1=semanticPrior3x3
        '''#  semantic1********************************************************************************************************************'''

        semanticPrior3x3 = self.point_conv(semanticPrior3x3)
        semanticPrior3x3 = torch.tanh(semanticPrior3x3)

        '''#  semantic2********************************************************************************************************************'''
        semanticPrior_1 = self.point_conv1(semanticPrior)
        semanticPrior_1 = torch.tanh((semanticPrior_1))
        semanticPrior_1 = self.point_conv(semanticPrior_1)
        semanticPrior_1 = torch.tanh(semanticPrior_1)
        # semanticPrior_1 = self.semantic_conv2(semanticPrior_1)
        # semanticPrior_1 = torch.tanh(semanticPrior_1)
        out = torch.cat([semanticPrior_1, semanticPrior3x3, semanticPrior5x5], dim=1)  # 拼接通道维度
        out = self.conv(out)
        # out = semanticPrior_1 + semanticPrior3x3
        return out


class MSMC(nn.Module):
    def __init__(self, upscale_factor, L, num_class):
        super(MSMC, self).__init__()

        self.mixed_conv1 = MixedConv(256, 256)
        self.conv1 = nn.Conv2d(L, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.se1 = Selayer(384)
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.se2 = Selayer(256)
        self.dropout = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, num_class * upscale_factor ** 2, 3, 1, 1)
        self.se3 = Selayer(num_class * upscale_factor ** 2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        '''#  semantic********************************************************************************************************************'''
        self.semantic = semantic(num_class, 256)

        self.lidar = semanticlidar(num_class, num_class)
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, num_class, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x, semanticPrior, z):
        x = self.conv1(x)
        out = self.semantic(semanticPrior)
        x = x + out
        for i in range(7):
            x = x + self.mixed_conv1(x)

        x = x + out
        x = self.dropout(x)
        x1 = x
        for i in range(1):
            x2 = x
            x = torch.tanh(self.conv2(x))
            x = x2 + torch.tanh(self.conv2(x))
        x = x + out
        x = x + x1
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        x1 = self.conv3(out)
        x1 = self.pixel_shuffle(x1)


        x = x + x1
        x = torch.softmax(x, dim=1)
        return x