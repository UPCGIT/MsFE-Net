import torch
import torch.nn as nn
# import cv2
import numpy as np
import torch.nn.functional as F
import math


class GAM_Attention(nn.Module):
    def __init__(self, c1, group=False, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1),
        )
        """
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, c1 // rate, kernel_size=3, padding=1, groups=rate)
            if group
            else nn.Conv2d(c1, int(c1 / rate), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c1, kernel_size=3, padding=1, groups=rate)
            if group
            else nn.Conv2d(int(c1 / rate), c1, kernel_size=7, padding=3),
            nn.BatchNorm2d(c1),
        )

    """
    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        """
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att"""
        return x


# GAM Attention End




class MixedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixedConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=(0, 0), groups=in_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), padding=(2, 2))
        # 深度卷积
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.point_conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.point_conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)


        self.relu = nn.ReLU(inplace=True)
        self.ga = GAM_Attention(in_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.point_conv(x1)
        x1 = self.ga(x1)
        return x1


import torchvision.ops as ops


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


class semantic(nn.Module):
    def __init__(self, num_class, out_channels):
        super(semantic, self).__init__()
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




class ours(nn.Module):
    def __init__(self, upscale_factor, L, num_class):
        super(ours, self).__init__()
        self.conv0 = MixedConv(64,64)
        self.conv3d = nn.Conv3d(1, 20, 3, 1, 1)
        self.conv1 = nn.Conv2d(L, 64, 3, 1, 1)
        self.conv11 = nn.Conv2d(num_class, 64, 3, 1, 1)
        self.conv = nn.Conv2d(512, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(64, num_class * upscale_factor ** 2, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.semantic = semantic(num_class, num_class)
        self.seman2 = semantic(num_class, num_class)

        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, num_class, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x, semanticPrior, z):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv0(x)

        out = self.conv11(semanticPrior)
        out = torch.relu(out)

        x = x + out
        x = self.dropout(x)

        x = self.conv3(x)

        x = self.pixel_shuffle(x)

        z = self.conv4(z)
        x = x + self.semantic(z)

        x = torch.softmax(x, dim=1)

        return x


class ESPCN(nn.Module):
    def __init__(self, upscale_factor, L, num_class):
        super(ESPCN, self).__init__()

        self.mixed_conv1 = MixedConv(256, 256)
        self.conv1 = nn.Conv2d(L, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.dropout = nn.Dropout(0.6)
        self.conv3 = nn.Conv2d(64, num_class * upscale_factor ** 2, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x,y,z):
        # x1= F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.relu(self.conv2(x))
        # x = self.dropout(x)
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        #  x = torch.relu(x)

        # x = torch.sigmoid(x)
        x = torch.softmax(x, dim=1)
        return x


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SCNet(nn.Module):
    def __init__(self, L, upscale_factor, num_class):
        super(SCNet, self).__init__()

        # 编码器部分
        self.encoder1 = nn.Sequential(
            CBR(L, 64),
            CBR(64, 64)
        )  # 假设输入通道为 3（例如 RGB 图像）
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            CBR(64, 64),
            CBR(64, 64)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = nn.Sequential(
            CBR(64, 64),
            CBR(64, 64),
            CBR(64, 64)
        )
        self.pool3 = nn.MaxPool2d(2)

        # 解码器部分
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = nn.Sequential(
            CBR(64, 64),
            CBR(64, 64),
            CBR(64, 64)
        )
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = nn.Sequential(
            CBR(64, 64),
            CBR(64, 64),
        )

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = nn.Sequential(
            CBR(64, 64),
            CBR(64, num_class * upscale_factor * upscale_factor),
        )

        # SPC

        self.spc = nn.PixelShuffle(upscale_factor)
        # dropout
        self.dropout = nn.Dropout(0.5)
        # segmatic
        self.encoder_LR1 = nn.Sequential(
            CBR(num_class, 64),
            CBR(64, 64)
        )  # 假设输入通道为 3（例如 RGB 图像）

    def forward(self, x, segmatic, y):
        # 编码器部分
        x = self.encoder1(x)
        segmatic = self.encoder_LR1(segmatic)
        x = x + segmatic
        x = self.pool1(x)
        segmatic = self.pool1(segmatic)
        x = self.encoder2(x)
        segmatic = self.encoder2(segmatic)
        x = x + segmatic
        x = self.pool2(x)
        segmatic = self.pool2(segmatic)
        x = self.dropout(x)
        x = self.encoder3(x)
        segmatic = self.encoder3(segmatic)
        x = x + segmatic
        x = self.pool3(x)
        x = self.dropout(x)
        # 解码器部分
        x = self.up3(x)
        x = self.decoder3(x)
        x = self.dropout(x)
        x = self.up2(x)
        x = self.decoder2(x)
        x = self.dropout(x)
        x = self.up1(x)
        x = self.decoder1(x)
        # SPC 和 Softmax
        x = self.spc(x)  # 假设 SPC 是全局池化
        output = torch.softmax(x, dim=1)
        return output


class SIM(nn.Module):
    def __init__(self, num_class):
        super(SIM, self).__init__()
        self.conv1 = nn.Conv2d(num_class, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x, sematic):
        scale = self.relu(self.conv1(sematic))
        scale = self.relu(self.conv2(scale))
        x = x * scale + scale
        return x


class SIM_ResBlock(nn.Module):
    def __init__(self, num_class):
        super(SIM_ResBlock, self).__init__()
        self.SIM = SIM(num_class)
        self.cr = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x, sematic):
        x1 = x
        x = self.SIM(x, sematic)
        x = self.cr(x)
        x = self.SIM(x, sematic)
        x = self.cr(x)
        return x + x1


class SIMNET(nn.Module):
    def __init__(self, L, num_class):
        super(SIMNET, self).__init__()
        self.cr1 = nn.Sequential(
            nn.Conv2d(L, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.cr = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.SIM = SIM(num_class)
        self.spc = nn.Sequential(
            nn.PixelShuffle(2),
            nn.ReLU()
        )
        self.cr2 = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.cr3 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, num_class, 3, 1, 1),
            nn.ReLU()
        )
        self.SIM_resblock = SIM_ResBlock(num_class)
        self.conv = nn.Conv2d(64, num_class, 3, 1, 1)

    def forward(self, x, sematic, lidar):
        x = self.cr1(x)
        for i in range(16):
            x = self.SIM_resblock(x, sematic)
        x1 = x
        x = self.SIM(x, sematic)
        x = self.cr(x)
        x = x + x1
        x = self.spc(x)
        x = self.spc(x)

        x = self.cr2(x)
        x = self.conv(x)

        return x
