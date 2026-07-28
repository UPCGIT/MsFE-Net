import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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
        # conv1_out = self.relu(conv1_out)
        conv1_out = conv1_out.squeeze(2)
        conv1_out = self.channel(conv1_out)

        # conv1_out=self.spatial(conv1_out)
        depth_conv_out = self.depth_conv(x)
        depth_conv_out = self.relu(depth_conv_out)
        depth_conv_out1 = self.point_conv(depth_conv_out)
        depth_conv_out1 = self.relu(depth_conv_out1)
        depth_conv_out = depth_conv_out + depth_conv_out1

        depth_conv_out = self.se(depth_conv_out)

        # 将不同卷积的输出进行拼接
        # out=conv1_out+depth_conv_out+point_conv_out
        out = torch.cat([conv1_out, depth_conv_out], dim=1)
        out = self.point_conv1(out)
        return self.relu(out)


class SRMCNN(nn.Module):
    def __init__(self, in_channels, num_class):
        super(SRMCNN, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        # 池化层1
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 卷积层2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # 池化层2
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 卷积层3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 卷积层4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(32, num_class, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x, y, z):
        x = self.conv1(x)
        x0 = x
        x = self.relu(x)
        x = self.pool1(x)
        x1 = x
        # 卷积层2 + 池化层2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x2 = x
        # 卷积层3
        x = self.conv3(x)
        x = self.relu(x)
        x = x2 + x
        # 卷积层4
        x = self.deconv(x)
        x = nn.functional.interpolate(x, (x1.size(2), x1.size(3)), mode="bilinear")
        x = self.conv4(x)
        x = self.relu(x)
        x = x + x1
        # 卷积层5
        x = self.deconv(x)
        x = nn.functional.interpolate(x, (x0.size(2), x0.size(3)), mode="bilinear")
        x = self.conv4(x)
        x = self.relu(x)
        x = x + x0
        # 卷积层6
        x = self.deconv(x)
        x = self.conv4(x)
        x = self.relu(x)
        # 卷积层7
        x = self.deconv1(x)
        x = self.conv7(x)
        x = self.relu(x)
        # 卷积8
        x = self.conv8(x)
        x = torch.softmax(x, dim=1)
        return x
