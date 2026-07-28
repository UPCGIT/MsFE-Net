import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
#from torchvision import models
# import fjn_util
import os
from modle1 import DeformConv, semantic
import warnings

warnings.filterwarnings('ignore')


class Default_Conv(nn.Module):
    def __init__(self, ch_in, ch_out, k_size=(3, 3), stride=1, padding=(1, 1), bias=False, groups=1):
        super(Default_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, stride=stride,
                              padding=padding, bias=bias, groups=groups)

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, ch_in, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_in // ratio, ch_in, 1, bias=False)
        )
        self.BN = nn.BatchNorm2d(num_features=ch_in)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # NC*1*1
        max_out = self.fc(self.max_pool(x))  # NC*1*1
        out = avg_out + max_out
        return x + self.activation(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.BN = nn.BatchNorm2d(1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 1*H*W
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 1*H*W
        out = torch.cat([avg_out, max_out], dim=1)  # 2*H*W
        out = self.conv1(out)  # 1*H*W
        out = self.BN(out)  # 1*H*W
        return x + self.activation(out) * x  # NC*H*W


class involution(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),  # (in_channels, out_channels, kernel_size)
            nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU()
        )
        self.conv2 = nn.Conv2d(
            in_channels=channels // reduction_ratio,  # in_channels
            out_channels=kernel_size ** 2 * self.groups,  # out_channels
            kernel_size=1,  # ,#kernel size
            stride=1,
        )
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out


class SemanticProcessPA(nn.Module):
    def __init__(self, ch_in, ch_out, ):
        super(SemanticProcessPA, self).__init__()
        self.PA_branch = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=ch_out),
            nn.ReLU())

    def forward(self, x):
        p_a = self.PA_branch(x)
        return p_a


class SemanticProcessPS(nn.Module):
    def __init__(self, ch_in, ch_out, ):
        super(SemanticProcessPS, self).__init__()
        self.PS_branch = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=ch_out),
            nn.ReLU())

    def forward(self, x):
        p_s = self.PS_branch(x)
        return p_s




# the whole network architecture
class SpectralFeedbackNet(nn.Module):
    def __init__(self, ch_in, ch_out, class_num, upscale_factor, semantic_up_scale):
        super(SpectralFeedbackNet, self).__init__()
        # the first conv block
        self.input_conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=ch_out),
            nn.ReLU(),
        )

        self.activation = nn.ReLU()

        '''# Spatial feedback block ***************************************************************************************************'''
        # Spatial feedback M1
        self.feedback = Spatial_Feedback_Block(class_num=class_num, ch_out=ch_out)
        self.involution = involution(channels=ch_out, kernel_size=3, stride=1)
        self.afe_out = AFE(ch_out=ch_out)
        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)

        # Spatial Feedback M2
        self.feedback1 = Spatial_Feedback_Block(class_num=class_num, ch_out=ch_out)
        self.involution1 = involution(channels=ch_out, kernel_size=3, stride=1)
        self.afe_out1 = AFE(ch_out=ch_out)
        self.down_sample1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Spatial Feedback M3
        self.feedback2 = Spatial_Feedback_Block(class_num=class_num, ch_out=ch_out)
        self.involution2 = involution(channels=ch_out, kernel_size=3, stride=1)
        self.afe_out2 = AFE(ch_out=ch_out)
        self.down_sample2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Spatial Feedback M4
        self.feedback3 = Spatial_Feedback_Block(class_num=class_num, ch_out=ch_out)
        self.involution3 = involution(channels=ch_out, kernel_size=3, stride=1)
        self.afe_out3 = AFE(ch_out=ch_out)

        '''#  multi-level fusion ********************************************************************************************************************'''
        self.fusion = CrossAttention(feature_dim=ch_out)

        self.fusion1 = CrossAttention(feature_dim=ch_out)

        self.fusion2 = CrossAttention(feature_dim=ch_out)

        self.sum_involution = involution(channels=ch_out, kernel_size=3, stride=1)
        self.sum_out_conv = Default_Conv(ch_in=ch_out, ch_out=ch_out, k_size=1, stride=1, padding=0)

        '''# fused feature SPC processed into the highest spatial resolution space***************************************************************************************************'''
        #  semantic prior global modulation
        upscale1 = []
        for _ in range(int(math.log2(semantic_up_scale))):
            upscale1.append(PixelShuffleBlock(in_channel=class_num, out_channel=class_num, upscale_factor=2))
        self.semantic_up = nn.Sequential(*upscale1)

        self.semantic_up_conv1 = Default_Conv(ch_in=class_num, ch_out=ch_out, k_size=3, stride=1, padding=1)
        self.semantic_up_conv2 = Default_Conv(ch_in=class_num, ch_out=ch_out, k_size=3, stride=1, padding=1)
        self.semantic_up_activation = nn.ReLU()

        # modulation
        self.semantic_up_sum_conv1 = Default_Conv(ch_in=ch_out, ch_out=ch_out, k_size=3, stride=1, padding=1)
        self.semantic_up_sum_conv2 = Default_Conv(ch_in=ch_out, ch_out=ch_out, k_size=1, stride=1, padding=0)
        self.semantic_up_sum_activation = nn.ReLU()
        self.global_involution = involution(channels=ch_out, kernel_size=3, stride=1)

        '''# main up************************************************************************************************************************************'''
        upscale = []
        for _ in range(int(math.log2(upscale_factor))):
            upscale.append(PixelShuffleBlock(in_channel=ch_out, out_channel=ch_out, upscale_factor=2))
        self.upscale_layers = nn.Sequential(*upscale)

        '''# classify**********************************************************************************************************************************************'''
        self.classify_layers = ClassifierBlock(ch_out=ch_out, num_class=class_num)

        self.semantic = semantic(class_num, class_num)
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, class_num, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x, semanticPrior, y):
        block_input = self.input_conv(x)  # the original conv block
        # h x w x ch_out

        '''#  feedback****************************************************************************************************************************************'''
        # M1
        out1 = self.feedback(block_input, semanticPrior)
        out1 = self.involution(out1)
        out1_1 = self.afe_out(out1)  # 2h x 2w x ch_out

        feedback_input_add1 = self.down_sample(out1)
        block_input1 = feedback_input_add1 + block_input  # h x w x ch_out

        # M2
        out2 = self.feedback1(block_input1, semanticPrior)
        out2 = self.involution1(out2)
        out2_1 = self.afe_out1(out2)  # 2h x 2w x ch_out

        feedback_input_add2 = self.down_sample1(out2)
        block_input2 = feedback_input_add2 + block_input1  # h x w x ch_out

        # M3
        out3 = self.feedback2(block_input2, semanticPrior)
        out3 = self.involution2(out3)
        out3_1 = self.afe_out2(out3)  # 2h x 2w x ch_out

        feedback_input_add3 = self.down_sample2(out3)
        block_input3 = feedback_input_add3 + block_input2  # h x w x ch_out

        # M4
        out4 = self.feedback3(block_input3, semanticPrior)
        out4 = self.involution3(out4)
        out4_1 = self.afe_out3(out4)  # 2h x 2w x ch_out

        '''#  Spatial feedback block >>> multi-level fusion**************************************************************************************'''
        out_sum1 = self.fusion(out1_1, out2_1)  # 2h x 2w x ch_out
        out_sum2 = self.fusion1(out_sum1, out3_1)  # 2h x 2w x ch_out
        out_sum = self.fusion2(out_sum2, out4_1)  # 2h x 2w x ch_out
        out_sum = self.sum_involution(out_sum)  # 2h x 2w x ch_out
        up_out = self.activation(self.sum_out_conv(out_sum))  # 2h x 2w x ch_out
        final_up_out = self.upscale_layers(up_out)

        '''# global semantic modulation*******************************************************************************************************************'''
        semantic_up_global = self.semantic_up(semanticPrior)  # up 4h x 4w x class_num
        semantic_up_a = self.semantic_up_activation(self.semantic_up_conv1(semantic_up_global))  # 4h x 4w x ch_out
        semantic_up_b = self.semantic_up_activation(self.semantic_up_conv2(semantic_up_global))  # 4h x 4w x ch_out


        semantic_main_sum = final_up_out * semantic_up_a + semantic_up_b + final_up_out  # 4h x 4w x ch_out
        main_out = self.semantic_up_sum_activation(
            self.semantic_up_sum_conv1(semantic_main_sum))  # 4h x 4w x ch_out
        main_out = self.semantic_up_sum_activation(self.semantic_up_sum_conv2(main_out))  # 4h x 4w x ch_out

        main_out = self.global_involution(main_out) # 4h x 4w x ch_out

        ''' # classify********************************************************************************************************************************************'''
        classify_out = self.classify_layers(main_out)
        # z = self.conv4(y)
        classify_out = classify_out
        return classify_out


def init_weights_normal(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        m.bias.data.fill_(0.01)


class Spatial_Feedback_Block(nn.Module):
    def __init__(self, class_num, ch_out, bias=False, activation=nn.ReLU()):
        super(Spatial_Feedback_Block, self).__init__()

        # self.attention = AttentionBlock(ch_out=ch_out)
        self.semantic = semanticBlock(num_class=class_num, ch_out=ch_out)
        self.spc = PixelShuffleBlock(in_channel=ch_out, out_channel=ch_out, upscale_factor=2)

        self.up_semantic = semanticBlock(num_class=class_num, ch_out=ch_out)
        self.semantic_spc = PixelShuffleBlock(in_channel=class_num, out_channel=class_num, upscale_factor=2)

    def forward(self, x, semanticPrior):
        # MR guide LR
        out = self.semantic(x, semanticPrior) + x
        out_up = self.spc(out)

        up_semantic = self.semantic_spc(semanticPrior)
        out = self.up_semantic(out_up, up_semantic) + out_up

        return out


class semanticBlock(nn.Module):
    def __init__(self, num_class, ch_out, bias=False, ):
        super(semanticBlock, self).__init__()
        # f(F) semantic prior
        self.semanticPA = SemanticProcessPA(ch_in=num_class, ch_out=ch_out)
        self.semanticPS = SemanticProcessPS(ch_in=num_class, ch_out=ch_out)
        self.conv = Default_Conv(ch_in=ch_out, ch_out=ch_out, k_size=3, stride=1, padding=1, bias=bias)

        self.activation = nn.ReLU()
        self.BatchNorm = nn.BatchNorm2d(num_features=ch_out)


    def forward(self, x, semanticPrior):
        input_processed = self.semanticPA(semanticPrior) * x + self.semanticPS(semanticPrior)
        out = self.activation(self.BatchNorm(self.conv(input_processed)))

        return out


class Conv1x1BNReLU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv1x1BNReLU, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1x1(x)))


class AFE(nn.Module):
    def __init__(self, ch_out):
        super(AFE, self).__init__()
        #  left
        self.sa1 = SpatialAttention()
        self.ca1 = ChannelAttention(ch_in=ch_out)
        # self.sigmoid = nn.Sigmoid()
        # right
        self.right_branch = nn.Sequential(
            Conv1x1BNReLU(ch_out, ch_out),
            Conv1x1BNReLU(ch_out, ch_out)
        )
        # mid
        self.mid_branch = nn.Sequential(
            Conv1x1BNReLU(ch_out, ch_out),
            Conv1x1BNReLU(ch_out, ch_out),
            nn.Sigmoid()
        )

        self.out_block = Conv1x1BNReLU(ch_out, ch_out)

    def forward(self, x):
        ca1 = self.ca1(x)
        left_out = self.sa1(ca1)
        right_out = self.right_branch(x)
        mid_sum = left_out + right_out
        mid_out = self.mid_branch(mid_sum)
        out = (mid_out + 1) * right_out
        out = self.out_block(out)
        return out  # NC*H*W


class CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, y):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # reshape x to (b, h*w, c)
        y = y.reshape(b, c, h * w).permute(0, 2, 1)  # reshape y to (b, h*w, c)

        query = self.query(x)  # (b, h*w, c)
        key = self.key(y)  # (b, h*w, c)
        value = self.value(y)  # (b, h*w, c)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        #  (b, h*w, h*w)
        attention = torch.softmax(scores, dim=-1)  # (b, h*w, h*w)
        out = attention @ value  # (b, h*w, c)
        out = self.proj(out)  # (b, h*w, c)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)  # reshape out back to original shape
        return out


# PixelShuffer
class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upscale_factor, kernel=3, stride=1, padding=1):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel * upscale_factor ** 2, kernel, stride, padding)
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        out = self.ps(self.conv(x))
        return out


# Softmax
class ClassifierBlock(nn.Module):
    def __init__(self, ch_out, num_class, bias=False, activation=nn.ReLU()):
        super(ClassifierBlock, self).__init__()
        self.conv1 = Default_Conv(ch_in=ch_out, ch_out=num_class, k_size=1, stride=1, padding=0, bias=bias)
        self.conv2 = Default_Conv(ch_in=num_class, ch_out=num_class, k_size=3, stride=1, padding=1, bias=bias)
        self.activate = activation
        self.BatchNorm = nn.BatchNorm2d(num_features=num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out1 = self.activate(self.BatchNorm(self.conv1(x)))
        out = self.conv2(out1)
        # out = self.softmax(out2)
        return out


