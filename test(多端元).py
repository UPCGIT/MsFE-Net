import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import io as sio
from utils_unmixing import sample_gt, metrics, show_results, load_mat_hsi, downsample
import pandas as pd
import math
method_name = '2DCNN'
"""最后采用方法，多条端元"""
Data = sio.loadmat(r"D:\study\方法代码尝试\亚像元制图\Houston2018\houston_hsi.mat")["houston_hsi"]
gt = sio.loadmat(r"D:\study\方法代码尝试\亚像元制图\Houston2018\GTall.mat")["GTall"]

Data = sio.loadmat(r"D:\study\方法代码尝试\亚像元制图\Muufl\HSI.mat")["HSI"]
gt = sio.loadmat(r"D:\study\方法代码尝试\亚像元制图\Muufl\GTall.mat")["GTall"]

Data = sio.loadmat(r'D:\study\方法代码尝试\亚像元制图\TrentoDateset-main\HSI_Trento.mat')['HSI_Trento']
gt = sio.loadmat(r'D:\study\方法代码尝试\亚像元制图\TrentoDateset-main\GTall_Trento.mat')['GTall_Trento']
[m, n, l] = np.shape(Data)
# 获取图像的尺寸
height, width, channel = Data.shape
# 下采样倍数
downsample_factor = 4
# 计算下采样后的尺寸
new_height = gt.shape[0] // downsample_factor
new_width = gt.shape[1] // downsample_factor
downsampled = downsample(Data, new_height, new_width, channel)  # 影像下采样

# 1. 均值池化
# 重塑数组并计算每个4x4块的均值
pooled = gt.reshape(m//4, 4, n//4, 4).mean(axis=(1, 3))
# 2. 保留整数部分，其他位置置零，寻找降采样后依然为整数即纯像元的位置
result = np.where(pooled % 1 == 0, pooled, 0)

'''划分训练集测试集，从每一类里选取多少比例的训练集'''
TrLabel, TsLabel = sample_gt(gt=result, train_size=0.4, mode='random', seed=42)
num_class = int(gt.max())


df = result.flatten()
df = pd.Series(df)
counts = df.value_counts().sort_index().tolist()[1:]
print(df.value_counts())

endmember = np.empty((0, l))
for i in range(1, num_class + 1):
    curve = downsampled[TrLabel == i]
    endmember = np.vstack((endmember, curve))
endmember = np.array(endmember)
num = endmember.shape[0]
"""
fig, ax = plt.subplots()
for i in range(num_class):
    ax.plot(endmember[i], label=f'Array {i + 1}')
ax.set_xlabel('Index')
ax.set_ylabel('Value')
plt.show()
"""
GT_endmember = endmember.T
endmember_init = torch.from_numpy(GT_endmember).unsqueeze(2).unsqueeze(3).float()


Data2 = np.transpose(downsampled, [2, 0, 1])
Data2 = torch.from_numpy(Data2).unsqueeze(0).cuda()

batchsize = 128
patchsize = 16
LR = 0.001
EPOCH = 2000
EPOCH2 = 100
weight_decay = 0


def SAD(output, target):
    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss


# construct the reconstruction network
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        # encoding layers

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=l,
                out_channels=24,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(24),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, num, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num),
            nn.Dropout(0.2),
        )

        self.softmax = nn.Softmax(dim=1)

        # decoding layers
        self.dconv1 = nn.Sequential(
            nn.Conv2d(num, l, 1, 1, 0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        abu = self.softmax(x)
        y = self.dconv1(abu)
        return abu, y


ae = CAE()
ae.cuda()
model_dict = ae.state_dict()
model_dict["dconv1.0.weight"] = endmember_init
ae.load_state_dict(model_dict)
ae.dconv1[0].weight.requires_grad = False
optimizer = torch.optim.Adam(ae.parameters(), lr=LR, weight_decay=weight_decay)
loss_func = nn.MSELoss()

BestAcc = 100

# train the AE and save the best model
for epoch in range(EPOCH):
    abu, output = ae(Data2)  # rnn output
    loss1 = SAD(output, Data2)  # cross entropy loss
    loss3 = torch.sum(torch.pow(torch.abs(abu) + 1e-8, 0.5))
    loss = loss1
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if epoch % 10 == 0:
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())

en_abundance, reconstruction_result = ae(Data2)
en_abundance = torch.squeeze(en_abundance)

en_abundance = torch.reshape(en_abundance, [num, new_height * new_width])
en_abundance = en_abundance.T
en_abundance = torch.reshape(en_abundance, [new_height, new_width, num])
en_abundance = en_abundance.cpu().detach().numpy()


selected_counts = [math.ceil(num * 0.4) for num in counts]  # 每类端元数量
# 确保数量总和正确
assert sum(selected_counts) == en_abundance.shape[2], "端元数量总和与丰度向量维度不匹配"
# 初始化结果数组
merged_abundance = np.zeros((new_height, new_width, num_class))
# 按类别合并丰度向量
start_idx = 0
for class_idx, count in enumerate(selected_counts):
    end_idx = start_idx + count
    # 对当前类别的所有端元丰度求和
    merged_abundance[:, :, class_idx] = np.sum(en_abundance[:, :, start_idx:end_idx], axis=2)
    start_idx = end_idx


# sio.savemat('Trento_abundance1.mat', {'Trento_abundance': merged_abundance})