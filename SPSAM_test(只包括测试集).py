"""传统方法subpixel mapping based on a spatial attraction model (SPSAM)"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from utils import abundance_maker, compute_metrics, block_maker, downsample, load_mat_hsi, save_metrics_to_csv
from sklearn.model_selection import train_test_split
import os

# 设置随机种子
seed = 42
np.random.seed(seed)  # 控制NumPy的随机性

"""只对测试集部分进行评价的传统方法，另一个代码是对全图"""
# random.seed(seed)  # 控制Python内置random模块的随机性

# 转换后的 SPSAM 函数
def SPSAM(abundance, s):
    # 获取丰度图的行数、列数和类别数
    row, col, class_num = abundance.shape
    # 初始化引力图 F
    F = np.ones((row * s, col * s))
    # 初始化最终结果数组 Temp_F
    Temp_F = np.zeros((row * s, col * s, class_num))
    # 遍历每个类别
    for k in range(class_num):
        # 获取当前类别的丰度图
        abundance_K = abundance[:, :, k]
        # 遍历每一行
        for i in range(row):
            # 根据行的位置设置补丁的前两个值
            if i == 0:
                pad = [0, 1, 1, 1]
            elif i == row - 1:
                pad = [1, 0, 1, 1]
            else:
                pad = [1, 1, 1, 1]
            # 遍历每一列
            for j in range(col):
                # 根据列的位置设置补丁的后两个值
                if j == 0:
                    pad[2] = 0
                    pad[3] = 1
                elif j == col - 1:
                    pad[2] = 1
                    pad[3] = 0
                else:
                    pad[2] = 1
                    pad[3] = 1
                # 遍历亚像元的行
                for m in range(s):
                    # 遍历亚像元的列
                    for n in range(s):
                        # 调用 attraction_field 函数计算引力值并赋值给 F
                        F[i * s + m, j * s + n] = attraction_field(i, j, m, n, abundance_K, s, pad)
        # 将当前类别的引力图赋值给 Temp_F
        Temp_F[:, :, k] = F
    return Temp_F


def attraction_field(ii, jj, m, n, abundance, s, pad):
    # 计算第ii行jj列像元的第m行n个亚像元所得8个领域的引力 SPSAM
    # abundance为丰度图，s为亚像元尺度
    F = 0
    k = 0
    rows, cols = abundance.shape
    d = np.zeros((rows, cols))
    for i in range(ii - pad[0], ii + pad[1] + 1):
        for j in range(jj - pad[2], jj + pad[3] + 1):
            k = k + 1
            if ii == i and jj == j:
                d[i, j] = 0
            else:
                a = ((ii - 1) * s + m - 1 + 0.5 - (i - 1 + 0.5) * s)
                b = ((jj - 1) * s + n - 1 + 0.5 - (j - 1 + 0.5) * s)
                d[i, j] = abundance[i, j] / np.sqrt(a ** 2 + b ** 2)
            F = F + d[i, j]
    F = F / k
    return F


datasetnames = {'houston': 'houston',
                'trento': 'trento',
                'muufl': 'muufl'
                }
dataset = "muufl"  # Replace the data set here
image, gt, lidar, abundance = load_mat_hsi(datasetnames[dataset], "D:/study/方法代码尝试/亚像元制图/")
model_type = 'SPSAM'

model_dir = "./results/" + model_type + '/' + datasetnames[dataset] + '/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

gt = gt - 1
# 类别数
num_classes = gt.max() + 1
# 下采样倍数
downsample_factor = 4
height, width, channel = image.shape
# 计算下采样后的尺寸
new_height = gt.shape[0] // downsample_factor
new_width = gt.shape[1] // downsample_factor

# 定义块的尺寸
block_size = (48, 48)
down_block_size = (block_size[0] // downsample_factor, block_size[1] // downsample_factor)

abundance = abundance_maker(gt, new_height, new_width, num_classes, downsample_factor)  # 制作丰度

downsampled = downsample(image, new_height, new_width, channel)  # 影像下采样
blocks_image = block_maker(new_height, new_width, down_block_size, downsampled)  # 影像分块
blocks_gt = block_maker(height, width, block_size, gt)  # 真值分块
blocks_abundance = block_maker(new_height, new_width, down_block_size, abundance)  # 丰度分块

# 设定训练集的比例
train_size = 0.4
# 使用 train_test_split 函数进行划分
abundance_train, abundance_test, image_train, image_test, gt_train, gt_test = train_test_split(
    blocks_abundance, blocks_image, blocks_gt, train_size=train_size, random_state=9)

result_test = []
for i in range(abundance_test.shape[0]):
    print(i)
    S_GT = abundance_test[i]
    abundance = S_GT
    ROW, COL, cla = S_GT.shape

    s = 4
    # 假设 SPSAM 函数已经定义
    # R.RasterSize = R.RasterSize * s 这行代码在 Python 里不清楚对应逻辑，这里忽略
    F = SPSAM(abundance, s)

    row, col, class_num = F.shape

    abundance_extend = np.zeros((ROW, COL, class_num))
    for i in range(class_num - 1):
        temp = np.floor(abundance[:, :, i] * 16)
        abundance_extend[:, :, i] = temp

    temp = 16 * np.ones((ROW, COL))
    abundance_extend[:, :, i + 1] = temp - np.sum(abundance_extend, axis=2)

    abundance_2D = abundance_extend.reshape(ROW * COL, class_num)
    abundance_sum = np.sum(np.abs(abundance_2D), axis=1)

    TYPE = [[None for _ in range(COL)] for _ in range(ROW)]
    for i in range(ROW):
        for j in range(COL):
            W = np.ones((s, s))
            type_matrix = np.zeros((s, s))
            for k in range(class_num):
                M = F[(i * s):((i + 1) * s), (j * s):((j + 1) * s), k]
                Temp = W * M
                temp = Temp.flatten()
                B = np.sort(temp)[::-1]
                num_class = int(abundance_extend[i, j, k])

                if num_class == 0:
                    continue
                else:
                    r, c = np.where(Temp >= B[num_class - 1])
                    for n in range(len(r)):
                        W[r[n], c[n]] = 0
                        type_matrix[r[n], c[n]] = k

            TYPE[i][j] = type_matrix

    T = np.zeros((row, col))
    for i in range(ROW):
        for j in range(COL):
            T[(i * s):((i + 1) * s), (j * s):((j + 1) * s)] = TYPE[i][j]

    result_test.append(T)

result_test = np.array(result_test)

oa, kappa, accuracies, aa = compute_metrics(result_test, gt_test, num_classes=num_classes)
print(f'测试集- oa={oa}, kappa={kappa}')
save_metrics_to_csv(oa, kappa, accuracies * 100, aa, model_dir + 'classification_metrics.csv')
