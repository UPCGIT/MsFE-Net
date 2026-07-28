import numpy as np
import scipy.io as sio
import cv2
from torch.utils.data import Dataset
import torch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import scipy.io as io
import os
import pandas as pd


# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image, gt, abundance, lidar):
        self.abundance = torch.tensor(abundance, dtype=torch.float32)
        self.image = torch.tensor(image, dtype=torch.float32)
        self.gt = torch.tensor(gt, dtype=torch.float32)
        self.lidar = torch.tensor(lidar, dtype=torch.float32)

    def __len__(self):
        return len(self.abundance)

    def __getitem__(self, idx):
        abundance_sample = self.abundance[idx].permute(2, 0, 1)
        image_sample = self.image[idx].permute(2, 0, 1)
        gt_sample = self.gt[idx].long()
        lidar_sample = self.lidar[idx].permute(2, 0, 1)
        return image_sample, gt_sample, abundance_sample, lidar_sample


def load_mat_hsi(dataset_name, dataset_dir):
    """ load HSI.mat dataset """
    image = None
    gt = None
    lidar = None
    abundance = None

    if dataset_name == 'houston':
        image = io.loadmat(os.path.join(dataset_dir, "Houston2018/houston_hsi.mat"))
        image = image['houston_hsi']
        gt = io.loadmat(os.path.join(dataset_dir, "Houston2018/GTall.mat"))
        gt = gt['GTall']
        lidar = io.loadmat(os.path.join(dataset_dir, "Houston2018/houston_lidar.mat"))
        lidar = lidar['houston_lidar']
        abundance = io.loadmat(os.path.join(dataset_dir, "Houston2018/Houston_abundance.mat"))
        abundance = abundance['Houston_abundance']

    elif dataset_name == 'trento':
        image = io.loadmat(os.path.join(dataset_dir, "TrentoDateset-main/HSI_Trento.mat"))
        image = image['HSI_Trento']
        gt = io.loadmat(os.path.join(dataset_dir, "TrentoDateset-main/GTall_Trento.mat"))
        gt = gt['GTall_Trento']
        lidar = io.loadmat(os.path.join(dataset_dir, "TrentoDateset-main/Lidar_Trento.mat"))
        lidar = lidar['LiDAR']
        abundance = sio.loadmat(r'D:\study\方法代码尝试\亚像元制图\TrentoDateset-main\Trento_abundance.mat')[
            'Trento_abundance']

    elif dataset_name == 'muufl':
        image = io.loadmat(os.path.join(dataset_dir, "Muufl/HSI.mat"))
        image = image['HSI']
        gt = io.loadmat(os.path.join(dataset_dir, "Muufl/GTall.mat"))
        gt = gt['GTall']
        lidar = io.loadmat(os.path.join(dataset_dir, "Muufl/LiDAR.mat"))
        lidar = lidar['LiDAR']
        abundance = sio.loadmat(r'D:\study\方法代码尝试\亚像元制图\Muufl\Muufl_abundance.mat')['Muufl_abundance']

    return image, gt, lidar, abundance


def downsample(image, new_height, new_width, channel):
    # 高斯模糊
    blurred = np.zeros_like(image)
    for i in range(channel):
        blurred[:, :, i] = cv2.GaussianBlur(image[:, :, i], (7, 7), 0.25)
    # 双三次插值降采样
    downsampled = np.zeros((new_height, new_width, channel), dtype=image.dtype)
    for i in range(channel):
        downsampled[:, :, i] = cv2.resize(blurred[:, :, i], (max(new_height, new_width), min(new_height, new_width)),
                                          interpolation=cv2.INTER_CUBIC)
    downsampled = (downsampled - np.min(downsampled)) / (np.max(downsampled) - np.min(downsampled))

    return downsampled


def abundance_maker(gt, new_height, new_width, num_classes, downsample_factor):
    # 初始化丰度图
    abundance_map = np.zeros((new_height, new_width, num_classes))

    # 滑动窗口进行下采样
    for i in range(new_height):
        for j in range(new_width):
            # 确定当前窗口的位置
            start_row = i * downsample_factor
            end_row = start_row + downsample_factor
            start_col = j * downsample_factor
            end_col = start_col + downsample_factor

            # 提取当前窗口
            window = gt[start_row:end_row, start_col:end_col]

            # 计算窗口内各类别的数量
            class_counts = np.bincount(window.flatten(), minlength=num_classes)

            # 计算各类别的比例
            class_proportions = class_counts / window.size

            # 将比例赋值给丰度图对应的位置
            abundance_map[i, j] = class_proportions

    return abundance_map


def block_maker(height, width, block_size, image):
    blocks_data = []
    # 逐行逐列进行分割
    for i in range(0, height, block_size[0]):
        for j in range(0, width, block_size[1]):
            # 计算当前块的起始和结束位置
            if i + block_size[0] > height:
                # 如果行数不够64，从最后一行往上数64行
                start_row = max(0, height - block_size[0])
                end_row = height
            else:
                start_row = i
                end_row = i + block_size[0]

            if j + block_size[1] > width:
                # 如果列数不够64，从最后一列往左数64列
                start_col = max(0, width - block_size[1])
                end_col = width
            else:
                start_col = j
                end_col = j + block_size[1]

            # 提取当前块
            block = image[start_row:end_row, start_col:end_col]

            # 将块添加到列表中
            blocks_data.append(block)
    blocks_data = np.array(blocks_data)

    return blocks_data


def display_image(sr_image, hr_image, path, title_sr='Sub-pixel mapping Image', title_hr='Ground Truth'):
    cmap = plt.get_cmap('tab20', int(np.max(sr_image) - np.min(sr_image) + 1))
    bounds = np.linspace(np.min(sr_image), np.max(sr_image), np.max(sr_image) - np.min(sr_image) + 2)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(sr_image, cmap=cmap, norm=norm)
    plt.colorbar(ticks=np.arange(np.min(sr_image), np.max(sr_image) + 1))
    plt.title(title_sr)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(hr_image, cmap=cmap, norm=norm)
    plt.colorbar(ticks=np.arange(np.min(hr_image), np.max(hr_image) + 1))
    plt.title(title_hr)
    plt.axis('off')
    plt.savefig(path + '分类结果.png')
    plt.show()


def display_image_new(sr_image, hr_image, path, type='trento', title_sr='Sub-pixel mapping Image',
                      title_hr='Ground Truth', seed=42):
    sr_image, small = label2color(sr_image, type)
    hr_image, small2 = label2color(hr_image, type)

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(sr_image)
    plt.title(title_sr)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(hr_image)
    plt.title(title_hr)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(small)
    plt.title(title_sr)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(small2)
    plt.title(title_hr)
    plt.axis('off')

    plt.savefig(path + f'分类结果{seed}.png', dpi=600)


def compute_metrics(prediction, ground_truth, num_classes):
    prediction = prediction.flatten()
    ground_truth = ground_truth.flatten()
    oa = accuracy_score(ground_truth, prediction)
    kappa = cohen_kappa_score(ground_truth, prediction)
    cm = confusion_matrix(ground_truth, prediction, labels=np.arange(num_classes))
    # 计算每一类的准确率
    accuracies = np.diagonal(cm) / (cm.sum(axis=1))
    aa = np.mean(accuracies)

    return oa, kappa, accuracies, aa


def save_metrics_to_csv(oa, kappa, accuracies, aa, csv_path):
    """将分类指标保存到CSV文件"""
    # 创建总体指标数据框
    overall_metrics = pd.DataFrame({
        '指标': ['总体准确率 (OA)', 'Kappa系数', '平均准确率 (AA)'],
        '值': [oa, kappa, aa]
    })

    # 创建类别准确率数据框
    class_accuracies = pd.DataFrame({
        '类别': [f'Class {i}' for i in range(len(accuracies))],
        '准确率': accuracies
    })

    # 将类别准确率转换为百分比格式
    class_accuracies['准确率(%)'] = (class_accuracies['准确率']).map('{:.2f}'.format)

    # 写入CSV文件
    with open(csv_path, 'w', newline='') as f:
        # 写入总体指标
        overall_metrics.to_csv(f, index=False)

        # 添加空行分隔
        f.write('\n')

        # 写入类别准确率
        class_accuracies[['类别', '准确率(%)']].to_csv(f, index=False)


def restore_image(blocks, height, width, block_size):
    # 初始化一个全零的图像数组
    restored_image = np.zeros((height, width), dtype=blocks[0].dtype)
    block_index = 0
    # 逐行逐列进行恢复
    for i in range(0, height, block_size[0]):
        for j in range(0, width, block_size[1]):
            # 计算当前块的起始和结束位置
            if i + block_size[0] > height:
                # 如果行数不够64，从最后一行往上数64行
                start_row = max(0, height - block_size[0])
                end_row = height
            else:
                start_row = i
                end_row = i + block_size[0]

            if j + block_size[1] > width:
                # 如果列数不够64，从最后一列往左数64列
                start_col = max(0, width - block_size[1])
                end_col = width
            else:
                start_col = j
                end_col = j + block_size[1]

            # 获取当前块
            block = blocks[block_index]
            # 将块放回图像中
            restored_image[start_row:end_row, start_col:end_col] = block
            block_index += 1

    return restored_image


def label2color(label, data_name):
    w, h = label.shape
    im = np.zeros((w, h, 3), dtype=np.uint8)

    data_name = data_name.lower()

    if data_name == 'uni':
        map = [[131, 90, 254], [190, 180, 148], [0, 255, 255], [0, 128, 0], [255, 0, 255], [165, 82, 41],
               [128, 0, 128], [255, 0, 0], [255, 255, 0]]
    elif data_name == 'trento':
        map = [[156, 209, 174], [231, 205, 238], [255, 210, 147], [240, 171, 163], [105, 134, 182], [173, 54, 136]]
    elif data_name == 'houston':
        map = [[248, 185, 79], [236, 123, 107], [105, 134, 182], [167, 211, 152], [73, 186, 200], [38, 70, 83],
               [182, 34, 130], [216, 191, 216], [255, 0, 0], [139, 0, 0], [51, 102, 255], [255, 255, 0],
               [238, 154, 0], [85, 26, 139], [255, 127, 80]]
    elif data_name == 'muufl':
        map = [[76, 147, 189], [153, 51, 0], [254, 168, 9], [0, 255, 0], [202, 157, 186], [0, 51, 255],
               [0, 155, 158], [255, 0, 0], [255, 255, 0], [138, 112, 103], [202, 100, 95]]
    elif data_name == 'yrddvd':
        map = [[202, 100, 95], [208, 190, 176], [255, 100, 0], [106, 179, 162], [0, 0, 255], [173, 54, 136],
               [149, 169, 56], [60, 91, 112], [255, 255, 0], [255, 0, 255], [255, 255, 125], [100, 0, 255],
               [0, 172, 254], [0, 255, 0], [171, 175, 80], [101, 193, 60]]
    elif data_name == 'yrdmvd':
        map = [[202, 100, 95], [208, 190, 176], [220, 180, 100], [106, 179, 162], [0, 0, 255], [173, 54, 136],
               [149, 169, 56], [60, 91, 112], [255, 255, 0], [255, 0, 255], [255, 255, 125], [100, 0, 255],
               [0, 172, 254], [0, 255, 0], [171, 175, 80], [101, 193, 60]]
    else:
        return None

    for i in range(w):
        for j in range(h):
            index = int(label[i, j])
            im[i, j, :] = np.uint8(map[index])

    im = np.uint8(im)
    classif = np.uint8(np.zeros((w, h, 3)))
    classif[:, :, 0] = im[:, :, 0]
    classif[:, :, 1] = im[:, :, 1]
    classif[:, :, 2] = im[:, :, 2]

    if data_name == 'trento':
        small = classif[0:100, 90:190, :]
    elif data_name == 'houston':
        small = classif[200:330, 0:130, :]
    elif data_name == 'muufl':
        small = classif[85:155, 155:225, :].transpose(1, 0, 2)
        classif = classif.transpose(1, 0, 2)
    elif data_name == 'yrddvd':
        small = classif[0:50, 0:50, :]
    elif data_name == 'yrdmvd':
        small = classif[0:50, 0:50, :]

    return classif, small
