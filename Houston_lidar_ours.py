import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from modle1 import ours
from scipy.io import savemat
import scipy.io as sio
from utils import downsample, block_maker, CustomDataset, display_image, compute_metrics, load_mat_hsi, \
    save_metrics_to_csv, restore_image
from utils import display_image_new, abundance_maker
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ===================== 5个随机种子 =====================
seeds = [40, 41, 42, 43, 44]
all_test_results = []


def SAD(output, target):
    _, band, h, w = output.shape
    output = torch.reshape(output, (-1, band, h * w))
    target = torch.reshape(target, (-1, band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=1))
    abundance_loss = torch.mean(abundance_loss)
    return abundance_loss


def test(model, lr_image, lr_gt_image, dem):
    with torch.no_grad():
        if lr_image.ndim == 3:
            lr_image = torch.tensor(lr_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).float().to(device)
            lr_gt_image = torch.tensor(lr_gt_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).float().to(
                device)
            dem = torch.tensor(dem, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).float().to(device)
            output = model(lr_image, lr_gt_image, dem)
            sr_image = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()
        else:
            lr_image = torch.tensor(lr_image, dtype=torch.float32).permute(0, 3, 1, 2).float().to(device)
            lr_gt_image = torch.tensor(lr_gt_image, dtype=torch.float32).permute(0, 3, 1, 2).float().to(device)
            dem = torch.tensor(dem, dtype=torch.float32).permute(0, 3, 1, 2).float().to(device)
            output = model(lr_image, lr_gt_image, dem)
            sr_image = torch.argmax(output, dim=1).cpu().numpy()
    return sr_image.astype(np.int64)


# 训练函数
def train(model, train_loader, criterion, optimizer, device, num_classes, num_epochs=1000):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for lr_imgs_batch, hr_imgs_batch, lr_gt_imgs_batch, dem in train_loader:
            lr_imgs_batch, lr_gt_imgs_batch, hr_imgs_batch, dem = lr_imgs_batch.to(device), lr_gt_imgs_batch.to(
                device), hr_imgs_batch.to(device), dem.to(device)
            optimizer.zero_grad()
            lr_imgs_batch = lr_imgs_batch.float()
            lr_gt_imgs_batch = lr_gt_imgs_batch.float()

            output = model(lr_imgs_batch, lr_gt_imgs_batch, dem)
            hr_imgs_batch_one_hot = torch.nn.functional.one_hot(hr_imgs_batch, num_classes=num_classes).permute(0, 3, 1,
                                                                                                                2).float()
            loss = criterion(output, hr_imgs_batch_one_hot)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}')
    model.eval()


# 示例用法
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    datasetnames = {'houston': 'houston',
                    'trento': 'trento',
                    'muufl': 'muufl',
                    }
    dataset = "houston"
    model_type = 'ours'
    model_dir = "./results/" + model_type + '/' + datasetnames[dataset] + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # ===================== 5次循环运行 =====================
    for run_idx, seed in enumerate(seeds):
        print(f"\n========== 第 {run_idx + 1} 次运行 | 随机种子 = {seed} ==========")

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 每次重新加载数据
        image, gt, lidar, abundance = load_mat_hsi(datasetnames[dataset], "D:/study/方法代码尝试/亚像元制图/")

        num_classes = gt.max()
        gt = gt - 1

        block_size = (36, 36)
        height, width, channel = image.shape
        downsample_factor = 4
        down_block_size = (block_size[0] // downsample_factor, block_size[1] // downsample_factor)
        new_height = gt.shape[0] // downsample_factor
        new_width = gt.shape[1] // downsample_factor

        downsampled = downsample(image, new_height, new_width, channel)
        lidar = lidar[:, :, np.newaxis]

        blocks_image = block_maker(new_height, new_width, down_block_size, downsampled)
        blocks_gt = block_maker(height, width, block_size, gt)
        blocks_abundance = block_maker(new_height, new_width, down_block_size, abundance)
        blocks_lidar = block_maker(height, width, block_size, lidar)

        if dataset == 'houston':
            rand = 8
        else:
            rand = 9

        train_size = 0.4
        abundance_train, abundance_test, image_train, image_test, gt_train, gt_test, lidar_train, lidar_test = \
            train_test_split(blocks_abundance, blocks_image, blocks_gt, blocks_lidar, train_size=train_size,
                             random_state=rand)

        # 数据集
        train_dataset = CustomDataset(image_train, gt_train, abundance_train, lidar_train)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # 模型
        model = ours(upscale_factor=4, L=channel, num_class=num_classes)
        criterion = nn.MSELoss()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = criterion.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

        # 训练
        train(model, train_loader, criterion, optimizer, device, num_classes, num_epochs=200)

        # 全图重建
        total_image = test(model, blocks_image, blocks_abundance, blocks_lidar)
        total_image = restore_image(total_image, height, width, block_size)
        mat_path = model_dir + f'reconstructed_image_seed{seed}.mat'
        savemat(mat_path, {'reconstructed_image': total_image})

        display_image_new(total_image, gt, model_dir, datasetnames[dataset], seed=seed)

        # ===================== 测试集指标 =====================
        test_image = test(model, image_test, abundance_test, lidar_test)
        oa_test, kappa_test, accuracies, aa_test = compute_metrics(test_image, gt_test, num_classes=num_classes)
        accuracies = accuracies * 100
        oa_test = oa_test * 100
        kappa_test = kappa_test * 100
        aa_test = aa_test * 100

        print(f"第 {run_idx + 1} 次测试集结果：")
        print(f"OA = {oa_test:.2f}, Kappa = {kappa_test:.2f}, AA = {aa_test:.2f}")

        # 保存本次结果，统一保留 2 位小数
        run_result = [
                         round(oa_test, 2),
                         round(aa_test, 2),
                         round(kappa_test, 2)
                     ] + [round(acc, 2) for acc in accuracies]

        all_test_results.append(run_result)

        # ===================== 计算均值 ± 标准差（统一 2 位小数） =====================
    all_res_np = np.array(all_test_results)
    mean_vals = np.mean(all_res_np, axis=0)
    std_vals = np.std(all_res_np, axis=0)

    mean_std_list = [
        f"{round(m, 2):.2f}±{round(s, 2):.2f}"
        for m, s in zip(mean_vals, std_vals)
    ]

    # 指标名称
    metric_names = ["OA", "AA", "Kappa"] + [f"Class_{i}_Acc" for i in range(num_classes)]

    # 构建CSV表格
    df = pd.DataFrame({
        "Metric": metric_names,
        "Seed_40": all_test_results[0],
        "Seed_41": all_test_results[1],
        "Seed_42": all_test_results[2],
        "Seed_43": all_test_results[3],
        "Seed_44": all_test_results[4],
        "Mean±Std": mean_std_list
    })

    # 保存最终CSV
    csv_save_path = model_dir + "classification_metrics_5runs.csv"
    df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 5次运行全部完成！结果已保存到：\n{csv_save_path}")
