import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from modle1 import ESPCN
from SFRN_MLFF_net import SpectralFeedbackNet
from scipy.io import savemat
from modle1 import SCNet
from modle1 import SIMNET
import scipy.io as sio
from utils import downsample, abundance_maker, block_maker, CustomDataset, display_image, compute_metrics, load_mat_hsi, \
    save_metrics_to_csv
from utils import display_image_new
from sklearn.model_selection import train_test_split
from modle import SRMCNN
from model_MSMCNet import MSMC
from model_MsFEIFN_spm import MsFE
# ===================== 核心修改：5个随机种子 =====================
seeds = [40, 41, 42, 43, 44]
all_test_results = []  # 存储5次测试集指标


def test(model, lr_image, lr_gt_image, dem):
    with torch.no_grad():
        if lr_image.ndim == 3:
            lr_image = torch.tensor(lr_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).float().to(device)
            lr_gt_image = torch.tensor(lr_gt_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).float().to(
                device)
            dem = torch.tensor(dem, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).float().to(
                device)
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
    torch.save(model.state_dict(), 'YRDMVD_zouxin.pkl')
    model.eval()


# 示例用法
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    datasetnames = {'houston': 'houston',
                    'trento': 'trento',
                    'muufl': 'muufl',
                    }
    dataset = "muufl"
    model_type = 'MsFE'
    model_dir = "./results/" + model_type + '/' + datasetnames[dataset] + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # ===================== 循环运行 5 次 =====================
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
        # random_state,houston,8

        if run_idx == 0:
            # 统计总 GT 数据里各类别的像素数量
            total_pixel_counts = np.bincount(blocks_gt.flatten())
            # 统计 gt_train 数据里各类别的像素数量
            train_pixel_counts = np.bincount(gt_train.flatten())
            # 计算不同类别被选中作为训练集的比例
            selection_ratios = train_pixel_counts / total_pixel_counts
            # 打印结果，可通过调整train_test_split中的random_state调整比例
            for class_label, ratio in enumerate(selection_ratios):
                print(
                    f"类别 {class_label} 总数量为 {total_pixel_counts[class_label]} 被选中作为训练集的比例: {ratio:.2%}")

        # 数据集
        train_dataset = CustomDataset(image_train, gt_train, abundance_train, lidar_train)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # 模型选择
        if model_type == 'ESPCN':
            model = ESPCN(upscale_factor=4, L=channel, num_class=num_classes)
            criterion = nn.MSELoss()
        elif model_type == 'MSMC':
            model = MSMC(upscale_factor=4, L=channel, num_class=num_classes)
            criterion = nn.MSELoss()
        elif model_type == "SFRNet":
            model = SpectralFeedbackNet(ch_in=channel, ch_out=32, class_num=num_classes, upscale_factor=2,
                                        semantic_up_scale=4)
            criterion = nn.CrossEntropyLoss()
        elif model_type == 'SIMNET':
            model = SIMNET(L=channel, num_class=num_classes)
            criterion = nn.MSELoss()
        elif model_type == 'SRMCNN':
            model = SRMCNN(channel, num_classes)
            criterion = nn.MSELoss()
        elif model_type == 'MsFE':
            model = MsFE(channel+num_classes, 1, num_classes, up=4)
            criterion = nn.MSELoss()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = criterion.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

        # 训练
        train(model, train_loader, criterion, optimizer, device, num_classes, num_epochs=200)

        # 全图测试
        total_image = test(model, downsampled, abundance, lidar)
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