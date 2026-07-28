# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import itertools
import spectral
import matplotlib.pyplot as plt
from scipy import io
import imageio
import os
import torch.utils.data as dataf
import torch
import time
import math
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


def svm_classifier(img, train_gt, test_gt):
    x_train = img[train_gt > 0].reshape(-1, img.shape[-1])
    x_test = img[test_gt > 0].reshape(-1, img.shape[-1])
    y_train = train_gt[train_gt > 0].reshape(-1, 1).ravel()
    svm_model_final = svm.SVC(kernel='rbf', probability=False, random_state=0, C=1024, gamma=0.125)
    svm_model_final.fit(x_train, y_train)
    svm_predict = svm_model_final.predict(x_test)
    x_test = None
    x_train = None
    y_train = None
    return svm_predict, svm_model_final


def randomForest(img, train_gt, test_gt=[]):
    x_train = img[train_gt > 0].reshape(-1, img.shape[-1])
    if len(test_gt) > 0:
        x_test = img[test_gt > 0].reshape(-1, img.shape[-1])
    y_train = train_gt[train_gt > 0].reshape(-1, 1).ravel()
    random_forest_model_test_random = RandomForestClassifier(random_state=0)
    random_forest_model_test_random.fit(x_train, y_train)
    if len(test_gt) > 0:
        random_forest_predict = random_forest_model_test_random.predict(x_test)
    importances = random_forest_model_test_random.feature_importances_
    indices = np.argsort(importances)[::-1]
    importances = None
    x_test = None
    x_train = None
    y_train = None
    if len(test_gt) > 0:
        return random_forest_predict, random_forest_model_test_random, indices
    else:
        return random_forest_model_test_random, indices


def load_mat_hsi(dataset_name, dataset_dir):
    """ load HSI.mat dataset """
    # available sets
    available_sets = [
        'houston',
        'trento',
        'muufl'
    ]
    assert dataset_name in available_sets, "dataset should be one of" + ' ' + str(available_sets)

    image = None
    gt = None
    labels = None

    if dataset_name == 'houston':
        image = io.loadmat(os.path.join(dataset_dir, "Houston2018/houston_hsi.mat"))
        image = image['houston_hsi']
        gt = io.loadmat(os.path.join(dataset_dir, "Houston2018/houston_gt.mat"))
        gt = gt['houston_gt']
        lidar = io.loadmat(os.path.join(dataset_dir, "Houston2018/houston_lidar.mat"))
        lidar = lidar['houston_lidar']
        num_class = int(gt.max())
        labels = list(range(1, num_class + 1))

    elif dataset_name == 'trento':
        image = io.loadmat(os.path.join(dataset_dir, "TrentoDateset-main/HSI_Trento.mat"))
        image = image['HSI_Trento']
        gt = io.loadmat(os.path.join(dataset_dir, "TrentoDateset-main/GT_Trento.mat"))
        gt = gt['GT_Trento']
        lidar = io.loadmat(os.path.join(dataset_dir, "TrentoDateset-main/Lidar_Trento.mat"))
        lidar = lidar['Lidar_Trento']
        num_class = int(gt.max())
        labels = list(range(1, num_class + 1))

    elif dataset_name == 'muufl':
        image = io.loadmat(os.path.join(dataset_dir, "Muufl/HSI.mat"))
        image = image['HSI']
        gt = io.loadmat(os.path.join(dataset_dir, "Muufl/gt.mat"))
        gt = gt['gt']
        lidar = io.loadmat(os.path.join(dataset_dir, "Muufl/LiDAR.mat"))
        lidar = lidar['LiDAR']
        num_class = int(gt.max())
        labels = list(range(1, num_class + 1))
    # after getting image and ground truth (gt), let us do data preprocessing!
    # step1 filter nan values out
    nan_mask = np.isnan(image.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("warning: nan values found in dataset {}, using 0 replace them".format(dataset_name))
        image[nan_mask] = 0
        gt[nan_mask] = 0

    # step2 normalise the HSI data (method from SSAN, TGRS 2020)
    image = np.asarray(image, dtype=np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    mean_by_c = np.mean(image, axis=(0, 1))

    for c in range(image.shape[-1]):
        image[:, :, c] = image[:, :, c] - mean_by_c[c]
        # image[:, :, c] = (image[:, :, c] - np.min(image[:, :, c])) / (np.max(image[:, :, c]) - np.min(image[:, :, c]))

    return image, gt, labels, lidar



def label2color(label, data_name):
    w, h = label.shape
    im = np.zeros((w, h, 3), dtype=np.uint8)

    data_name = data_name.lower()

    if data_name == 'uni':
        map = [[131, 90, 254], [190, 180, 148], [0, 255, 255], [0, 128, 0], [255, 0, 255], [165, 82, 41],
               [128, 0, 128], [255, 0, 0], [255, 255, 0]]
    elif data_name == 'india':
        map = [[202, 100, 95], [208, 190, 176], [255, 100, 0], [106, 179, 162], [0, 0, 255], [173, 54, 136],
               [149, 169, 56], [60, 91, 112], [255, 255, 0], [255, 0, 255], [255, 255, 125], [100, 0, 255],
               [0, 172, 254], [0, 255, 0], [171, 175, 80], [101, 193, 60]]
    elif data_name == 'houston':
        map = [[0, 205, 0], [127, 255, 0], [46, 139, 87], [0, 139, 0], [160, 82, 45], [0, 255, 255],
               [255, 255, 255], [216, 191, 216], [255, 0, 0], [139, 0, 0], [51, 102, 255], [255, 255, 0],
               [238, 154, 0], [85, 26, 139], [255, 127, 80]]
    elif data_name == 'dc':
        map = [[204, 102, 102], [153, 51, 0], [204, 153, 0], [0, 255, 0], [0, 102, 0], [0, 51, 255],
               [153, 153, 153]]
    else:
        return None

    for i in range(w):
        for j in range(h):
            index = int(label[i, j])
            if index == 0:
                im[i, j, :] = np.uint8([0, 0, 0])
                continue
            im[i, j, :] = np.uint8(map[index - 1])

    im = np.uint8(im)
    classif = np.uint8(np.zeros((w, h, 3)))
    classif[:, :, 0] = im[:, :, 0]
    classif[:, :, 1] = im[:, :, 1]
    classif[:, :, 2] = im[:, :, 2]

    return classif


def seed_worker(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metrics(prediction, target, n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, accuracy by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    ignored_mask[target < 0] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]
    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy /= float(total)

    results["Accuracy"] = accuracy * 100.0

    # Compute accuracy of each class
    class_acc = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            acc = cm[i, i] / np.sum(cm[i, :])
        except ZeroDivisionError:
            acc = 0.
        class_acc[i] = acc

    results["class acc"] = class_acc * 100.0
    results['AA'] = np.mean(class_acc) * 100.0
    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa * 100.0

    return results


def show_results(results, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        aa = [r['AA'] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]

        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        aa = results['AA']
        classacc = results["class acc"]
        kappa = results["Kappa"]

    if agregated:
        text += ("Accuracy: {:.02f} +- {:.02f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.02f}%\n".format(accuracy)
    text += "---\n"

    text += "class acc :\n"
    if agregated:
        for label, score, std in zip(label_values, class_acc_mean,
                                     class_acc_std):
            text += "\t{}: {:.02f} +- {:.02f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, classacc):
            text += "\t{}: {:.02f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("AA: {:.02f} +- {:.02f}\n".format(np.mean(aa),
                                                   np.std(aa)))
        text += ("Kappa: {:.02f} +- {:.02f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.02f}\n".format(kappa)

    print(text)


def save_results_to_csv(results_list, csv_file_path):
    # 获取所有结果中类别数量的最大值，以确保CSV文件的列数一致
    max_num_classes = max(len(result['class acc']) for result in results_list)

    # 定义CSV文件的表头
    fieldnames = ['Class ' + str(i) + ' Acc' for i in range(max_num_classes)] + ['OA', 'AA', 'Kappa']

    # 打开或创建CSV文件
    with open(csv_file_path, mode='w', newline='') as csvfile:
        # 创建一个DictWriter对象
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 遍历每次运行的结果并写入CSV文件
        for result in results_list:
            class_acc = [f"{acc:.2f}" for acc in (result['class acc'])]

            # 创建一行数据字典
            row_data = {
                'OA': f"{result['Accuracy']:.2f}",
                'AA': f"{result['AA']:.2f}",
                'Kappa': f"{result['Kappa']:.2f}"
            }
            # 将类别准确率添加到数据字典中
            for i, acc in enumerate(class_acc):
                row_data['Class ' + str(i) + ' Acc'] = acc

            # 写入一行数据
            writer.writerow(row_data)


def sample_gt(gt, train_size, mode="random", less_choice_per=0.5, seed=None):
    """
    Extract a fixed precentage of samples from an array of labels
    :param gt: a 2D array of int labels
    :param train_size: [0,1] float
    :param mode:
    :return: train_gt, test_gt (2D array of int labels)
    """
    if seed is not None:
        seed_worker(5)
    indices = np.nonzero(gt)
    X = list(zip(*indices))
    y = gt[indices].ravel()
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if gt.ndim == 2:
        if mode == "random":
            if train_size >= 1:
                data_label = gt.reshape(gt.shape[0] * gt.shape[1])
                train_gt = np.zeros_like(data_label)
                test_gt = data_label.copy()
                data_location = np.arange(data_label.shape[0])
                train_size = int(train_size)
                for i in np.unique(y):
                    if train_size >= np.sum(data_label == i):
                        train_size_class = math.ceil(np.sum(data_label == i) * less_choice_per)
                        replaceOptions = False
                    else:
                        train_size_class = train_size
                        replaceOptions = False
                    np.random.seed(int(time.time()))
                    if seed is not None:
                        seed_worker(5)
                    choice_train_location = np.random.choice(data_location[data_label == i],
                                                             min(train_size_class, np.sum(data_label == i)),
                                                             replace=replaceOptions)
                    train_gt[choice_train_location] = i
                    test_gt[choice_train_location] = 0
                train_gt = train_gt.reshape(gt.shape[0], gt.shape[1])
                test_gt = test_gt.reshape(gt.shape[0], gt.shape[1])
            else:
                data_label = gt.reshape(gt.shape[0] * gt.shape[1])
                train_gt = np.zeros_like(data_label)
                test_gt = data_label.copy()
                data_location = np.arange(data_label.shape[0])
                for i in np.unique(y):
                    replaceOptions = False
                    np.random.seed(int(time.time()))
                    if seed is not None:
                        seed_worker(5)
                    choice_train_location = np.random.choice(data_location[data_label == i],
                                                             math.ceil(train_size * np.sum(data_label == i)),
                                                             replace=replaceOptions)
                    train_gt[choice_train_location] = i
                    test_gt[choice_train_location] = 0
                train_gt = train_gt.reshape(gt.shape[0], gt.shape[1])
                test_gt = test_gt.reshape(gt.shape[0], gt.shape[1])
        elif mode == "disjoint":
            test_gt = np.copy(gt)
            for c in np.unique(gt):
                mask = gt == c
                for x in range(gt.shape[0]):
                    first_half_count = np.count_nonzero(mask[:x, :])
                    second_half_count = np.count_nonzero(mask[x:, :])
                    try:
                        ratio = first_half_count / (second_half_count + first_half_count)
                        if ratio > train_size - 0.05 and ratio < train_size + 0.05:
                            break
                    except ZeroDivisionError:
                        continue
                mask[:x, :] = 0
                train_gt[mask] = 0
                mask = None
            test_gt[train_gt > 0] = 0
        else:
            raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


def HSIDataset(Data, TrLabel, TsLabel, patchsize, batchsize):
    # boundary interpolation
    [m, n, l] = np.shape(Data)
    x = Data
    temp = x[:, :, 0]
    pad_width = np.floor(patchsize / 2)
    pad_width = int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape
    x2 = np.empty((m2, n2, l), dtype='float32')

    for i in range(l):
        temp = x[:, :, i]
        pad_width = np.floor(patchsize / 2)
        pad_width = int(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2[:, :, i] = temp2

    # construct the training and testing set
    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum = len(ind1)
    TrainPatch = np.empty((TrainNum, l, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    pos1 = ind1
    pos2 = ind2
    pos3 = pos1 + patchsize
    pos4 = pos2 + patchsize
    for i in range(len(ind1)):
        patch = x2[pos1[i]:pos3[i], pos2[i]:pos4[i], :]
        patch = np.reshape(patch, (patchsize * patchsize, l))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (l, patchsize, patchsize))
        TrainPatch[i, :, :, :] = patch
        patchlabel = TrLabel[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel

    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, l, patchsize, patchsize), dtype='float32')
    TestLabel = np.empty(TestNum)
    pos1 = ind1
    pos2 = ind2
    pos3 = pos1 + patchsize
    pos4 = pos2 + patchsize
    for i in range(len(ind1)):
        patch = x2[pos1[i]:pos3[i], pos2[i]:pos4[i], :]
        patch = np.reshape(patch, (patchsize * patchsize, l))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (l, patchsize, patchsize))
        TestPatch[i, :, :, :] = patch
        patchlabel = TsLabel[ind1[i], ind2[i]]
        TestLabel[i] = patchlabel

    print('Training size and testing size are:', TrainPatch.shape, 'and', TestPatch.shape)

    # step3: change data to the input type of PyTorch
    TrainPatch = torch.from_numpy(TrainPatch)
    # set class index starts from 0
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    dataset = dataf.TensorDataset(TrainPatch, TrainLabel)
    train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)

    TestPatch = torch.from_numpy(TestPatch)
    TestLabel = torch.from_numpy(TestLabel) - 1
    TestLabel = TestLabel.long()

    return train_loader, TestPatch, TestLabel, x2


def test(TestLabel, TestPatch, model, total=1000):
    pred_y = np.empty((len(TestLabel)), dtype='float32')
    number = len(TestLabel) // total
    for i in range(number):
        temp = TestPatch[i * total:(i + 1) * total, :, :]
        temp = temp.cuda()
        temp2 = model(temp)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[i * total:(i + 1) * total] = temp3.cpu()
        del temp, temp2, temp3

    if (i + 1) * total < len(TestLabel):
        temp = TestPatch[(i + 1) * total:len(TestLabel), :, :]
        temp = temp.cuda()
        temp2 = model(temp)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i + 1) * total:len(TestLabel)] = temp3.cpu()
        del temp, temp2, temp3

    return pred_y


def plot_result(Data, patchsize, x2, gt, model, color_type, path, part=1000):
    [m, n, l] = np.shape(Data)
    pred_all = np.empty((m * n, 1), dtype='float32')

    number = m * n // part
    for i in range(number):
        D = np.empty((part, l, patchsize, patchsize), dtype='float32')
        count = 0
        for j in range(i * part, (i + 1) * part):
            row = j // n
            col = j - row * n
            patch = x2[row:row + patchsize, col:col + patchsize, :]
            patch = np.reshape(patch, (patchsize * patchsize, l))
            patch = np.transpose(patch)
            patch = np.reshape(patch, (l, patchsize, patchsize))
            D[count, :, :, :] = patch
            count += 1

        temp = torch.from_numpy(D)
        temp = temp.cuda()
        temp2 = model(temp)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_all[i * part:(i + 1) * part, 0] = temp3.cpu()
        del temp, temp2, temp3, D

    if (i + 1) * part < m * n:
        D = np.empty((m * n - (i + 1) * part, l, patchsize, patchsize), dtype='float32')
        count = 0
        for j in range((i + 1) * part, m * n):
            row = j // n
            col = j - row * n
            patch = x2[row:row + patchsize, col:col + patchsize, :]
            patch = np.reshape(patch, (patchsize * patchsize, l))
            patch = np.transpose(patch)
            patch = np.reshape(patch, (l, patchsize, patchsize))
            D[count, :, :, :] = patch
            count += 1

        temp = torch.from_numpy(D)
        temp = temp.cuda()
        temp2 = model(temp)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_all[(i + 1) * part:m * n, 0] = temp3.cpu()
        del temp, temp2, temp3, D

    pred_all = np.reshape(pred_all, (m, n)) + 1

    result_map = label2color(pred_all, color_type)
    gt = label2color(gt, color_type)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))
    ax1.imshow(result_map)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Pred', fontsize=22)
    ax2.imshow(gt)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('GT', fontsize=22)
    plt.tight_layout()
    plt.savefig(path + '分类结果.png')

    io.savemat(path + 'result.mat', {'result': pred_all})




