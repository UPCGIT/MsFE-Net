import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import sample_gt, metrics, show_results, load_mat_hsi, HSIDataset, test, plot_result, save_results_to_csv
from model_ExVit import MViT
import os
import matplotlib.pyplot as plt

'''导入数据集与标签图india, pu, '''
datasetnames = {'houston': 'houston',
                'trento': 'trento',
                'muufl': 'muufl'
                }
dataset = "houston"  # Replace the data set here
Data, gt, labels, lidar = load_mat_hsi(datasetnames[dataset], "D:/study/方法代码尝试/亚像元制图/")
method_name = 'ExVit'
'''划分训练集测试集，从每一类里选取多少比例的训练集'''
TrLabel, TsLabel = sample_gt(gt=gt, train_size=1000, mode='random', seed=42)

lidar = lidar[:, :, np.newaxis]
Data = np.concatenate((Data, lidar), axis=2)  # 按高光谱、Lidar顺序叠加，导入到网络后再分开

df = gt.flatten()
df = pd.Series(df)
print(df.value_counts())

patchsize = 10  # input spatial size for 2D-CNN
batchsize = 64  # select from [16, 32, 64, 128], the best is 64
# 迭代
EPOCH = 40
LR = 0.001
num_run = 1
num_class = int(gt.max())
[m, n, l] = np.shape(Data)

results = []
model_dir = "./results/" + method_name + '/' + datasetnames[dataset] + '/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
check_dir = model_dir + 'checkpoints/'
if not os.path.exists(check_dir):
    os.makedirs(check_dir)
# train and test the designed model
for run in range(num_run):
    # construct the training and testing set
    train_loader, TestPatch, TestLabel, Data_padding = HSIDataset(Data, TrLabel, TsLabel, patchsize, batchsize)

    # construct the network
    model = MViT(
        patch_size=patchsize,
        num_patches=[l-1, 1],
        num_classes=num_class,
        dim=64,
        depth=6,
        heads=4,
        mlp_dim=32,
        dropout=0.1,
        emb_dropout=0.1,
        mode='MViT'
    )
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    BestAcc = 0
    bestloss = 10

    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            # move train data to GPU
            b_x = b_x.cuda()
            b_y = b_y.cuda()

            output = model(b_x)  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            # scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()

            pred_y = test(TestLabel, TestPatch, model=model)
            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
            print('Epoch: ', epoch + 1, '| train loss: %.4f' % loss.data.cpu().numpy(),
                  '| test accuracy: %.2f' % accuracy)
            # save the parameters in network
            if  loss < bestloss:
                torch.save(model.state_dict(), check_dir+'net_params_'+method_name+'_'+str(run)+'.pkl')
                BestAcc = accuracy
                bestloss = loss

        model.train()

    model.load_state_dict(torch.load(check_dir+'net_params_'+method_name+'_'+str(run)+'.pkl'))
    model.eval()

    # 评价指标
    pred_y = test(TestLabel, TestPatch, model=model)
    run_results = metrics(pred_y, TestLabel.numpy(), n_classes=num_class)  # only for test set
    results.append(run_results)
    show_results(run_results, label_values=labels)

    if (run+1) == num_run:
        # show the whole image
        plot_result(Data, patchsize, Data_padding, gt, model=model, color_type='india', path=model_dir)
        save_results_to_csv(results, model_dir + 'accuracy.csv')
    del TestPatch, b_x, b_y, train_loader, model
if num_run > 1:
    show_results(results, label_values=labels, agregated=True)


