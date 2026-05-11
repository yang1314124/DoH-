
import argparse
import numpy as np
import shap
import torch
from alg.krcnn.model import KAResNet
from alg.krcnn.loader import doh_dataloader
from torch import nn
import time
from sklearn.model_selection import KFold
import torch.optim as optim
from alg.krcnn.loss import CustomMulticlassLoss
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from scipy.stats import randint
from scipy.stats import uniform
import pandas as pd
import random
from tqdm import tqdm
#目前最优0.5 0.9807 Best DR: 0.9737 Best Parameters: {'weight': 0.8}
#Best DR: 0.9081 Best Parameters: {'delta': 2.0}

data_class = {"Doh": 3, "Malicious_DoH": 3}
data_lr = {"Doh": 0.001, "Malicious_DoH": 0.001}
data_feats = {"Doh": 32, "Malicious_DoH": 32}
weight = 1.5
beta = 2.0
delta = 1.5
def fit(dataset, train_loader, test_loader, valid_loader, num_class, data_len1, model_save_path,loss_value , epoch, per_print=100):
    net = KAResNet(data_feats[dataset], num_class)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=data_lr[dataset], weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #criterion = CustomMulticlassLoss(weight, beta)
    criterion = None
    if loss_value == 0:
        criterion = nn.CrossEntropyLoss()
    elif loss_value == 1:
        criterion = CustomMulticlassLoss(weight, beta, delta)
    #criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    start_time = time.time()
    for e in range(epoch):
        net.train()
        losses1, nums1 = [], []
        all_labels1,all_preds1 = [],[]
        for idx, (x, y) in enumerate(train_loader):
            # x[b, feature_num, 1, 1]
            # print(x.shape)
            x = x.to(device)
            y = y.to(device)
            output = net(x)
            output = torch.clamp(output, min=-1e6, max=1e6)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses1.append(loss.item())
            nums1.append(x.size(0))
            if idx % per_print == 0:
                end_time = time.time()
                print('Epoch: {}, idx: {}, loss: {}'.format(e + 1, idx, loss.item()))
        test_avg_loss1 = np.sum(np.multiply(losses1, nums1)) / np.sum(nums1)
        print("训练集平均损失：{:.4f}".format(test_avg_loss1))
        # save model
        # 保存模型参数
        torch.save(net.state_dict(), model_save_path)
        # 验证阶段
        net.eval()
        total_acc, total_dr, total_far = 0.0, 0.0, 0.0
        all_preds = []
        all_labels = []
        losses, nums = [], []
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                y = y.to(device)

                out = net(x)
                out = torch.clamp(out, min=-1e6, max=1e6)
                if torch.isnan(out).any() or torch.isinf(out).any():
                    print(f"Model output contains NaN or Inf in validation")
                loss = criterion(out, y)
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("Loss contains NaN or Inf during validation")
                y_pred = np.argmax(out, axis=1)
                y_true = np.argmax(y, axis=1)
                all_preds.extend(y_pred.cpu().numpy())
                all_labels.extend(y_true.cpu().numpy())

                losses.append(loss.item())
                nums.append(x.size(0))

            matrix = confusion_matrix(all_labels, all_preds)
            print("验证集混淆矩阵:\n", matrix)

            h1 = torch.tensor(matrix, dtype=torch.float32)
            tp = torch.diag(h1)
            fp = torch.sum(h1, dim=0) - tp
            fn = torch.sum(h1, dim=1) - tp
            tn = torch.sum(h1) - (fp + fn + tp)
            DR = tp / (tp + fn)
            FAR = fp / (fp + tn)

            total_tp = torch.sum(tp).item()
            total_fn = torch.sum(fn).item()
            total_fp = torch.sum(fp).item()
            total_tn = torch.sum(tn).item()
            total_dr = total_tp / (total_tp + total_fn)
            total_far = total_fp / (total_fp + total_tn)
            total_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
            test_avg_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            print("验证集平均损失：{:.4f}".format(test_avg_loss))
            print(f"\nOverall Validation Results - Epoch {e + 1}:")
            print(f"  Total DR (Recall): {total_dr:.4f}")
            print(f"  Total FAR: {total_far:.4f}")
            print(f"  Total ACC: {total_acc:.4f}")

        # 测试阶段 (只在最后一次迭代中进行测试)
        if e == epoch - 1:
            print("\nStarting final evaluation on test set...")
            all_preds = []
            all_labels = []
            # criterion = CustomMulticlassLoss(weight, beta)
            criterion = None
            if loss_value == 0:
                criterion = nn.CrossEntropyLoss()
            elif loss_value == 1:
                criterion = CustomMulticlassLoss(weight, beta, delta)
            criterion = criterion.to(device)
            # 在不计算梯度的情况下进行测试（节省内存，加速计算）
            with torch.no_grad():
                # 遍历测试数据加载器中的每个批次
                for x, y in test_loader:
                    # 将输入数据和标签移动到指定的设备上（如GPU）
                    x = x.to(device)
                    y = y.to(device)

                    # 通过神经网络进行预测
                    out = net(x)
                    out = torch.clamp(out, min=-1e6, max=1e6)
                    # 计算预测与真实标签之间的损失
                    loss = criterion(out, y)

                    # 获取预测结果和真实标签中最大值对应的类别索引
                    y_pred = np.argmax(out, axis=1)
                    y_true = np.argmax(y, axis=1)

                    # 将预测结果和真实标签从Tensor转换为NumPy数组，并添加到对应的列表中
                    all_preds.extend(y_pred.cpu().numpy())
                    all_labels.extend(y_true.cpu().numpy())

                    # 记录当前批次的损失和样本数
                    losses.append(loss.item())
                    nums.append(x.size(0))

                # 在所有测试数据上计算混淆矩阵
                matrix = confusion_matrix(all_labels, all_preds)
                print(matrix)

                # 将混淆矩阵转换为Tensor，并进行后续处理
                h = torch.tensor(matrix)
                # print(h)
                h1 = h.float()  # 转换为浮点数类型以便进行数学运算

                # 计算混淆矩阵中的真正例（TP）、假正例（FP）、真反例（TN）和假反例（FN）
                tp = torch.diag(h1)  # 对角线元素为真正例
                fp = torch.sum(h1, dim=0) - tp  # 列总和减去真正例得到假正例
                # print(torch.sum(h1, dim=0))
                fn = torch.sum(h1, dim=1) - tp  # 行总和减去真正例得到假反例
                # print(torch.sum(h1, dim=1))
                tn = torch.sum(h1) - (fp + fn + tp)  # 总和减去其他三类得到真反例
                DR = tp / (tp + fn)  # Detection Rate (Recall)
                FAR = fp / (fp + tn)  # False Acceptance Rate
                # 打印各类别的统计指标
                # #print("TP:{} TN:{} FP:{} FN:{}".format(tp.sum(), tn.sum(), fp.sum(), fn.sum()))
                # print(torch.diag(h1).sum())  # 打印所有真正例的总和
                # print(h1.sum())  # 打印混淆矩阵中所有元素的总和
                # 计算总体的 DR, FAR, 和 ACC
                total_tp = torch.sum(tp).item()
                total_fn = torch.sum(fn).item()
                total_fp = torch.sum(fp).item()
                total_tn = torch.sum(tn).item()

                total_dr = total_tp / (total_tp + total_fn)
                total_far = total_fp / (total_fp + total_tn)
                total_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
                # 遍历每个类别，打印其详细的性能指标（检测率DR和误报率FAR）
                for i in range(len(tp)):
                    print(f"Class {i}:")
                    print(f" TP: {tp[i].item()}")  # 真正例数量
                    print(f" FP: {fp[i].item()}")  # 假正例数量
                    print(f" FN: {fn[i].item()}")  # 假反例数量
                    print(f" TN: {tn[i].item()}")  # 真反例数量
                    print(f" DR(recall):{DR[i].item():.4f}")  # 打印检测率
                    print(f" FAR:{FAR[i].item():.4f}")  # 打印误报率

                # 输出总体的结果
                print(f"\nOverall Results:")
                print(f"  Total DR (Recall): {total_dr:.4f}")
                print(f"  Total FAR: {total_far:.4f}")
                print(f"  Total ACC: {total_acc:.4f}")
                # print(np.sum(nums))
                test_avg_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
                print("前值{} 后值{}".format(np.sum(np.multiply(losses, nums)), np.sum(nums)))
                print('Epoch: {}, 测试集平均损失：{}'.format(e + 1, test_avg_loss))
                print('\n')

    print("\nTraining, validation, and testing completed.")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET = ['Doh', 'Malicious_DoH']
    isbinary = [1, 0]
    loss_func = ['old', 'new','FocalLoss']
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',
                   help='Experimental dataset.',
                   type=str,
                   default='Doh',
                   choices=DATASET)
    p.add_argument('--binary',
                   help='Perform binary or muticlass task',
                   type=int,
                   choices=isbinary,
                   default=1)
    p.add_argument('--loss_func',
                   help='new or old loss',
                   type=str,
                   default='new',
                   choices=loss_func)
    p.add_argument('--epoch',
                   help='choose your epoch number',
                   type=int,
                   default=10)
    args = p.parse_args()

    dataset = args.dataset
    if args.binary:
        num_class = 3
    else:
        num_class = data_class[dataset]

    print('num_class: {}'.format(num_class))
    if args.loss_func == 'old':
        loss_value = 0
    elif args.loss_func == 'new':
        loss_value = 1
        print("weight= {} beta= {} delta= {}".format(weight, beta, delta))
    epoch = args.epoch
    print('loss_value: {}'.format(loss_value))
    data_path1 = 'alg/Dataset/dataset_doh3.csv'
    data_path2 = 'alg/Dataset/dataset_malicious_doh3.csv'
    model_save_path1 = 'alg/model/krcnn-doh.pth'
    model_save_path2 = 'alg/model/krcnn-malicious_doh.pth'
    if args.dataset == 'Doh':
        train_loader, test_loader, valid_loader, data_len1 = doh_dataloader(data_path1, num_class, batch_size=64)
        # Training and testing
        fit(dataset, train_loader, test_loader, valid_loader, num_class, data_len1, model_save_path1, loss_value, epoch=epoch, per_print=100)
    else:
        train_loader, test_loader, valid_loader, data_len1 = doh_dataloader(data_path2, num_class, batch_size=64)
        # Training and testing
        fit(dataset, train_loader, test_loader, valid_loader, num_class, data_len1, model_save_path2, loss_value, epoch=epoch, per_print=100)