
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
from collections import Counter
# 自定义数据集类，继承自Dataset类
class CustomDataset(Dataset):
    def __init__(self, data, targets, num_classes):
        self.data = data.astype(np.float32)
        self.targets = targets.astype(np.int64)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        # 自动one-hot编码
        y = torch.zeros(self.num_classes)
        y[self.targets[idx]] = 1
        return x, y


# 根据dataframe创建创建DataLoader对象
def get_dataloader(df, num_classes, batch_size=64):
    # 将DataFrame转换为NumPy数组
    data = df.drop('Label', axis=1).values  # data[n, n_feature]
    #print(data.shape)
    # Reshape the data to (n, n_feature, 1, 1)
    # reshaped_data.shape will be (n, (n_feature, 1, 1))
    data = data.reshape(data.shape[0], data.shape[1], 1, 1)
    print(data.shape[0])
    print(data.shape[1])
    #print(data.shape)
    targets = df['Label'].values
    #print(targets)
    # 创建自定义数据集对象
    dataset = CustomDataset(data, targets, num_classes)
    # 创建DataLoader对象，使用自定义数据集
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader, dataset, targets


# def get_splited_dataloader(df, num_classes, batch_size, ):
#     data_loader, dataset, targets = get_dataloader(df, num_classes, batch_size)
#     num_targets = torch.tensor(targets, dtype=torch.long)  # 标签数组
#     # 使用 train_test_split 将数据集划分为训练集和测试集
#     train_indices, test_indices = train_test_split(range(len(num_targets)), test_size=0.2, stratify=num_targets)
#     # 创建训练集和测试集的子集
#     train_dataset = Subset(dataset, train_indices)
#     test_dataset = Subset(dataset, test_indices)
#     # 设定每个batch中的目标数量
#     target_counts = torch.tensor([42, 9, 13], dtype=torch.float32)  # doh类别0：类别1：类别2=45:5:14 malicious doh  42:9:13
#     # 计算训练集每个类的样本数
#     class_sample_count = torch.tensor([(num_targets[train_indices] == t).sum() for t in torch.unique(num_targets)])
#
#     # 计算每个类的权重，样本数越少，权重越大
#     class_weights = target_counts / (class_sample_count.float()+ 1e-6)
#
#     # 为训练集中的每个样本分配权重
#     train_sample_weights = torch.tensor([class_weights[t] for t in num_targets[train_indices]])
#
#     # 创建 WeightedRandomSampler
#     train_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights),
#                                           replacement=True)
#     # 创建训练集的 DataLoader，使用 WeightedRandomSampler
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
#
#     # 创建测试集的 DataLoader，不使用 WeightedRandomSampler，通常设置 shuffle=False
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     # 检查训练集和测试集的大小
#     #print(f"训练集大小: {len(train_dataset)}")
#     print(f"测试集大小: {len(test_dataset)}")
#     # 检查训练集的类别分布
#     train_labels = num_targets[train_indices]
#     print("训练集类别分布:")
#     print(torch.tensor([(train_labels == t).sum().item() for t in torch.unique(train_labels)]))
#     # 检查测试集的类别分布
#     test_labels = num_targets[test_indices]
#     print("测试集类别分布:")
#     print(torch.tensor([(test_labels == t).sum().item() for t in torch.unique(test_labels)]))
#     # for batch_idx, (data, target) in enumerate(train_dataloader):
#     #     #print("data:{} target:{}".format(data, target))
#     #     if batch_idx >= 100:  # 仅检查前5个批次
#     #         break
#     #     # target 是独热编码，转换为类别标签以检查分布
#     #     target_classes = torch.argmax(target, dim=1)
#     #     print(target_classes)
#     #     #hashable_elements = [inner_list[0] for inner_list in target_classes.tolist()]
#     #     print(f"批次 {batch_idx + 1} 中的类别分布: {Counter(target_classes.tolist())}")
#     return train_dataloader, test_dataloader


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torch


def get_splited_dataloader(df, num_classes, batch_size):
    # 加载数据集
    data_loader, dataset, targets = get_dataloader(df, num_classes, batch_size)
    num_targets = torch.tensor(targets, dtype=torch.long)  # 标签数组

    # 使用 train_test_split 将数据集划分为训练集和剩余集（验证集+测试集）
    train_indices, remaining_indices = train_test_split(
        range(len(num_targets)), test_size=0.3, stratify=num_targets
    )

    # 将剩余集划分为验证集和测试集（各占10%）
    valid_indices, test_indices = train_test_split(
        remaining_indices, test_size=0.5, stratify=num_targets[remaining_indices]
    )

    # 创建训练集、验证集和测试集的子集
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)

    # 设定每个batch中的目标数量
    target_counts = torch.tensor([42, 9, 13], dtype=torch.float32)  # doh类别0：类别1：类别2=45:5:14 malicious doh  42:9:13
    # 计算训练集每个类的样本数
    class_sample_count = torch.tensor([(num_targets[train_indices] == t).sum() for t in torch.unique(num_targets)])

    # 计算每个类的权重，样本数越少，权重越大
    class_weights = target_counts / (class_sample_count.float() + 1e-6)

    # 为训练集中的每个样本分配权重
    train_sample_weights = torch.tensor([class_weights[t] for t in num_targets[train_indices]])

    # 创建 WeightedRandomSampler
    train_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights),
                                          replacement=True)

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 检查数据集的大小和类别分布
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(valid_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    train_labels = num_targets[train_indices]
    print("训练集类别分布:", torch.tensor([(train_labels == t).sum().item() for t in torch.unique(train_labels)]))

    valid_labels = num_targets[valid_indices]
    print("验证集类别分布:", torch.tensor([(valid_labels == t).sum().item() for t in torch.unique(valid_labels)]))

    test_labels = num_targets[test_indices]
    print("测试集类别分布:", torch.tensor([(test_labels == t).sum().item() for t in torch.unique(test_labels)]))

    return train_dataloader, valid_dataloader, test_dataloader


def doh_dataloader(path, num_classes, batch_size):
    df = pd.read_csv(path)
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    data_len = len(df)
    data_len1 = np.arange(data_len)
    #if num_classes == 2:
    #    df['label'] = df['bin']
    #elif num_classes == 10:
    #    df['label'] = df['mul']
    #else:
    #    print("Error: num_classes must be 2 or 10")
    #    return
    #df = df.drop(["mul", "bin"], axis=1)
    # test
    train_dataloader, test_dataloader, valid_dataloader = get_splited_dataloader(shuffled_df, num_classes, batch_size)
    return train_dataloader, test_dataloader, valid_dataloader,data_len1

if __name__ == '__main__':
    print()
    doh_dataloader('C:/Users/Administrator/Desktop/trafficdet-main/alg/Dataset/data-doh1.csv', 3, 64)