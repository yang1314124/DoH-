import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# 定义 KANLayer 类，继承自 nn.Module
class KANLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, attention_size=16):
        super(KANLayer, self).__init__()
        # 定义卷积层，使用 2D 卷积，输入通道为 in_channels，输出通道为 out_channels，卷积核大小为 kernel_size，填充使输出大小不变
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        # 定义批量归一化层，用于加速训练并提高稳定性
        self.bn = nn.BatchNorm2d(out_channels)
        # 定义注意力机制，使用两个 1x1 的卷积层和 ReLU 激活函数
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, attention_size, kernel_size=1),  # 第一个 1x1 卷积，减少通道数
            nn.ReLU(inplace=True),  # 使用 ReLU 激活函数
            nn.Conv2d(attention_size, out_channels, kernel_size=1),  # 第二个 1x1 卷积，恢复通道数
            nn.Softmax(dim=1)  # 使用 Softmax 在通道维度上归一化
        )

    def forward(self, x):
        # 先进行卷积和批量归一化
        conv_out = self.bn(self.conv(x))
        # 计算注意力权重
        attention_weights = self.attention(conv_out)
        # 将卷积结果与注意力权重相乘，以获得加权后的输出
        out = conv_out * attention_weights
        return out

# 定义 ResidualBlock 类，继承自 nn.Module
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, attention_size=16):
        super(ResidualBlock, self).__init__()
        # 定义第一个 KANLayer 层
        self.kan1 = KANLayer(in_channels, out_channels, kernel_size, attention_size)
        # 定义第二个 KANLayer 层
        self.kan2 = KANLayer(out_channels, out_channels, kernel_size, attention_size)
        # 定义一个 identity 路径，如果输入和输出通道数不同，则使用 1x1 卷积调整通道数，否则直接传递输入
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        # 定义批量归一化层
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 获取输入的 identity 分支
        identity = self.identity(x)
        # 通过第一个 KANLayer 层并进行 ReLU 激活
        out = F.relu(self.kan1(x))
        # 通过第二个 KANLayer 层
        out = self.kan2(out)
        # 将 identity 和主路径的输出相加，实现残差连接
        out += identity
        # 最终再经过一个 ReLU 激活函数和批量归一化
        return F.relu(self.bn(out))

# 定义 KRCNN 类，继承自 nn.Module
class KRCNN(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(KRCNN, self).__init__()
        # 定义三个残差块，每个块中的通道数分别为 32, 64, 128
        self.layer1 = ResidualBlock(in_channel, 32, 3, 16)
        self.layer2 = ResidualBlock(32, 64, 3, 32)
        self.layer3 = ResidualBlock(64, 128, 3, 64)
        # 定义最大池化层，核大小为 1x1（此处池化作用不大，可能为代码残留）
        self.pool = nn.MaxPool2d(1, 1)
        # 定义全局平均池化层，输出大小为 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义全连接层，用于分类，包含两个线性层和 ReLU 激活函数，以及 Dropout 防止过拟合
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # Dropout，随机丢弃 50% 的神经元
            nn.Linear(128, 64),  # 线性层，将特征从 128 维降到 64 维
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Dropout(0.5),  # 再次使用 Dropout
            nn.Linear(64, num_classes)  # 线性层，将特征从 64 维降到 num_classes 维（类别数）
        )

    def forward(self, x):
        # 通过第一个残差块
        x = self.pool(self.layer1(x))
        # 通过第二个残差块
        x = self.pool(self.layer2(x))
        # 通过第三个残差块
        x = self.pool(self.layer3(x))
        # 进行全局平均池化
        x = self.global_avg_pool(x)
        # 展平特征图为向量形式，准备输入全连接层
        x = x.view(x.size(0), -1)
        # 通过全连接层进行分类
        x = self.fc(x)
        return x
