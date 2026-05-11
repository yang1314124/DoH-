import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64, 256)  # Assuming input image size is 28x28
        self.fc2 = nn.Linear(256, num_classes)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_CNN():
    net = CNN(29,3)
    x = Variable(torch.randn(1, 29, 32, 32))
    y = net(x)
    print(y.shape)

    # net = CNN(29, 3)
    # # x[b, feature_num]
    # x = Variable(torch.zeros(10, 29))
    # x = x.unsqueeze(-1).unsqueeze(-1)  # 将 x 扩展为 (b, 43, 1, 1) 的四维张量
    # y = net(x)
    # print(y.shape)
if __name__ == '__main__':
    test_CNN()