import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.25):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化LSTM的隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM层
        lstm_out, _ = self.lstm(x, (h0, c0))

        # 取LSTM最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 添加dropout防止过拟合
        lstm_out = self.dropout(lstm_out)

        # 全连接层
        fc1_out = F.relu(self.fc1(lstm_out))
        fc1_out = self.dropout(fc1_out)
        output = self.fc2(fc1_out)

        return output

# 示例使用
input_size = 32  # 假设网络流量数据有10个特征
hidden_size = 64  # LSTM隐藏层大小
num_layers = 2  # LSTM层数
output_size = 3  # 假设是二分类问题，例如正常流量和异常流量


def test_lstm():
    model = LSTMNet(input_size, hidden_size, num_layers, output_size)
    x = Variable(torch.randn(1, 29))
    y = model(x)
    print(y.shape)
if __name__ == '__main__':
    test_lstm()