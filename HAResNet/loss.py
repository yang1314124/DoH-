import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomMulticlassLoss(nn.Module):
    def __init__(self, weight=0.5, beta=2.0, delta=1.0):
        """
        初始化自定义多分类损失函数的超参数。

        Args:
            weight (float): 权重系数，用于调整第一部分损失的影响。
            beta (float): 调整指数项的系数，控制损失的敏感度。
            delta (float): 调整 tanh 函数的系数。
        """
        super(CustomMulticlassLoss, self).__init__()
        self.weight = weight
        self.beta = beta
        self.delta = delta

    def forward(self, y_pred, y_true):
        """
        计算损失。

        Args:
            y_pred (torch.Tensor): 模型的预测输出，形状为 (batch_size, num_classes)。
            y_true (torch.Tensor): 真实标签，形状为 (batch_size, num_classes)，通常以独热编码形式提供。

        Returns:
            torch.Tensor: 批次中所有样本的平均损失。
        """
        # 确保 y_pred 经过 softmax
        y_pred = F.softmax(y_pred, dim=1)
        # 初始化损失为 0
        batch_size = y_true.shape[0]
        loss = torch.zeros(batch_size, device=y_pred.device)
        epsilon = 1e-9  # 为了防止数值不稳定，添加一个小常数

        # 对每个类别分别计算损失
        for i in range(y_pred.shape[1]):
            # 计算绝对误差
            e = torch.abs(y_true[:, i] - y_pred[:, i])

            #计算每个类别的损失部分
            loss_first_part = -self.weight * y_true[:, i] * (torch.tanh(self.delta * e) ** self.beta) * torch.log(
                y_pred[:, i] + epsilon)
            loss_second_part = -(1 - y_true[:, i]) * (1 - torch.tanh(self.delta * e)) ** self.beta * torch.log(
                1 - y_pred[:, i] + epsilon)
            # 累加到总损失中
            loss = loss + (loss_first_part + loss_second_part)

        # 返回批次中所有样本的平均损失
        return torch.mean(loss)



def cross_entropy_loss(y_pred, y_true):
    # 初始化 PyTorch 的交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # y_true 需要是类别索引而不是 one-hot 编码
    y_true_indices = torch.argmax(y_true, dim=1)

    # 计算交叉熵损失
    loss = criterion(y_pred, y_true_indices)
    return loss


def test():
    # 示例输入
    y_pred = torch.randn(64, 3,requires_grad=True)  # 假设有10个样本和3个类别
    y_true = torch.eye(3)[torch.randint(0, 3, (64,))]  # 假设真实标签是独热编码
    # print(y_pred)
    # print(y_true)
    # weight = torch.tensor(0.5, dtype=torch.float32)
    # beta = torch.tensor(2.0, dtype=torch.float32)
    weight = 0.5
    beta = 2.0
    delta = 1.0
    # 计算损失
    loss_fn = CustomMulticlassLoss(weight, beta, delta)
    loss = loss_fn(y_pred, y_true)
    loss.backward()  # 计算梯度
    #print("Output Gradients:", output.grad)
    # print(type(loss))
    # print(loss)
    #loss.backward()
    loss_value2 = cross_entropy_loss(y_pred, y_true)
    loss_value2.backward()
    print("new Loss:{} CE loss:{}".format(loss, loss_value2.item()))
if __name__ == '__main__':
    test()
