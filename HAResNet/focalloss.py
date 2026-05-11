import torch
from torch import nn
class FocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=1, epsilon=1.e-9, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha, device=device)
        else:
            self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        num_labels = input.size(-1)
        idx = target.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key[:, 0] = 0  # ignore 0 index.
        logits = torch.softmax(input, dim=-1)
        loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss = loss.sum(1)
        return loss.mean()


if __name__ == '__main__':
    y_pred = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1], [0.1, 0.2, 2.0]], dtype=torch.float32)
    y_true = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)  # one-hot 编码的标签
    loss = FocalLoss(alpha=[0.3, 0.4, 0.3])
    #input = torch.randn(3, 5, requires_grad=True)
    #print(input)
    target = torch.empty(3, dtype=torch.long).random_(5)
    print(target)
    output = loss(y_pred, y_true)
    print(type(output))
    print(output)
    output.backward()
