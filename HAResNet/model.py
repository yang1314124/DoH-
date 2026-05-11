import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


# KANLinear类（假设你已经有它的实现）
class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.ReLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
                    )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class CombinedAttention(nn.Module):
    def __init__(self, in_channels, attention_size=8, num_heads=4):
        super(CombinedAttention, self).__init__()

        # 通道注意力部分
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // attention_size, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(in_channels // attention_size, in_channels, bias=True)
        self.sigmoid = nn.Sigmoid()

        # 多头自注意力部分
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = (in_channels // num_heads) ** -0.5

    def forward(self, x):
        b, c, h, w = x.size()

        # 通道注意力部分
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        channel_attention = self.sigmoid(y).view(b, c, 1, 1)

        # 多头自注意力部分
        q = self.query_conv(x).view(b, self.num_heads, c // self.num_heads, -1)
        k = self.key_conv(x).view(b, self.num_heads, c // self.num_heads, -1)
        v = self.value_conv(x).view(b, self.num_heads, c // self.num_heads, -1)

        q = q.permute(0, 1, 3, 2)  # (b, num_heads, hw, c // num_heads)
        k = k.permute(0, 1, 2, 3)  # (b, num_heads, c // num_heads, hw)
        attention = torch.matmul(q, k) * self.scale  # (b, num_heads, hw, hw)
        attention = self.softmax(attention)

        v = v.permute(0, 1, 3, 2)  # (b, num_heads, hw, c // num_heads)
        out = torch.matmul(attention, v)  # (b, num_heads, hw, c // num_heads)
        out = out.permute(0, 1, 3, 2).contiguous()  # (b, num_heads, c // num_heads, hw)
        out = out.view(b, c, h, w)  # 重新调整为输入大小

        # 将多头自注意力的结果应用到通道注意力上
        channel_attention = channel_attention * out

        return channel_attention


class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True, attention=None):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        # conv2 的输入通道数与 conv1 的输出通道数一致
        # conv1 stride=1, conv2 stride=1
        # conv1 stride=2, conv2 stride=1
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.attention = attention
        # 如果输入输出形状不同，需要把输入形状转换成输出形状
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        if self.attention:
            out = self.attention(out)
        # 如果输入输出形状不同，需要把输入形状转换成输出形状
        if not self.same_shape:
            x = self.conv3(x)

        # 残差连接
        # 原始输入（可能经过形状变换）和经过卷积批归一化处理的数据相加，
        # 然后应用 ReLU 激活函数。
        return F.relu(x + out, True)


class KAResNet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(KAResNet, self).__init__()
        attention64 = CombinedAttention(64)
        attention128 = CombinedAttention(128)
        attention256 = CombinedAttention(256)
        self.block1 = nn.Conv2d(in_channel, 64, 1, 1)

        self.block2 = nn.Sequential(
            residual_block(64, 64, attention=attention64),
            residual_block(64, 64, attention=attention64)
        )

        self.block3 = nn.Sequential(
            residual_block(64, 128, False, attention=attention128),
            residual_block(128, 128, attention=attention128)
        )

        self.block4 = nn.Sequential(
            residual_block(128, 256, False, attention=attention256),
            residual_block(256, 256, attention=attention256)
        )

        # 全局池化, 降维
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = KANLinear(256, num_classes)

        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.block1(x)
        # print('block 1 output: {}'.format(x.shape))

        x = self.block2(x)
        # print('block 2 output: {}'.format(x.shape))

        x = self.block3(x)
        # print('block 3 output: {}'.format(x.shape))

        x = self.block4(x)
        # print('block 4 output: {}'.format(x.shape))

        x = self.pooling(x)
        # print('pooling output: {}'.format(x.shape))

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def test_karesnet():
    net = KAResNet(32, 3)
    x = Variable(torch.randn(1, 32, 96, 96))
    y = net(x)
    print(y.shape)

    net = KAResNet(29, 2)
    # x[b, feature_num]
    x = Variable(torch.zeros(3, 29))
    x = x.unsqueeze(-1).unsqueeze(-1)  # 将 x 扩展为 (b, 43, 1, 1) 的四维张量
    y = net(x)
    print(y.shape)
    print(net)
if __name__ == '__main__':
    test_karesnet()

