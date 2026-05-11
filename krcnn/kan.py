import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,  # 输入特征的数量
            out_features,  # 输出特征的数量
            grid_size=5,  # 栅格的大小，默认为5
            spline_order=3,  # 样条曲线的阶数，默认为3
            scale_noise=0.1,  # 噪声缩放因子，默认为0.1
            scale_base=1.0,  # 基础权重的缩放因子，默认为1.0
            scale_spline=1.0,  # 样条曲线的缩放因子，默认为1.0
            enable_standalone_scale_spline=True,  # 是否启用独立的样条曲线缩放，默认为True
            base_activation=torch.nn.SiLU,  # 基础激活函数，默认为SiLU激活函数
            grid_eps=0.02,  # 网格计算中的 epsilon 值，用于初始化样条权重，默认为0.02
            grid_range=[-1, 1],  # 网格的范围，默认为[-1, 1]
    ):
        super(KANLinear, self).__init__()  # 调用父类的构造函数，进行初始化
        self.in_features = in_features  # 保存输入特征的数量
        self.out_features = out_features  # 保存输出特征的数量
        self.grid_size = grid_size  # 保存栅格的大小
        self.spline_order = spline_order  # 保存样条曲线的阶数

        # 计算网格步长 h，并根据网格大小和范围创建网格张量 grid
        h = (grid_range[1] - grid_range[0]) / grid_size  # 计算网格步长
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)  # 扩展网格的维度，使其与输入特征数匹配
            .contiguous()  # 确保网格张量在内存中是连续的
        )
        self.register_buffer("grid", grid)  # 将 grid 注册为模型的 buffer，以确保其在训练中不会被更新

        # 初始化基础权重和样条权重
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))  # 定义基础权重
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)  # 定义样条权重
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)  # 如果启用独立的样条缩放，定义对应的缩放参数
            )

        # 保存噪声和缩放相关的参数
        self.scale_noise = scale_noise  # 保存噪声缩放因子
        self.scale_base = scale_base  # 保存基础权重的缩放因子
        self.scale_spline = scale_spline  # 保存样条曲线的缩放因子
        self.enable_standalone_scale_spline = enable_standalone_scale_spline  # 保存是否启用独立的样条缩放标志
        self.base_activation = base_activation()  # 初始化基础激活函数
        self.grid_eps = grid_eps  # 保存 epsilon 值，用于样条权重初始化

        # 初始化模型的权重
        self.reset_parameters()  # 调用权重初始化方法

    def reset_parameters(self):
        # 使用 Kaiming 均匀分布初始化基础权重
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        # 在不计算梯度的上下文中初始化样条权重
        with torch.no_grad():
            # 生成一个随机噪声张量，用于初始化样条权重
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)  # 生成随机数张量
                            - 1 / 2  # 将随机数中心移动到0
                    )
                    * self.scale_noise  # 乘以噪声缩放因子
                    / self.grid_size  # 将噪声缩放与栅格大小成反比
            )

            # 初始化样条权重
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)  # 如果启用独立的样条缩放，则不应用缩放因子
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],  # 提取与样条阶数对应的网格部分
                    noise,  # 使用噪声张量作为插值的目标
                )
            )

            # 如果启用独立的样条缩放参数，则初始化该参数
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler,
                                               a=math.sqrt(5) * self.scale_spline)  # 使用 Kaiming 均匀分布初始化样条缩放参数

    def b_splines(self, x: torch.Tensor):
        """
        计算给定输入张量的 B 样条基函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: B 样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        # 检查输入张量的维度是否符合要求，即必须是二维张量且特征数必须与 in_features 匹配
        assert x.dim() == 2 and x.size(1) == self.in_features

        # 获取已注册的网格张量 grid，该张量的形状为 (in_features, grid_size + 2 * spline_order + 1)
        grid: torch.Tensor = self.grid

        # 在输入张量 x 的最后一维增加一维，以便进行后续的计算
        x = x.unsqueeze(-1)

        # 计算基础的 B 样条基函数，检查 x 是否落在 grid 的各区间内，并转化为浮点型
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # 通过递归的方式计算更高阶的 B 样条基函数
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        # 确保最终计算出的 B 样条基函数张量具有预期的大小
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )

        # 返回计算出的 B 样条基函数张量，并确保其在内存中是连续的
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算插值曲线的系数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。

        返回:
            torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        # 检查输入张量的维度和形状是否符合要求
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # 计算 B 样条基函数矩阵 A，并进行转置，使其形状为 (in_features, batch_size, grid_size + spline_order)
        A = self.b_splines(x).transpose(0, 1)

        # 转置输出张量 y，使其形状为 (in_features, batch_size, out_features)
        B = y.transpose(0, 1)

        # 通过最小二乘法求解 A * coef = B，得到系数矩阵
        solution = torch.linalg.lstsq(A, B).solution  # 形状为 (in_features, grid_size + spline_order, out_features)

        # 转置系数矩阵 solution 使其形状为 (out_features, in_features, grid_size + spline_order)
        result = solution.permute(2, 0, 1)

        # 确保结果张量的大小符合预期
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )

        # 返回计算出的系数张量，并确保其在内存中是连续的
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        计算缩放后的样条权重。

        返回:
            torch.Tensor: 缩放后的样条权重。如果启用了独立的样条缩放参数，
            则样条权重会乘以 `spline_scaler`，否则不进行缩放。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)  # 如果启用了独立的样条缩放参数，则将其应用到样条权重上
            if self.enable_standalone_scale_spline
            else 1.0  # 否则，不进行缩放
        )

    def forward(self, x: torch.Tensor):
        """
        执行前向传播，计算给定输入的输出。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # 确保输入的最后一维的大小与 in_features 匹配
        assert x.size(-1) == self.in_features

        # 保存输入张量的原始形状
        original_shape = x.shape

        # 将输入张量重塑为二维形状，以便进行线性运算
        x = x.reshape(-1, self.in_features)

        # 计算基础输出，先通过激活函数，再通过线性变换
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 计算样条插值输出
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),  # 将 B 样条基函数的结果展平成二维
            self.scaled_spline_weight.view(self.out_features, -1),  # 将缩放后的样条权重展平成二维
        )

        # 将基础输出和样条插值输出相加，得到最终输出
        output = base_output + spline_output

        # 将输出张量重塑为与原始输入形状匹配
        output = output.reshape(*original_shape[:-1], self.out_features)

        return output  # 返回最终输出

    @torch.no_grad()  # 指定该方法在执行时不会计算梯度
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        根据输入数据动态更新样条插值的网格，并重新计算样条权重。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            margin (float): 更新网格时的边距，默认为 0.01。
        """
        # 确保输入张量的维度和大小符合要求
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)  # 获取批次大小

        # 计算 B 样条基函数并调整维度顺序
        splines = self.b_splines(x)  # 计算 B 样条基函数 (batch, in_features, coeff)
        splines = splines.permute(1, 0, 2)  # 变换维度顺序为 (in_features, batch_size, coeff)

        # 获取当前的样条权重并调整维度顺序
        orig_coeff = self.scaled_spline_weight  # 获取缩放后的样条权重 (out_features, in_features, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # 变换维度顺序为 (in_features, coeff, out_features)

        # 计算未简化的样条插值输出
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # 执行批量矩阵乘法 (in_features, batch_size, out_features)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0,
                                                                  2)  # 变换维度顺序为 (batch_size, in_features, out_features)

        # 对每个通道分别排序以收集数据分布
        x_sorted = torch.sort(x, dim=0)[0]  # 对输入张量按特征维度排序

        # 计算自适应网格
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # 计算均匀网格步长并生成均匀网格
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        # 组合自适应网格和均匀网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # 扩展网格以适应样条插值的阶数
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

        # 更新网格并重新计算样条权重
        self.grid.copy_(grid.T)  # 更新 grid 缓存
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))  # 根据新的网格重新计算样条权重

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算正则化损失。

        这是一种模拟原始论文中提到的 L1 正则化的方法，因为原始的 L1 正则化要求计算
        扩展后的中间张量 (batch, in_features, out_features) 的绝对值和熵，
        如果我们想要实现一个内存高效的实现，F.linear 函数隐藏了这个中间张量的计算过程。

        当前的 L1 正则化通过样条权重的平均绝对值来计算。
        作者的实现还包括了基于样本的正则化。

        参数:
            regularize_activation (float): 正则化激活项的系数，默认值为1.0。
            regularize_entropy (float): 正则化熵项的系数，默认值为1.0。

        返回:
            torch.Tensor: 计算得到的正则化损失值。
        """
        # 计算样条权重在最后一维上的绝对值的平均值
        l1_fake = self.spline_weight.abs().mean(-1)

        # 计算激活正则化损失（伪L1正则化损失）
        regularization_loss_activation = l1_fake.sum()

        # 计算概率分布 p
        p = l1_fake / regularization_loss_activation

        # 计算熵正则化损失
        regularization_loss_entropy = -torch.sum(p * p.log())

        # 返回总的正则化损失
        return (
                regularize_activation * regularization_loss_activation  # 激活正则化损失乘以系数
                + regularize_entropy * regularization_loss_entropy  # 熵正则化损失乘以系数
        )


class KAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,  # 隐藏层的大小列表
            grid_size=5,  # 栅格大小，默认为5
            spline_order=3,  # 样条曲线阶数，默认为3
            scale_noise=0.1,  # 噪声缩放因子，默认为0.1
            scale_base=1.0,  # 基础权重的缩放因子，默认为1.0
            scale_spline=1.0,  # 样条曲线的缩放因子，默认为1.0
            base_activation=torch.nn.SiLU,  # 基础激活函数，默认为SiLU
            grid_eps=0.02,  # 栅格计算中的 epsilon 值，默认为0.02
            grid_range=[-1, 1],  # 栅格范围，默认为[-1, 1]
    ):
        super(KAN, self).__init__()  # 调用父类的构造函数

        # 保存类的属性
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 创建一个 ModuleList 来存储 KANLinear 层
        self.layers = torch.nn.ModuleList()

        # 根据给定的隐藏层大小，逐层初始化 KANLinear
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features=in_features,  # 当前层的输入特征数
                    out_features=out_features,  # 当前层的输出特征数
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        """
        执行前向传播。

        参数:
            x (torch.Tensor): 输入张量。
            update_grid (bool): 是否在前向传播过程中更新网格，默认为 False。

        返回:
            torch.Tensor: 前向传播的结果张量。
        """
        # 依次通过每一层 KANLinear 层
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)  # 如果指定，则更新网格
            x = layer(x)  # 通过该层进行计算
        return x  # 返回最终输出

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算所有层的正则化损失的总和。

        参数:
            regularize_activation (float): 正则化激活项的系数，默认值为1.0。
            regularize_entropy (float): 正则化熵项的系数，默认值为1.0。

        返回:
            torch.Tensor: 所有层正则化损失的总和。
        """
        # 遍历所有层，计算正则化损失的总和
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )