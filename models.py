import torch
import torch.nn as nn

# --- Discriminators (for Classic QGAN variants) ---


class Discriminator(nn.Module):
    """经典QGAN的全连接判别器"""

    def __init__(self, image_size=8):
        super().__init__()
        input_features = image_size * image_size
        self.model = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 确保输入是扁平化的
        x = x.view(x.size(0), -1)
        return self.model(x)


class ImprovedDiscriminator(nn.Module):
    """带有LeakyReLU和Dropout的改进判别器"""

    def __init__(self, image_size=8, dropout_rate=0.3):
        super().__init__()
        input_features = image_size * image_size
        self.model = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 确保输入是扁平化的
        x = x.view(x.size(0), -1)
        return self.model(x)


# --- Critics (for WGAN-GP variants) ---


class Critic(nn.Module):
    """WGAN-GP的基础评论家网络"""

    def __init__(self, image_size=8, dropout_rate=0.3):
        super().__init__()
        input_features = image_size * image_size
        self.model = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),  # WGAN输出层无激活函数
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, 0.8)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x):
        # 确保输入是扁平化的
        x = x.view(x.size(0), -1)
        return self.model(x)


class MinibatchDiscrimination(nn.Module):
    """Minibatch Discrimination层"""

    def __init__(self, in_features, out_features, intermediate_features=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.intermediate_features = intermediate_features
        # 可学习的变换张量 T
        self.T = nn.Parameter(
            torch.Tensor(in_features, out_features * intermediate_features)
        )
        nn.init.normal_(self.T, 0, 1)  # 使用正态分布初始化

    def forward(self, x):
        batch_size = x.size(0)
        if batch_size <= 1:
            zeros = torch.zeros(batch_size, self.out_features, device=x.device)
            return torch.cat([x, zeros], dim=1)

        M = x.mm(self.T)
        M = M.view(batch_size, self.out_features, self.intermediate_features)

        M_expanded_1 = M.unsqueeze(1).expand(
            batch_size, batch_size, self.out_features, self.intermediate_features
        )
        M_expanded_2 = M.unsqueeze(0).expand(
            batch_size, batch_size, self.out_features, self.intermediate_features
        )
        l1_dist = torch.sum(torch.abs(M_expanded_1 - M_expanded_2), dim=3)
        similarity = torch.exp(-l1_dist)

        mask = 1.0 - torch.eye(batch_size, device=x.device).unsqueeze(-1).expand_as(
            similarity
        )
        o_b = torch.sum(similarity * mask, dim=1)
        combined = torch.cat([x, o_b], dim=1)
        return combined


class CriticWithMinibatchDiscrimination(nn.Module):
    """带有Minibatch Discrimination的WGAN-GP评论家网络"""

    def __init__(
        self,
        image_size=8,
        dropout_rate=0.3,
        mb_in_features=32,  # MBD层的输入特征数，应与features模块输出一致
        mb_out_features=5,
        mb_intermediate_features=16,
    ):
        super().__init__()
        input_features = image_size * image_size
        self.mb_out_features = mb_out_features

        # 特征提取部分
        self.features = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(64, mb_in_features),  # 输出维度为MBD的输入维度
            nn.LayerNorm(mb_in_features),
            nn.LeakyReLU(0.2),
        )

        # Minibatch Discrimination层
        self.minibatch_discrimination = MinibatchDiscrimination(
            in_features=mb_in_features,
            out_features=self.mb_out_features,
            intermediate_features=mb_intermediate_features,
        )

        # 最终线性层
        self.final_layer = nn.Linear(mb_in_features + self.mb_out_features, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, 0.8)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 确保输入扁平化
        features_out = self.features(x)
        mb_out = self.minibatch_discrimination(features_out)
        output = self.final_layer(mb_out)
        return output


# --- Quantum Generators ---


class PatchQuantumGenerator(nn.Module):
    """分块量子生成器基类 (由特定变体继承和修改)"""

    def __init__(
        self,
        n_qubits,
        n_a_qubits,
        q_depth,
        n_generators,
        device,
        partial_measure,  # 量子测量/后处理函数
        q_delta=1,
        param_init_func=None,  # 参数初始化函数
        post_process_net=None,  # 经典后处理网络
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_a_qubits = n_a_qubits
        self.q_depth = q_depth
        self.n_generators = n_generators
        self.device = device
        self.partial_measure = partial_measure
        self.post_process_net = post_process_net

        if param_init_func is None:
            # 默认初始化 (经典QGAN)
            param_init_func = lambda: q_delta * torch.rand(q_depth * n_qubits)

        self.q_params = nn.ParameterList([
            nn.Parameter(param_init_func(), requires_grad=True)
            for _ in range(n_generators)
        ])

        self.patch_size = 2 ** (self.n_qubits - self.n_a_qubits)
        self.total_patch_output_size = self.patch_size * self.n_generators

    def forward(self, x):
        # x 是输入的噪声批次, shape: (batch_size, n_qubits)
        batch_size = x.size(0)

        # 创建张量以"捕获"来自所有子生成器的拼接输出
        all_patches_batch = torch.Tensor(batch_size, 0).to(self.device)

        # 遍历所有子生成器
        for params in self.q_params:
            # 创建张量以"捕获"来自单个子生成器的批量patch
            current_patches_batch = torch.Tensor(0, self.patch_size).to(self.device)
            # 对批次中的每个噪声样本应用量子电路
            for noise_sample in x:  # noise_sample shape: (n_qubits,)
                # partial_measure 需要单个噪声向量和参数
                q_out = (
                    self.partial_measure(noise_sample, params)
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )  # shape: (1, patch_size)
                current_patches_batch = torch.cat(
                    (current_patches_batch, q_out), dim=0
                )  # shape: (current_batch_size, patch_size)

            # 将当前子生成器的patch批次拼接到总输出
            all_patches_batch = torch.cat(
                (all_patches_batch, current_patches_batch), dim=1
            )  # shape: (batch_size, n_generators * patch_size)

        # 如果有后处理网络，则应用
        if self.post_process_net:
            final_output = self.post_process_net(all_patches_batch)
        else:
            final_output = all_patches_batch

        # 确保输出形状为 (batch_size, flattened_image_size)
        return final_output.view(batch_size, -1)


# --- Specific Generator Configurations ---


def create_classic_qgan_generator(
    n_qubits,
    n_a_qubits,
    q_depth,
    n_generators,
    device,
    partial_measure_classic,
    q_delta=1,
):
    """创建经典QGAN的生成器实例"""
    # 经典QGAN无特定后处理网络
    return PatchQuantumGenerator(
        n_qubits,
        n_a_qubits,
        q_depth,
        n_generators,
        device,
        partial_measure_classic,
        q_delta,
    )


def create_wgan_qgan_generator(
    n_qubits,
    n_a_qubits,
    q_depth,
    n_generators,
    device,
    partial_measure_wgan,
    q_delta=0.1,
    final_image_size=64,
):
    """创建WGAN-GP QGAN的生成器实例 (带后处理)"""
    # WGAN-GP 参数初始化
    param_init_wgan = lambda: q_delta * (2 * torch.rand(q_depth * n_qubits) - 1)

    # WGAN-GP 后处理网络
    total_patch_size = (2 ** (n_qubits - n_a_qubits)) * n_generators
    post_process_wgan = nn.Sequential(
        nn.Linear(total_patch_size, 128),
        nn.LayerNorm(128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, final_image_size),
        nn.Tanh(),  # 输出范围[-1,1]
    )

    return PatchQuantumGenerator(
        n_qubits,
        n_a_qubits,
        q_depth,
        n_generators,
        device,
        partial_measure_wgan,
        q_delta,
        param_init_func=param_init_wgan,
        post_process_net=post_process_wgan.to(device),  # 确保网络在正确设备上
    )


# WGAN-GP with MBD 通常使用与 WGAN-GP 相同的生成器结构
create_wgan_mbd_qgan_generator = create_wgan_qgan_generator
