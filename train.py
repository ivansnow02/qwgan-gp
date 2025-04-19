import argparse
import math
import os
import random
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 导入本地模块
from datasets import DigitsDataset
from models import (
    Critic,
    CriticWithMinibatchDiscrimination,
    Discriminator,
    ImprovedDiscriminator,
    create_classic_qgan_generator,
    create_wgan_mbd_qgan_generator,
    create_wgan_qgan_generator,
)
from quantum_circuits import setup_classic_quantum_circuit, setup_wgan_quantum_circuit
from quantum_circuits import (
    setup_wgan_mbd_quantum_circuit as setup_wgan_quantum_circuit_mbd,
)

# 导入数据集源
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    print("错误：需要 'ucimlrepo' 包来加载数据集。请运行 'pip install ucimlrepo'")
    exit()

# --- 辅助函数 ---


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """计算WGAN-GP的梯度惩罚项"""
    batch_size = real_samples.size(0)
    # 随机权重项，用于两个样本的线性组合
    alpha = torch.rand((batch_size, 1), device=device)
    # 确保alpha的形状与样本匹配以进行广播
    alpha = alpha.expand(batch_size, real_samples.nelement() // batch_size).contiguous()
    alpha = alpha.view(real_samples.size())

    # 获取真实样本和生成样本间的线性插值
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )

    # 计算critic对插值样本的评分
    d_interpolates = critic(interpolates)

    # 创建全1张量以计算梯度
    fake_output_shape = (batch_size, 1)  # Critic 输出通常是 (batch_size, 1)
    fake = torch.ones(fake_output_shape, requires_grad=False, device=device)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # 计算梯度的范数
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def save_generated_images(images, filename, image_size=8, normalize_range=(-1, 1)):
    """保存生成的图像到文件"""
    fig = plt.figure(figsize=(8, 2))
    num_images = min(len(images), 8)  # 最多显示8张

    min_val, max_val = normalize_range

    for i in range(num_images):
        image = images[i]
        ax = plt.subplot(1, num_images, i + 1)
        plt.axis("off")
        # 将图像数据转换回 [0, 1] 范围以正确显示
        img_display = (
            image.reshape(image_size, image_size).cpu().detach().numpy() - min_val
        ) / (max_val - min_val)
        plt.imshow(img_display, cmap="gray", vmin=0, vmax=1)

    plt.savefig(filename)
    plt.close(fig)
    return fig


# --- 训练函数 ---


def train(config):
    """
    训练量子GAN模型

    Args:
        config (dict): 包含所有训练参数和模型选择的配置字典
    """
    # 设置随机种子
    set_seed(config.get("seed", 42))

    # 设置设备
    if torch.cuda.is_available() and config.get("use_gpu", True):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(f"使用GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("使用CPU")

    # 创建TensorBoard日志目录
    run_name = config.get(
        "run_name", f"{config['gan_type']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")

    # 加载数据集
    print("加载数据集...")
    try:
        optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)
        X = optical_recognition_of_handwritten_digits.data.features
        y = optical_recognition_of_handwritten_digits.data.targets
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    # 根据GAN类型确定归一化范围
    is_wgan = config["gan_type"].startswith("wgan")
    normalize_range = (-1, 1) if is_wgan else (0, 1)

    # 设置数据加载器
    transform = transforms.Compose([
        transforms.ToTensor()
    ])  # ToTensor 已经将数据缩放到 [0, 1]
    dataset = DigitsDataset(
        X,
        y,
        label=config["digit_label"],
        transform=transform,
        normalize_range=normalize_range,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=config.get("num_workers", 0),  # 根据系统调整
    )

    # 显示一些真实样本图像
    real_samples, _ = next(iter(dataloader))
    save_generated_images(
        real_samples,
        os.path.join(log_dir, "real_samples.png"),
        config["image_size"],
        normalize_range,
    )
    img_grid_real = torchvision.utils.make_grid(
        real_samples.view(
            config["batch_size"], 1, config["image_size"], config["image_size"]
        ),
        normalize=True,
        value_range=normalize_range,
    )
    writer.add_image("Real Samples", img_grid_real, 0)

    # --- 初始化模型、量子电路和优化器 ---
    print(f"初始化模型: {config['gan_type']}")
    image_size = config["image_size"]
    n_qubits = config["n_qubits"]
    n_a_qubits = config["n_a_qubits"]
    q_depth = config["q_depth"]
    n_generators = config["n_generators"]
    lrG = config["lrG"]
    lrD = config.get("lrD", config.get("lrC"))  # 兼容 lrD 和 lrC

    # 选择量子电路设置函数和部分测量函数
    if config["gan_type"] == "classic" or config["gan_type"] == "improved":
        partial_measure_func = setup_classic_quantum_circuit(
            n_qubits, n_a_qubits, q_depth
        )
        generator = create_classic_qgan_generator(
            n_qubits,
            n_a_qubits,
            q_depth,
            n_generators,
            device,
            partial_measure_func,
            config.get("q_delta", 1),
        )
        if config["gan_type"] == "classic":
            discriminator = Discriminator(image_size).to(device)
        else:  # improved
            discriminator = ImprovedDiscriminator(
                image_size, config.get("dropout_rate", 0.3)
            ).to(device)
        critic = None  # 经典GAN没有Critic
    elif config["gan_type"] == "wgan_gp":
        partial_measure_func = setup_wgan_quantum_circuit(n_qubits, n_a_qubits, q_depth)
        generator = create_wgan_qgan_generator(
            n_qubits,
            n_a_qubits,
            q_depth,
            n_generators,
            device,
            partial_measure_func,
            config.get("q_delta", 0.1),
            image_size * image_size,
        )
        critic = Critic(image_size, config.get("dropout_rate", 0.3)).to(device)
        discriminator = None  # WGAN有Critic
    elif config["gan_type"] == "wgan_gp_mbd":
        partial_measure_func = setup_wgan_quantum_circuit_mbd(
            n_qubits, n_a_qubits, q_depth
        )
        generator = create_wgan_mbd_qgan_generator(
            n_qubits,
            n_a_qubits,
            q_depth,
            n_generators,
            device,
            partial_measure_func,
            config.get("q_delta", 0.1),
            image_size * image_size,
        )
        critic = CriticWithMinibatchDiscrimination(
            image_size,
            config.get("dropout_rate", 0.3),
            config.get("mb_in_features", 32),
            config.get("mb_out_features", 5),
            config.get("mb_intermediate_features", 16),
        ).to(device)
        discriminator = None
    else:
        raise ValueError(f"未知的 GAN 类型: {config['gan_type']}")

    # 设置优化器
    optimizer_type = config.get("optimizer", "Adam")  # 默认为Adam
    betas = config.get("betas", (0.5, 0.9))

    if optimizer_type.lower() == "adam":
        optG = optim.Adam(generator.parameters(), lr=lrG, betas=betas)
        if discriminator:
            optD = optim.Adam(discriminator.parameters(), lr=lrD, betas=betas)
        if critic:
            optC = optim.Adam(critic.parameters(), lr=lrD, betas=betas)
    elif optimizer_type.lower() == "sgd":
        optG = optim.SGD(generator.parameters(), lr=lrG)
        if discriminator:
            optD = optim.SGD(discriminator.parameters(), lr=lrD)
        if critic:
            # SGD 通常不用于 WGAN-GP 的 Critic
            print("警告：对WGAN-GP Critic使用SGD可能不是最佳选择，考虑Adam")
            optC = optim.SGD(critic.parameters(), lr=lrD)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    # 学习率调度器 (可选)
    schedulerG = None
    schedulerD = None
    schedulerC = None
    if config.get("use_scheduler", False):
        scheduler_type = config.get("scheduler_type", "CosineAnnealingLR")
        num_iter = config["num_iter"]
        if scheduler_type == "CosineAnnealingLR":
            schedulerG = optim.lr_scheduler.CosineAnnealingLR(optG, num_iter)
            if discriminator:
                schedulerD = optim.lr_scheduler.CosineAnnealingLR(optD, num_iter)
            if critic:
                schedulerC = optim.lr_scheduler.CosineAnnealingLR(optC, num_iter)
        else:
            print(
                f"警告：不支持的学习率调度器类型 '{scheduler_type}'，将不使用调度器。"
            )

    # 损失函数 (经典GAN)
    criterion = nn.BCELoss() if not is_wgan else None

    # 固定噪声，用于可视化
    fixed_noise = torch.rand(8, n_qubits, device=device) * math.pi / 2

    # --- 训练循环 ---
    print("开始训练...")
    global_step = 0
    generator_iters = 0

    # 经典GAN标签
    real_labels = (
        torch.full((config["batch_size"],), 1.0, dtype=torch.float, device=device)
        if not is_wgan
        else None
    )
    fake_labels = (
        torch.full((config["batch_size"],), 0.0, dtype=torch.float, device=device)
        if not is_wgan
        else None
    )

    while global_step < config["num_iter"]:
        for i, (data, _) in enumerate(dataloader):
            # 准备真实数据
            real_data = data.view(config["batch_size"], -1).to(device)
            current_batch_size = real_data.size(0)
            if current_batch_size != config["batch_size"]:
                # 如果最后一个批次大小不同，跳过或调整标签大小
                # print(f"跳过大小为 {current_batch_size} 的批次")
                continue  # 简单跳过

            # 生成随机噪声
            noise = (
                torch.rand(current_batch_size, n_qubits, device=device) * math.pi / 2
            )

            # --- 训练 Discriminator 或 Critic ---
            if is_wgan:
                # --- WGAN-GP Critic 训练 ---
                critic.train()
                generator.eval()  # 训练Critic时冻结生成器
                for _ in range(
                    config["n_critic"]
                ):  # 每个生成器步骤训练 n_critic 次 Critic
                    optC.zero_grad()

                    # 生成假数据
                    with torch.no_grad():  # Critic训练时不计算生成器的梯度
                        fake_data = generator(noise)

                    # 计算Critic评分
                    critic_real = critic(real_data).mean()
                    critic_fake = critic(fake_data).mean()

                    # 计算梯度惩罚
                    gradient_penalty = compute_gradient_penalty(
                        critic, real_data.data, fake_data.data, device
                    )

                    # Critic损失
                    critic_loss = (
                        -critic_real
                        + critic_fake
                        + config["lambda_gp"] * gradient_penalty
                    )

                    # 反向传播和优化
                    critic_loss.backward()
                    optC.step()

                # 记录 WGAN Critic 指标 (记录最后一次迭代的)
                writer.add_scalar("Loss/Critic", critic_loss.item(), global_step)
                writer.add_scalar(
                    "Metrics/Critic_real", critic_real.item(), global_step
                )
                writer.add_scalar(
                    "Metrics/Critic_fake", critic_fake.item(), global_step
                )
                writer.add_scalar(
                    "Metrics/Gradient_penalty", gradient_penalty.item(), global_step
                )
                if schedulerC:
                    writer.add_scalar(
                        "LearningRate/Critic", schedulerC.get_last_lr()[0], global_step
                    )

            else:
                # --- Classic GAN Discriminator 训练 ---
                discriminator.train()
                generator.eval()
                optD.zero_grad()

                # 真实数据损失
                outD_real = discriminator(real_data).view(-1)
                errD_real = criterion(outD_real, real_labels[:current_batch_size])
                errD_real.backward()

                # 假数据损失
                with torch.no_grad():
                    fake_data = generator(noise)
                outD_fake = discriminator(fake_data.detach()).view(-1)
                errD_fake = criterion(outD_fake, fake_labels[:current_batch_size])
                errD_fake.backward()

                # 总损失和优化
                errD = errD_real + errD_fake
                optD.step()

                # 记录 Classic Discriminator 指标
                writer.add_scalar("Loss/Discriminator", errD.item(), global_step)
                writer.add_scalar(
                    "Performance/D(x)", outD_real.mean().item(), global_step
                )
                writer.add_scalar(
                    "Performance/D(G(z))_D_step", outD_fake.mean().item(), global_step
                )  # 判别器步骤中的D(G(z))
                if schedulerD:
                    writer.add_scalar(
                        "LearningRate/Discriminator",
                        schedulerD.get_last_lr()[0],
                        global_step,
                    )

            # --- 训练 Generator ---
            # 每 n_critic 次迭代训练一次生成器 (WGAN)，或每次都训练 (Classic GAN)
            train_gen = (
                (global_step % config.get("n_critic", 1) == 0) if is_wgan else True
            )

            if train_gen:
                generator.train()
                if discriminator:
                    discriminator.eval()  # 训练生成器时冻结判别器
                if critic:
                    critic.eval()
                optG.zero_grad()

                # 生成新的假数据
                fake_data = generator(noise)

                if is_wgan:
                    # WGAN 生成器损失
                    gen_score = critic(fake_data).mean()
                    generator_loss = -gen_score  # 生成器目标是最大化Critic评分
                else:
                    # Classic GAN 生成器损失
                    outG_fake = discriminator(fake_data).view(-1)
                    generator_loss = criterion(
                        outG_fake, real_labels[:current_batch_size]
                    )  # 生成器目标是让判别器认为是真的

                # 反向传播和优化
                generator_loss.backward()
                optG.step()
                generator_iters += 1

                # 记录 Generator 指标
                writer.add_scalar("Loss/Generator", generator_loss.item(), global_step)
                if is_wgan:
                    writer.add_scalar(
                        "Metrics/Critic_fake_G_step", gen_score.item(), global_step
                    )  # 生成器步骤中的Critic(G(z))
                else:
                    writer.add_scalar(
                        "Performance/D(G(z))_G_step",
                        outG_fake.mean().item(),
                        global_step,
                    )  # 生成器步骤中的D(G(z))
                if schedulerG:
                    writer.add_scalar(
                        "LearningRate/Generator",
                        schedulerG.get_last_lr()[0],
                        global_step,
                    )

                # 更新学习率调度器 (如果使用)
                if schedulerG:
                    schedulerG.step()
                if schedulerD:
                    schedulerD.step()
                if schedulerC:
                    schedulerC.step()

                # --- 日志记录和可视化 ---
                if generator_iters % config.get("log_interval", 10) == 0:
                    if is_wgan:
                        print(
                            f"[{global_step}/{config['num_iter']}] "
                            f"Loss_C: {critic_loss.item():.4f} Loss_G: {generator_loss.item():.4f} "
                            f"C(x): {critic_real.item():.4f} C(G(z)): {critic_fake.item():.4f}/{gen_score.item():.4f} "
                            f"GP: {gradient_penalty.item():.4f}"
                        )
                    else:
                        print(
                            f"[{global_step}/{config['num_iter']}] "
                            f"Loss_D: {errD.item():.4f} Loss_G: {generator_loss.item():.4f} "
                            f"D(x): {outD_real.mean().item():.4f} D(G(z)): {outD_fake.mean().item():.4f}/{outG_fake.mean().item():.4f}"
                        )

                # 每 N 次迭代生成并保存图像
                if generator_iters % config.get("image_interval", 50) == 0:
                    generator.eval()  # 设置为评估模式
                    with torch.no_grad():
                        test_images = generator(fixed_noise)
                    img_path = os.path.join(
                        log_dir, f"generated_images_step_{global_step}.png"
                    )
                    save_generated_images(
                        test_images, img_path, image_size, normalize_range
                    )
                    # 添加到TensorBoard
                    img_grid_gen = torchvision.utils.make_grid(
                        test_images.view(8, 1, image_size, image_size),
                        normalize=True,
                        value_range=normalize_range,
                    )
                    writer.add_image("Generated Images", img_grid_gen, global_step)
                    generator.train()  # 恢复训练模式

                # 每 N 次迭代保存模型检查点
                if generator_iters % config.get("checkpoint_interval", 100) == 0:
                    checkpoint_path = os.path.join(
                        log_dir, f"checkpoint_step_{global_step}.pt"
                    )
                    save_payload = {
                        "global_step": global_step,
                        "generator_state_dict": generator.state_dict(),
                        "generator_optimizer": optG.state_dict(),
                        "config": config,
                    }
                    if discriminator:
                        save_payload["discriminator_state_dict"] = (
                            discriminator.state_dict()
                        )
                        save_payload["discriminator_optimizer"] = optD.state_dict()
                    if critic:
                        save_payload["critic_state_dict"] = critic.state_dict()
                        save_payload["critic_optimizer"] = optC.state_dict()
                    torch.save(save_payload, checkpoint_path)
                    print(f"检查点已保存至: {checkpoint_path}")

            global_step += 1
            if global_step >= config["num_iter"]:
                break

    # --- 训练完成 ---
    print("训练完成!")

    # 保存最终模型
    final_model_path = os.path.join(log_dir, "final_model.pt")
    final_payload = {"generator_state_dict": generator.state_dict(), "config": config}
    if discriminator:
        final_payload["discriminator_state_dict"] = discriminator.state_dict()
    if critic:
        final_payload["critic_state_dict"] = critic.state_dict()
    torch.save(final_payload, final_model_path)
    print(f"最终模型已保存至: {final_model_path}")

    writer.close()
    print(f"TensorBoard 日志已保存至: {log_dir}")


# --- 配置字典 ---

config_classic = {
    "gan_type": "classic",
    "run_name": "qgan_classic",
    "image_size": 8,
    "batch_size": 8,
    "digit_label": 3,
    "n_qubits": 5,
    "n_a_qubits": 1,
    "q_depth": 4,
    "n_generators": 4,
    "lrG": 0.3,
    "lrD": 0.01,
    "num_iter": 3000,
    "optimizer": "SGD",
    "seed": 42,
    "log_interval": 10,
    "image_interval": 50,
    "checkpoint_interval": 500,
}

config_improved = {
    "gan_type": "improved",
    "run_name": "qgan_improved",
    "image_size": 8,
    "batch_size": 8,
    "digit_label": 3,
    "n_qubits": 5,
    "n_a_qubits": 1,
    "q_depth": 4,
    "n_generators": 4,
    "lrG": 0.1,
    "lrD": 0.01,
    "num_iter": 5000,
    "optimizer": "Adam",  # 尝试Adam
    "betas": (0.5, 0.999),
    "dropout_rate": 0.3,
    "seed": 42,
    "log_interval": 10,
    "image_interval": 50,
    "checkpoint_interval": 500,
}

config_wgan_gp = {
    "gan_type": "wgan_gp",
    "run_name": "qwgan_gp",
    "image_size": 8,
    "batch_size": 8,
    "digit_label": 3,  # 之前训练的是4，这里改为3
    "n_qubits": 4,
    "n_a_qubits": 1,
    "q_depth": 4,
    "n_generators": 2,
    "lrG": 1e-4,
    "lrC": 1e-4,
    "num_iter": 5000,  # 增加迭代次数
    "optimizer": "Adam",
    "betas": (0.5, 0.9),
    "n_critic": 5,
    "lambda_gp": 10,
    "q_delta": 0.1,
    "dropout_rate": 0.3,
    "seed": 42,
    "use_scheduler": True,
    "scheduler_type": "CosineAnnealingLR",
    "log_interval": 10,
    "image_interval": 50,
    "checkpoint_interval": 500,
}

config_wgan_gp_mbd = {
    "gan_type": "wgan_gp_mbd",
    "run_name": "qwgan_gp_mbd",
    "image_size": 8,
    "batch_size": 8,
    "digit_label": 4,
    "n_qubits": 5,
    "n_a_qubits": 1,
    "q_depth": 6,
    "n_generators": 4,
    "lrG": 1e-4,
    "lrC": 1e-4,
    "num_iter": 10000,
    "optimizer": "Adam",
    "betas": (0.5, 0.9),
    "n_critic": 5,
    "lambda_gp": 50,  # 调整梯度惩罚系数
    "q_delta": 0.1,
    "dropout_rate": 0.3,
    "mb_in_features": 32,
    "mb_out_features": 10,
    "mb_intermediate_features": 32,
    "seed": 42,
    "use_scheduler": True,
    "scheduler_type": "CosineAnnealingLR",
    "log_interval": 10,
    "image_interval": 50,
    "checkpoint_interval": 500,
}

# --- 主执行块 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练量子生成对抗网络")
    parser.add_argument(
        "--config",
        type=str,
        default="classic",
        choices=["classic", "improved", "wgan_gp", "wgan_gp_mbd"],
        help="选择要使用的配置 (classic, improved, wgan_gp, wgan_gp_mbd)",
    )
    args = parser.parse_args()

    if args.config == "classic":
        config_to_run = config_classic
    elif args.config == "improved":
        config_to_run = config_improved
    elif args.config == "wgan_gp":
        config_to_run = config_wgan_gp
    elif args.config == "wgan_gp_mbd":
        config_to_run = config_wgan_gp_mbd
    else:
        # 这个分支理论上不会执行，因为choices限制了选项
        print(f"错误：无效的配置名称 '{args.config}'")
        exit()

    print(f"--- 使用配置: {args.config} ---")

    train(config_to_run)
