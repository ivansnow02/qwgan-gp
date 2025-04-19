

# 量子生成对抗网络 (QGAN) 实现

---
[English](README.md)

## 中文
本仓库提供了多种量子生成对抗网络 (QGAN) 模型的 PyTorch 实现。项目专注于基于 UCI 手写数字光学识别数据集生成手写数字图像。

### 功能

* **经典 QGAN:** 基于原始 QGAN 概念的实现。
* **改进 QGAN:** 在经典 QGAN 基础上进行改进，例如在判别器中加入 Dropout。
* **量子 WGAN-GP:** 使用 Wasserstein GAN 梯度惩罚 (WGAN-GP) 框架并适配量子生成器的实现。
* **带 Minibatch Discrimination 的量子 WGAN-GP:** 在 WGAN-GP QGAN 的 Critic 中加入 Minibatch Discrimination，旨在提高生成样本的多样性。

### 环境要求

* Python 3.x
* PyTorch (`torch`, `torchvision`)
* NumPy
* Matplotlib
* TensorBoard (`tensorboard`)
* UCIMLRepo (`ucimlrepo`)
* PennyLane

你可以使用 pip 安装所需的 Python 包：

```bash
pip install -r requirements.txt
```

### 安装

1. 克隆仓库：

   ```bash
   git clone https://github.com/ivansnow02/qwgan-gp.git
   cd qwgan-gp
   ```

2. 根据“环境要求”部分安装依赖项。

### 使用方法

通过命令行运行主训练脚本 `train.py`，并指定所需的配置：

```bash
python train.py --config <config_name>
```

可用的配置 (`<config_name>`):

* `classic`
* `improved`
* `wgan_gp`
* `wgan_gp_mbd`

示例:

```bash
python train.py --config wgan_gp_mbd
```

### 配置

每种模型类型的详细训练参数（例如，学习率、批次大小、迭代次数、量子线路参数、图像大小等）都在 `train.py` 脚本末尾的配置字典中定义（例如 `config_classic`, `config_wgan_gp`）。您可以修改这些字典来尝试不同的设置。

### 结果与监控

* **TensorBoard:** 训练进度、损失值、性能指标（如 D(x)、D(G(z))、Critic 分数）以及生成的图像样本会使用 TensorBoard 记录。在项目根目录启动 TensorBoard：

  ```bash
  tensorboard --logdir runs
  ```

  然后在您的网络浏览器中访问 `http://localhost:6006`（或 TensorBoard 指定的端口）。

* **保存的文件:** 训练过程中生成的图像和模型检查点会定期保存在 `runs/<run_name>/` 目录下，其中 `<run_name>` 对应所运行的配置（例如 `runs/qwgan_gp/`）。

