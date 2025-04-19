import torch
import pennylane as qml

# --- Quantum Circuit and Measurement for Classic/Improved QGAN ---


def setup_classic_quantum_circuit(n_qubits=5, n_a_qubits=1, q_depth=4):
    """设置经典QGAN的量子电路和部分测量函数"""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def quantum_circuit(noise, weights):
        weights = weights.reshape(q_depth, n_qubits)

        # 初始化latent向量
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)

        # 重复层
        for i in range(q_depth):
            # 参数化层
            for y in range(n_qubits):
                qml.RY(weights[i][y], wires=y)

            # 控制Z门
            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])

        return qml.probs(wires=list(range(n_qubits)))

    # 用于非线性变换和后处理 (Classic QGAN version)
    def partial_measure_classic(noise, weights):
        probs = quantum_circuit(noise, weights)
        # 截取数据量子比特的概率
        probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
        # 归一化条件概率 (分母是所有概率之和，确保总和为1)
        probsgiven0 /= torch.sum(probs)  # 注意：这里原文分母是probs，不是probsgiven0

        # 后处理：除以最大值
        probsgiven = probsgiven0 / torch.max(probsgiven0)
        return probsgiven

    return partial_measure_classic


# --- Quantum Circuit and Measurement for WGAN-GP QGAN ---


def setup_wgan_quantum_circuit(n_qubits=4, n_a_qubits=1, q_depth=4):
    """设置WGAN-GP QGAN的量子电路和部分测量函数"""
    dev = qml.device("default.qubit", wires=n_qubits)  # 或 lightning.qubit

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def quantum_circuit_wgan(noise, weights):
        weights = weights.reshape(q_depth, n_qubits)  # 确保形状正确

        # 1. 更丰富的初始状态嵌入
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(noise[i], wires=i)
            if i % 2 == 0:
                # 使用 % n_qubits 确保索引有效
                qml.RX(noise[(i + 1) % n_qubits], wires=i)

        # 重复参数化层
        for i in range(q_depth):
            # 2. 增加参数化门 (RY 和 RZ)
            for y in range(n_qubits):
                qml.RY(weights[i][y], wires=y)
            # 添加 RZ 门，注意参数索引
            for y in range(n_qubits):
                # 使用不同的参数索引避免直接重用RY的参数
                qml.RZ(
                    weights[i][(y + n_qubits // 2) % n_qubits], wires=y
                )  # 示例索引，需确认参数形状

            # 3. 增强纠缠结构 (CZ 和 CNOT)
            # 基础纠缠
            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])
            # 附加纠缠
            if i % 2 == 0:
                for y in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[y, (y + 1) % n_qubits])
            else:
                for y in range(1, n_qubits - 1, 2):
                    # 确保第二个线索引有效
                    qml.CNOT(wires=[y, (y + 1) % n_qubits])

        return qml.probs(wires=list(range(n_qubits)))

    # 用于非线性变换和后处理 (WGAN-GP version)
    def partial_measure_wgan(noise, weights):
        probs = quantum_circuit_wgan(noise, weights)
        probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
        # 确保条件概率和为1
        sum_probs_given0 = torch.sum(probsgiven0)
        # 避免除以零
        if sum_probs_given0 > 1e-8:
            probsgiven0 /= sum_probs_given0
        else:
            # 如果概率和接近零，可能需要特殊处理，例如返回均匀分布或零向量
            probsgiven0 = torch.ones_like(probsgiven0) / probsgiven0.numel()

        # 后处理 - 映射到[-1,1]范围
        probsgiven = 2 * probsgiven0 - 1
        return probsgiven

    return partial_measure_wgan


# --- Quantum Circuit for WGAN-GP QGAN with Minibatch Discrimination ---
# 通常 Minibatch Discrimination 版本使用与 WGAN-GP 相同的量子电路结构
def setup_wgan_mbd_quantum_circuit(n_qubits=5, n_a_qubits=1, q_depth=6):
    """设置带有MBD的WGAN-GP QGAN的量子电路和部分测量函数"""
    # 复用 WGAN 电路设置，但使用不同的参数
    return setup_wgan_quantum_circuit(n_qubits, n_a_qubits, q_depth)
