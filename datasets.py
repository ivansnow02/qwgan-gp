import torch
from torch.utils.data import Dataset
import numpy as np


class DigitsDataset(Dataset):
    """PyTorch数据集，用于手写数字识别数据集"""

    def __init__(self, X, y, label=0, transform=None, normalize_range=(-1, 1)):
        """
        Args:
            X (array-like): 特征数据.
            y (array-like): 目标标签.
            label (int): 要筛选的标签（只保留该数字的图像）.
            transform (callable, optional): 可选的变换应用于样本.
            normalize_range (tuple): 归一化范围，例如 (0, 1) 或 (-1, 1).
        """
        self.transform = transform
        self.normalize_range = normalize_range

        # 如果是pandas DataFrame，转换为numpy数组
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        # 确保y是一维数组
        if len(y.shape) > 1:
            y = y.flatten()

        # 根据标签过滤样本
        mask = y == label

        # 仅沿第一维度（样本）应用掩码
        self.images = X[mask]
        self.labels = np.full(len(self.images), label)  # 所有标签相同(label)

        # 数据归一化
        # 原始数据范围是 0-16
        min_val, max_val = normalize_range
        self.images = ((self.images / 16.0) * (max_val - min_val)) + min_val

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]  # 已归一化
        image = np.array(image, dtype=np.float32).reshape(8, 8)

        if self.transform:
            image = self.transform(image)

        # 返回图像和标签
        return image, int(self.labels[idx])
