import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    定义 Q 网络，输入状态特征，输出每个动作的 Q 值
    Parameters:
        num_features (int): 状态特征维度  (B, D)
        num_actions (int): 动作空间大小   (B, A)
    """
    def __init__(self, num_features: int, num_actions: int):
        super().__init__()
        # network architecture
        # 输入层 -> 隐藏层1 (ReLU)
        # 隐藏层1 -> 隐藏层2 (ReLU)
        # 隐藏层2 -> 输出层 (A个动作的Q值)
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # (B, A)
        return x