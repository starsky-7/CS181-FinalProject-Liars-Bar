import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from rl_utils import FeatureExtractor, ActionDecoder
from network import QNetwork


class DQNAgent:
    """
    强化学习代理，使用深度Q网络（DQN）算法
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        buffer_size: int = 20000,
        batch_size: int = 64,
        warmup_steps: int = 500,
        target_update_every: int = 500,   # 不想要 target 可设成 None，并改 update 里用 q_net
        device: str | None = None,
    ):
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.feature_extractor = FeatureExtractor()
        self.action_decoder = ActionDecoder()
        self.action_dim = self.action_decoder.num_total_actions()  # A = N+2

        self.batch_size = batch_size

        # 如果一开始就训练，数据少且相关性高，训练会很不稳定
        # 所以需要预热步骤，先填充一些数据，攒够一定数量经验后再开始训练
        self.warmup_steps = warmup_steps

        # 每隔多少次梯度更新同步 target 网络，这是 DQN 稳定训练的关键之一
        self.target_update_every = target_update_every

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # lazy init（第一次见到 state 才知道 feature 维度）
        self.num_features = None
        self.q_net = None
        self.target_net = None
        self.optimizer = None
        self.loss_fn = nn.SmoothL1Loss()  # Huber

        # 简单 replay buffer
        self.replay = deque(maxlen=buffer_size)  # queue of (s, a, r, s2, mask2, done)
        self.step = 0
        self.grad_updates = 0
        self.lr = learning_rate

    def _lazy_init(self, state):
        if self.q_net is not None:
            return
        x = np.asarray(self.feature_extractor.get_features(state), dtype=np.float32)
        self.num_features = x.shape[0]

        # 一开始 Q 网络和 target 网络参数相同
        self.q_net = QNetwork(self.num_features, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.num_features, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval() # 目标网络不参与训练

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

    @staticmethod
    def _masked_argmax(q_all: np.ndarray, mask: np.ndarray) -> int:
        """mask=1 保留; mask=0 置为极小"""

        # 若 mask=1：加 0，Q 值保留
        # 若 mask=0：加 -1e9，使该动作 Q 值极小，argmax 永远选不到它
        q_masked = q_all + (mask.astype(np.float32) - 1.0) * 1e9
        return int(np.argmax(q_masked))

    def choose_action(self, state, mask: np.ndarray, is_training: bool = True) -> int:
        self._lazy_init(state)

        # 合法动作列表（用于随机探索）
        legal_actions = [i for i, m in enumerate(mask) if m > 0.5]

        if is_training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        # 状态特征提取 --> 转换为 numpy 数组
        x = np.asarray(self.feature_extractor.get_features(state), dtype=np.float32)
        # 特征转换为张量 --> numpy 数组 --> 张量 --> 添加 batch 维度
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1,D)
        with torch.no_grad():
            q_all = self.q_net(xt).squeeze(0).cpu().numpy()     # (B,A) --> (A,) --> numpy 数组
        return self._masked_argmax(q_all, mask)

    def update(self, state, action, reward, next_state, mask, done, is_training=True):
        if not is_training:
            return

        self._lazy_init(state)
        self.step += 1

        # ---- 1) 处理 next_mask：RLPlayer 传进来的是 mask（np.ndarray），终局可能传 []（list）
        if isinstance(mask, np.ndarray):
            next_mask = mask.astype(np.float32)
        else:
            # 终局或不提供时，next_mask 全 0
            next_mask = np.zeros(self.action_dim, dtype=np.float32)

        # ---- 2) 提取特征:
        # 注意：这里的s、s2并不是state、next_state，
        # 而是已经过feature_extractor提取后的状态特征，因此其维度不是(state_dim,)，而是(feature_dim,)
        s = np.asarray(self.feature_extractor.get_features(state), dtype=np.float32)
        if done or next_state is None:
            s2 = np.zeros_like(s, dtype=np.float32)
        else:
            s2 = np.asarray(self.feature_extractor.get_features(next_state), dtype=np.float32)

        # ---- 3) 存 replay
        self.replay.append((s, int(action), float(reward), s2, next_mask, bool(done)))

        # ---- 4) warmup：先攒一些样本再训练, warmup 阶段不训练，只收集数据，同时慢慢降 ε。
        if len(self.replay) < self.warmup_steps:
            self._decay_epsilon()
            return

        # ---- 5) 采样一个 batch 训练
        batch = random.sample(self.replay, self.batch_size)

        S = torch.from_numpy(np.stack([b[0] for b in batch])).to(self.device)               # (B,D)
        A = torch.from_numpy(np.array([b[1] for b in batch], dtype=np.int64)).to(self.device)  # (B,)
        R = torch.from_numpy(np.array([b[2] for b in batch], dtype=np.float32)).to(self.device) # (B,)
        S2 = torch.from_numpy(np.stack([b[3] for b in batch])).to(self.device)              # (B,D)
        M2 = torch.from_numpy(np.stack([b[4] for b in batch])).to(self.device)              # (B,A)
        Done = torch.from_numpy(np.array([b[5] for b in batch], dtype=np.float32)).to(self.device) # (B,)

        # target: y = r + gamma*(1-done)*max_a'( Q_target(s2,a') on legal )
        with torch.no_grad():
            q2_all = self.target_net(S2)                         # (B,A)
            q2_masked = q2_all + (M2 - 1.0) * 1e9               # 屏蔽非法动作
            max_q2, _ = torch.max(q2_masked, dim=1)             # (B,)
            y = R + self.gamma * (1.0 - Done) * max_q2          # (B,)

        # current Q(s,a)
        q_all = self.q_net(S)                                    # (B,A)
        q_sa = q_all.gather(1, A.view(-1, 1)).squeeze(1)         # (B,)

        loss = self.loss_fn(q_sa, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 在线网络 q_net 每步在变
        # 目标网络 target_net 固定一段时间，提供稳定的 target
        self.grad_updates += 1

        # ---- 6) target 网络同步
        if self.target_update_every is not None and self.grad_updates % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # ---- 7) epsilon 衰减
        self._decay_epsilon()

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path: str):
        if self.q_net is None:
            raise ValueError("q_net not initialized yet.")
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict() if self.target_net is not None else None,
            "epsilon": self.epsilon,
            "num_features": self.num_features,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "lr": self.lr,
        }, path)

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device)

        self.gamma = ckpt.get("gamma", self.gamma)
        self.lr = ckpt.get("lr", self.lr)
        self.epsilon = ckpt.get("epsilon", self.epsilon)

        self.num_features = ckpt["num_features"]
        self.action_dim = ckpt["action_dim"]

        # 重新建网络结构
        self.q_net = QNetwork(self.num_features, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.num_features, self.action_dim).to(self.device)

        # 加载参数
        self.q_net.load_state_dict(ckpt["q_net"])
        if ckpt.get("target_net") is not None:
            self.target_net.load_state_dict(ckpt["target_net"])
        else:
            self.target_net.load_state_dict(ckpt["q_net"])

        self.target_net.eval()

        # 重建优化器（继续训练用）
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

