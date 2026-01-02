import numpy as np
import random
import joblib
from typing import List, Tuple
from sklearn.linear_model import SGDRegressor
from rl_utils import FeatureExtractor, StateEncoder, ActionDecoder

class LinearQAgent:
    """
    强化学习代理，使用线性Q-learning算法
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.feature_extractor = FeatureExtractor()
        self.state_encoder = StateEncoder()
        self.action_decoder = ActionDecoder()
        
        # # 初始化Q表（状态空间较大时可考虑使用函数近似）
        # self.q_table = {}

        # Approximate Q-learing using a linear model
        # 函数近似器：为每个动作创建一个线性模型（One-Vs-All 方法）
        self.action_models = {}  # 键：动作，值：SGDRegressor

    def _initialize_models(self, state: Tuple, legal_actions: List[int]) -> None:
        """
        初始化每个动作的线性模型
        """
        # 提取状态特征以确定特征维度
        state_features = self.feature_extractor.get_features(state)
        
        for action in legal_actions:
            if action not in self.action_models:
                self.action_models[action] = SGDRegressor(
                    learning_rate='constant',
                    eta0=self.lr,
                    max_iter=1000,
                    tol=1e-3
                )
                # 初始化模型（SGDRegressor需要至少一次fit调用）
                self.action_models[action].fit([state_features], [0.0])


    def choose_action(self, state: Tuple, mask: np.ndarray, is_training: bool = True) -> int:
        """
        根据当前状态选择动作
        """
        
        # 过滤出合法动作
        legal_actions = [i for i, m in enumerate(mask) if m > 0.5]

        # 探索：随机选择动作
        if is_training and random.uniform(0, 1) < self.epsilon:
            # print("choose random action")
            return random.choice(legal_actions)
        
        # 利用：选择Q值最高的动作
        q_values = self._get_q_values(state, legal_actions)
        # print(f"choose action by argmax: {legal_actions[np.argmax(q_values)]}")
        return legal_actions[np.argmax(q_values)]
    
    # def update(self, state: Tuple, action: int, reward: float, next_state: Tuple, is_training: bool = True) -> None:
    #     """
    #     更新Q表
    #     """
    #     # 非训练模式下，不更新Q表
    #     if not is_training:
    #         return
        
    #     state_key = self._get_state_key(state)
    #     next_state_key = self._get_state_key(next_state) if next_state else None
        
    #     # 获取当前Q值
    #     current_q = self.q_table.get(state_key, {}).get(action, 0.0)
        
    #     if next_state_key:
    #         # 获取下一个状态的所有Q值
    #         next_q_dict = self.q_table.get(next_state_key, {})
    #         max_next_q = max(next_q_dict.values()) if next_q_dict else 0.0
    #         # Q-learning更新规则
    #         new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
    #     else:
    #         # 终端状态
    #         new_q = current_q + self.lr * (reward - current_q)
        
    #     # 更新Q表
    #     if state_key not in self.q_table:
    #         self.q_table[state_key] = {}
    #     self.q_table[state_key][action] = new_q
    #     # print(f"Q-table updated: {state_key} -> {action} -> {new_q:.4f}")
        
    #     # 衰减探索率
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    
    # def _get_state_key(self, state: Tuple) -> Tuple: # 以下划线_开头，说明是私有方法，不应该被外部调用
    #     """
    #     将状态转换为可哈希的键，方便高效查找
    #     """
    #     return tuple(state)

    def update(self, state, action, reward, next_state, mask, done, is_training=True):
        """更新函数近似器的参数（Q值）"""
        if not is_training:
            return

        if isinstance(mask, np.ndarray):
            next_legal_actions = [i for i, m in enumerate(mask) if m > 0.5]

        # 确保当前 action 对应的模型存在
        self._initialize_models(state, [action])

        # 1) 计算 TD target
        if done or next_state is None or not next_legal_actions:
            target_q = reward
        else:
            # _get_q_values 内部会初始化 next_legal_actions 的模型
            next_q_values = self._get_q_values(next_state, next_legal_actions)
            target_q = reward + self.gamma * float(max(next_q_values))

        # 2) 用 supervised 方式把 Q(s,a) 拟合到 target_q
        state_features = self.feature_extractor.get_features(state)
        self.action_models[action].partial_fit([state_features], np.array([target_q]))

        # 3) epsilon 衰减
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _get_q_values(self, state: Tuple, legal_actions: List[int]) -> List[float]:
        """
        获取合法动作的Q值
        """
        # 初始化模型（如果未初始化）
        self._initialize_models(state, legal_actions)
        state_features = self.feature_extractor.get_features(state)
        q_values = []
        for action in legal_actions:
            q_value = self.action_models[action].predict([state_features])[0]
            q_values.append(q_value)
        return q_values
    
    def save_model(self, file_path):
        """保存模型参数"""
        model_data = {
            'epsilon': self.epsilon,
            'action_models': self.action_models,
            'learning_rate': self.lr,
            'discount_factor': self.gamma,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }
        with open(file_path, 'wb') as f:
            joblib.dump(model_data, f)  # 使用joblib保存sklearn模型

    def load_model(self, file_path):
        """加载模型参数"""
        with open(file_path, 'rb') as f:
            model_data = joblib.load(f)
        
        self.epsilon = model_data['epsilon']
        self.action_models = model_data['action_models']
        self.lr = model_data['learning_rate']   
        self.gamma = model_data['discount_factor']
        self.epsilon_decay = model_data['epsilon_decay']
        self.epsilon_min = model_data['epsilon_min']
    
    def print_q_table_summary(self, max_entries: int = 10, output_file: str = None) -> None:
        """
        打印Q表摘要（前max_entries个条目）
        
        参数:
            max_entries: 要显示的最大条目数
            output_file: 可选的输出文件路径，如果提供则将输出写入文件，否则打印到控制台
        """
        # 准备输出内容
        output_lines = []
        output_lines.append(f"Size of Q-table: {len(self.q_table)} entries")
        output_lines.append(f"Current epsilon: {self.epsilon}")
        
        if self.q_table:
            output_lines.append(f"First {max_entries} Q-value entries:")
            for i, (state, actions) in enumerate(self.q_table.items()):
                if i >= max_entries:
                    break
                output_lines.append(f"  State {state}: {actions}")
        else:
            output_lines.append("Q-table is empty")
        
        # 输出内容
        output_text = "\n".join(output_lines)
        
        if output_file:
            # 写入文件
            with open(output_file, 'a') as f:  # 使用'a'模式追加写入
                f.write(output_text)
                f.write("\n" + "="*50 + "\n")  # 添加分隔线
            print(f"Q-table summary saved to file: {output_file}")
        else:
            # 打印到控制台
            print(output_text)
