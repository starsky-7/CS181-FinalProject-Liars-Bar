import numpy as np
import random
from typing import List, Tuple

class RLAgent:
    """
    强化学习代理，使用Q-learning算法
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
        
        # 初始化Q表（状态空间较大时可考虑使用函数近似）
        self.q_table = {}
        
    def choose_action(self, state: Tuple, legal_actions: List[int], is_training: bool = True) -> int:
        """
        根据当前状态选择动作
        """
        state_key = self._get_state_key(state)
        
        # 探索：随机选择动作
        if is_training and random.uniform(0, 1) < self.epsilon:
            return random.choice(legal_actions)
        
        # 利用：选择Q值最高的动作
        q_values = self._get_q_values(state_key, legal_actions)
        return legal_actions[np.argmax(q_values)]
    
    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple) -> None:
        """
        更新Q表
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state) if next_state else None
        
        # 获取当前Q值
        current_q = self.q_table.get(state_key, {}).get(action, 0.0)
        
        if next_state_key:
            # 获取下一个状态的所有Q值
            next_q_dict = self.q_table.get(next_state_key, {})
            max_next_q = max(next_q_dict.values()) if next_q_dict else 0.0
            # Q-learning更新规则
            new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        else:
            # 终端状态
            new_q = current_q + self.lr * (reward - current_q)
        
        # 更新Q表
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action] = new_q
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _get_state_key(self, state: Tuple) -> Tuple: # 以下划线_开头，说明是私有方法，不应该被外部调用
        """
        将状态转换为可哈希的键，方便高效查找
        """
        return tuple(state)
    
    def _get_q_values(self, state_key: Tuple, legal_actions: List[int]) -> List[float]:
        """
        获取合法动作的Q值
        """
        q_values = []
        for action in legal_actions:
            q_value = self.q_table.get(state_key, {}).get(action, 0.0)
            q_values.append(q_value)
        return q_values
    
    def save_model(self, file_path: str) -> None:
        """
        保存模型
        """
        import pickle
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
    
    def print_q_table_summary(self, max_entries: int = 10) -> None:
        """
        打印Q表摘要（前max_entries个条目）
        """
        print(f"Q表大小: {len(self.q_table)} 条目")
        print(f"当前epsilon: {self.epsilon}")
        if self.q_table:
            print("前几个Q值条目:")
            for i, (state, actions) in enumerate(self.q_table.items()):
                if i >= max_entries:
                    break
                print(f"  状态 {state}: {actions}")
        else:
            print("Q表为空")
    
    def load_model(self, file_path: str) -> None:
        """
        加载模型
        """
        import pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
