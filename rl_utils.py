from typing import List, Dict, Tuple
import numpy as np
import re

# 修改rl_utils.py中的StateEncoder

# 修改rl_utils.py中的StateEncoder

class StateEncoder:
    """修改状态编码器以支持心理特征"""
    
    def encode_play_state(self, round_base_info, round_action_info, play_decision_info,
                         hand, target_card, current_bullet, psych_features=None):
        """编码出牌阶段状态"""
        base_state = self._encode_base_state(round_base_info, hand, target_card, current_bullet)
        
        # 添加心理特征
        if psych_features:
            base_state = list(base_state)
            base_state.extend(psych_features)
        
        return tuple(base_state)
    
    def encode_challenge_state(self, round_base_info, round_action_info, challenge_decision_info,
                              challenging_player_performance, hand, target_card, current_bullet,
                              psych_features=None):
        """编码质疑阶段状态"""
        base_state = list(self._encode_base_state(round_base_info, hand, target_card, current_bullet))
        
        # 提取声称牌数
        claimed_cards = 0
        match = re.search(r'宣称打出(\d+)张', challenging_player_performance)
        if match:
            claimed_cards = int(match.group(1))
        base_state.append(claimed_cards)
        
        # 添加心理特征
        if psych_features:
            base_state.extend(psych_features)
        
        return tuple(base_state)
    
    def _encode_base_state(
        self,
        round_base_info: str,
        hand: List[str],
        target_card: str,
        current_bullet: int
    ) -> Tuple:
        """
        编码两个阶段共有的基础状态（提取公共逻辑）
        """
        # 提取轮次信息
        round_id_match = re.search(r'第(\d+)轮', round_base_info)
        if round_id_match:
            round_id = int(round_id_match.group(1))
        else:
            round_id = 1
        
        # 手牌编码
        hand_counts = self._encode_hand(hand, target_card)
        
        # 基础状态特征
        base_state = (
            round_id,
            target_card,
            current_bullet,
            hand_counts["q_count"],
            hand_counts["k_count"],
            hand_counts["a_count"],
            hand_counts["joker_count"],
            hand_counts["target_count"],
            len(hand)
        )
        
        return base_state
    
    def _encode_hand(self, hand: List[str], target_card: str) -> Dict:
        """
        编码手牌信息
        """
        counts = {
            "q_count": hand.count("Q"),
            "k_count": hand.count("K"),
            "a_count": hand.count("A"),
            "joker_count": hand.count("Joker"),
            "target_count": hand.count(target_card) + hand.count("Joker")  # Joker可以当作目标牌
        }
        return counts

CARD_TYPES = ("Q", "K", "A", "Joker")

class PlayActionTemplateSpace:
    """
    固定的出牌动作模板空间：
    模板 = (q,k,a,j) 表示分别打出多少张 Q/K/A/Joker
    """
    def __init__(self, max_hand_size: int = 5):
        self.max_hand_size = max_hand_size
        self.templates: List[Tuple[int,int,int,int]] = self._build_templates()

    def _build_templates(self) -> List[Tuple[int,int,int,int]]:
        temps = []
        H = self.max_hand_size
        for total in range(0, H + 1):
            for q in range(total + 1):
                for k in range(total - q + 1):
                    for a in range(total - q - k + 1):
                        j = total - q - k - a
                        temps.append((q, k, a, j))
        return temps

    @property
    def n(self) -> int:
        return len(self.templates)

    def to_cards(self, tpl: Tuple[int,int,int,int]) -> List[str]:
        q, k, a, j = tpl
        return (["Q"] * q) + (["K"] * k) + (["A"] * a) + (["Joker"] * j)

    def is_legal(self, tpl: Tuple[int,int,int,int], hand: List[str]) -> bool:
        q, k, a, j = tpl
        return (q <= hand.count("Q")
                and k <= hand.count("K")
                and a <= hand.count("A")
                and j <= hand.count("Joker"))


class ActionDecoder:
    """
    将强化学习代理的动作转换为游戏可以理解的形式
    """
    def __init__(self, max_hand_size: int = 5):
        # max_hand_size 设成你游戏里"可能出现的最大手牌数"
        self.play_space = PlayActionTemplateSpace(max_hand_size=max_hand_size)

    def num_play_actions(self) -> int:
        return self.play_space.n

    def get_legal_play_actions(self, hand: List[str]) -> List[int]:
        """
        返回合法的模板动作 id（固定在 [0, N-1]）
        """
        legal = []
        for i, tpl in enumerate(self.play_space.templates):
            if self.play_space.is_legal(tpl, hand):
                legal.append(i)
        return legal

    def get_play_action_mask(self, hand: List[str]) -> np.ndarray:
        """
        DQN 用：返回 shape=(N,) 的 0/1 mask
        """
        mask = np.zeros(self.play_space.n, dtype=np.float32)
        for i in self.get_legal_play_actions(hand):
            mask[i] = 1.0
        return mask

    def decode_play_action(self, action_idx: int, hand: List[str]) -> Dict:
        """
        action_idx -> 实际要打出的牌列表
        """
        tpl = self.play_space.templates[action_idx]
        played_cards = self.play_space.to_cards(tpl)
        return {"played_cards": played_cards}
    
    def num_total_actions(self) -> int:
        return self.play_space.n + 2

    def get_total_action_mask(self, phase: str, hand: List[str]) -> np.ndarray:
        """
        phase: "play" 或 "challenge"
        """
        N = self.num_play_actions()
        mask = np.zeros(N + 2, dtype=np.float32)
        if phase == "play":
            mask[:N] = self.get_play_action_mask(hand)
        else:
            mask[N:] = 1.0  # 两个质疑动作都合法
        return mask
    
# 修改rl_utils.py中的FeatureExtractor

class FeatureExtractor:
    """修改特征提取器以处理心理特征"""
    
    def get_features(self, state: Tuple) -> np.ndarray:
        # 基础特征提取
        round_id = state[0]
        target_card = state[1]
        current_bullet = state[2]
        q_count = state[3]
        k_count = state[4]
        a_count = state[5]
        joker_count = state[6]
        target_count = state[7]
        hand_size = state[8]
        
        # 判断状态类型
        is_challenge_state = len(state) > 9
        claimed_cards = state[9] if is_challenge_state else 0
        
        features = []
        
        # 1. 游戏轮次特征
        features.append(round_id)
        
        # 2. 目标牌特征
        features.extend(self._encode_target_card(target_card))
        
        # 3. 子弹位置特征
        features.append(current_bullet)
        features.append(6 - current_bullet)
        features.append(1 if current_bullet == 0 else 0)
        
        # 4. 手牌特征
        features.append(target_count)
        features.append(target_count / hand_size if hand_size > 0 else 0)
        features.append(1 if target_count > 0 else 0)
        
        # 5. 手牌大小
        features.append(hand_size)
        features.append(1 if hand_size == 0 else 0)
        
        # 6. 特殊牌
        features.append(joker_count)
        features.append(1 if joker_count > 0 else 0)
        
        # 7. 质疑相关特征
        if is_challenge_state:
            features.append(claimed_cards)
        else:
            features.append(0)
        
        # 8. 心理特征（最后4个位置）
        psych_start_idx = 10 if is_challenge_state else 9
        if len(state) >= psych_start_idx + 4:
            psych_features = list(state[psych_start_idx:psych_start_idx+4])
            features.extend(psych_features)
        else:
            # 没有心理特征时使用默认值
            features.extend([0.5, 0.5, 0.0, 0.5])  # 信任度、攻击性、熟悉度、趋势
        
        return np.array(features, dtype=np.float32)
    
    def _encode_target_card(self, target_card: str) -> List[int]:
        """
        对目标牌进行one-hot编码
        """
        if target_card == 'Q':
            return [1, 0, 0]
        elif target_card == 'K':
            return [0, 1, 0]
        elif target_card == 'A':
            return [0, 0, 1]
        else:
            return [0, 0, 0]  # 未知目标牌