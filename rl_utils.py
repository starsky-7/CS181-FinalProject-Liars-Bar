from typing import List, Dict, Tuple
import numpy as np
import re

class StateEncoder:
    """
    将游戏状态编码为强化学习代理可以理解的形式
    state: 游戏状态元组，包含以下元素：
                - round_id: 轮次
                - target_card: 目标牌 ('Q', 'K', 或 'A')
                - current_bullet: 当前子弹位置
                - q_count: 手牌中Q的数量
                - k_count: 手牌中K的数量
                - a_count: 手牌中A的数量
                - joker_count: 手牌中Joker的数量
                - target_count: 手牌中可当作目标牌的数量
                - hand_size: 当前手牌总数
                - [claimed_cards]: 质疑阶段特有，上一个玩家声称打出的牌数
                - [opponent_features]: 对手特征 (cheat_rate, challenge_rate, aggression, confidence)
    """
    
    def _extract_opponent_name(self, text: str) -> str:
        """
        从文本信息中提取对手名字
        
        例子：
        - "Opponent1是你的下家，决定是否质疑你的出牌" -> "Opponent1"
        - "你正在判断是否质疑Opponent2的出牌" -> "Opponent2"
        """
        # 尝试不同的模式
        patterns = [
            r'(\w+)是你的下家',  # 出牌阶段
            r'质疑(\w+)的出牌',  # 质疑阶段
            r'(Opponent\d+|RL_Player)',  # 通用玩家名匹配
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def encode_play_state(
        self,
        round_base_info: str,
        round_action_info: str,
        play_decision_info: str,
        hand: List[str],
        target_card: str,
        current_bullet: int,
        stats_manager=None  # 新增参数，暂时不使用
    ) -> Tuple:
        """
        编码出牌阶段的状态
        """
        # 获取基础状态
        base_state = self._encode_base_state(round_base_info, hand, target_card, current_bullet)
        
        # 提取对手特征（如果有stats_manager）
        if stats_manager is not None:
            # 从 play_decision_info 提取下家名字
            opponent_name = self._extract_opponent_name(play_decision_info)
            
            if opponent_name:
                # 获取对手特征
                features = stats_manager.get_features(opponent_name, use_recent=True)
                opponent_features = (
                    features['cheat_rate'],
                    features['challenge_rate'],
                    features['aggression'],
                    features['confidence']
                )
                return (*base_state, *opponent_features)
        
        # 没有stats_manager或提取失败，返回默认特征
        default_features = (0.1, 0.4, 0.06, 0.0)  # 默认值
        return (*base_state, *default_features)
    
    def encode_challenge_state(
        self,
        round_base_info: str,
        round_action_info: str,
        challenge_decision_info: str,
        challenging_player_performance: str,
        hand: List[str],
        target_card: str,
        current_bullet: int,
        stats_manager=None  # 新增参数，暂时不使用
    ) -> Tuple:
        """
        编码质疑阶段的状态
        """
        # 获取基础状态
        base_state = list(self._encode_base_state(round_base_info, hand, target_card, current_bullet))
        
        # 提取上一个玩家声称打出的牌数（质疑阶段特有）
        claimed_cards = 0
        match = re.search(r'宣称打出(\d+)张', challenging_player_performance)
        if match:
            claimed_cards = int(match.group(1))
        
        # 添加质疑阶段特有特征
        base_state.append(claimed_cards)
        
        # 提取对手特征（如果有stats_manager）
        if stats_manager is not None:
            # 从 challenge_decision_info 提取上家名字
            opponent_name = self._extract_opponent_name(challenge_decision_info)
            
            if opponent_name:
                # 获取对手特征
                features = stats_manager.get_features(opponent_name, use_recent=True)
                base_state.extend([
                    features['cheat_rate'],
                    features['challenge_rate'],
                    features['aggression'],
                    features['confidence']
                ])
                return tuple(base_state)
        
        # 没有stats_manager或提取失败，返回默认特征
        base_state.extend([0.1, 0.4, 0.06, 0.0])  # 默认值
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
        round_id = int(re.search(r'第(\d+)轮', round_base_info).group(1))
        
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
        # max_hand_size 设成你游戏里“可能出现的最大手牌数”
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
        # （可选）安全校验：确保是手牌子集
        # for c in played_cards:
        #     assert c in hand, f"illegal decode: {played_cards} not subset of hand {hand}"
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
    
class FeatureExtractor:
    """
    提取游戏状态的特征向量
    """

    def get_features(self, state: Tuple) -> np.ndarray:
        """
        从游戏状态中提取有价值的特征向量
        
        参数:
            state: 游戏状态元组，包含以下元素：
                基础特征（9个）：
                - round_id: 轮次
                - target_card: 目标牌 ('Q', 'K', 或 'A')
                - current_bullet: 当前子弹位置
                - q_count: 手牌中Q的数量
                - k_count: 手牌中K的数量
                - a_count: 手牌中A的数量
                - joker_count: 手牌中Joker的数量
                - target_count: 手牌中可当作目标牌的数量
                - hand_size: 当前手牌总数
                
                质疑阶段额外特征（1个）：
                - claimed_cards: 上一个玩家声称打出的牌数
                
                对手特征（4个）：
                - opponent_cheat_rate: 对手作弊率
                - opponent_challenge_rate: 对手质疑率
                - opponent_aggression: 对手激进度
                - opponent_confidence: 统计置信度
        
        返回:
            特征向量
        """
        # 解析状态元组
        round_id = state[0]
        target_card = state[1]
        current_bullet = state[2]
        q_count = state[3]
        k_count = state[4]
        a_count = state[5]
        joker_count = state[6]
        target_count = state[7]
        hand_size = state[8]
        
        # 判断是否为质疑状态（有claimed_cards字段）
        # 出牌状态：9个基础 + 4个对手 = 13个元素
        # 质疑状态：9个基础 + 1个claimed_cards + 4个对手 = 14个元素
        is_challenge_state = len(state) == 14
        
        if is_challenge_state:
            claimed_cards = state[9]
            opponent_cheat_rate = state[10]
            opponent_challenge_rate = state[11]
            opponent_aggression = state[12]
            opponent_confidence = state[13]
        else:  # 出牌状态
            claimed_cards = 0
            opponent_cheat_rate = state[9]
            opponent_challenge_rate = state[10]
            opponent_aggression = state[11]
            opponent_confidence = state[12]
        
        features = []
        
        # 1. 游戏轮次特征
        features.append(round_id)
        
        # 2. 目标牌特征 (one-hot编码)
        target_card_features = self._encode_target_card(target_card)
        features.extend(target_card_features)
        
        # 3. 子弹位置特征
        features.append(current_bullet)
        features.append(6 - current_bullet)  # 距离枪膛底部的距离
        features.append(1 if current_bullet == 0 else 0)  # 是否在起始位置
        
        # 4. 目标牌充足程度特征
        features.append(target_count)  # 可使用的目标牌总数
        features.append(target_count / hand_size if hand_size > 0 else 0)  # 目标牌在手牌中的比例
        features.append(1 if target_count > 0 else 0)  # 是否有可用的目标牌
        
        # 5. 手牌大小特征
        features.append(hand_size)
        features.append(1 if hand_size == 0 else 0)  # 是否手牌为空
        
        # 6. 特殊牌特征
        features.append(joker_count)  # Joker数量
        features.append(1 if joker_count > 0 else 0)  # 是否有Joker
        
        # 7. 质疑相关特征 (仅在质疑状态下有效)
        features.append(claimed_cards)  # 对方声称打出的牌数
        
        # 8. 对手特征（新增）
        features.append(opponent_cheat_rate)       # 对手作弊率
        features.append(opponent_challenge_rate)   # 对手质疑率
        features.append(opponent_aggression)       # 对手激进度
        features.append(opponent_confidence)       # 统计置信度
        
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