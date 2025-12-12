from typing import List, Dict, Tuple
import re

class StateEncoder:
    """
    将游戏状态编码为强化学习代理可以理解的形式
    """
    
    def encode_play_state(
        self,
        round_base_info: str,
        round_action_info: str,
        play_decision_info: str,
        hand: List[str],
        target_card: str,
        current_bullet: int
    ) -> Tuple:
        """
        编码出牌阶段的状态
        """
        # 获取基础状态
        base_state = self._encode_base_state(round_base_info, hand, target_card, current_bullet)
        
        # 出牌阶段不需要额外特征
        return base_state
    
    def encode_challenge_state(
        self,
        round_base_info: str,
        round_action_info: str,
        challenge_decision_info: str,
        challenging_player_performance: str,
        hand: List[str],
        target_card: str,
        current_bullet: int
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

class ActionDecoder:
    """
    将强化学习代理的动作转换为游戏可以理解的形式
    """
    
    def _get_all_play_actions(self, hand: List[str]) -> List[Tuple[str]]:
        """
        生成所有可能的出牌动作组合（私有辅助方法）
        """
        # 生成所有可能的出牌组合
        all_combinations = []
        for i in range(1, len(hand) + 1):
            all_combinations.extend(self._get_combinations(hand, i))
        
        # 去重并添加不出牌的动作
        unique_actions = [tuple(sorted(c)) for c in all_combinations]
        unique_actions = list(set(unique_actions))
        unique_actions.append(())  # 不出牌
        
        return unique_actions
    
    def get_legal_play_actions(self, hand: List[str]) -> List[int]:
        """
        获取合法的出牌动作索引列表
        """
        all_actions = self._get_all_play_actions(hand)
        return list(range(len(all_actions)))
    
    def decode_play_action(self, action_idx: int, hand: List[str]) -> Dict:
        """
        解码出牌动作，将动作索引转换为实际的出牌组合
        """
        all_actions = self._get_all_play_actions(hand)
        played_cards = list(all_actions[action_idx])
        return {"played_cards": played_cards}
    
    def _get_combinations(self, hand: List[str], length: int) -> List[List[str]]:
        """
        获取指定长度的所有组合
        """
        if length == 0:
            return [[]]
        if not hand:
            return []
        
        combinations = []
        for i in range(len(hand)):
            # 选择当前牌
            current_card = hand[i]
            # 递归获取剩余牌的组合
            remaining = hand[i+1:]
            for combo in self._get_combinations(remaining, length-1):
                combinations.append([current_card] + combo)
        
        return combinations