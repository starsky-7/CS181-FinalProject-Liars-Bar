# player.py

import random
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from abc import ABC, abstractmethod

from LinearQAgent import LinearQAgent
from DQNAgent import DQNAgent
from rl_utils import StateEncoder, ActionDecoder


class BasePlayer(ABC):
    """
    所有玩家（手动 / Minimax / LinearQ）的统一接口。
    负责维护：
    - 基本状态（手牌、存活、子弹位置等）
    - 统一的决策函数签名
    """

    def __init__(self, name: str, showDetails: bool = True):
        self.name = name
        self.hand: List[str] = []
        self.alive: bool = True
        self.bullet_position: int = random.randint(0, 5)  # 子弹所在位置
        self.current_bullet_position: int = 0             # 当前弹仓位置
        self.opinions: Dict[str, str] = {}                # 对其他玩家的印象
        self.target_card: Optional[str] = None            # 当前局的目标牌（由 Game 设置）
        self.showDetails = showDetails
        
        # 心理侧写数据
        self.psych_profiles = defaultdict(lambda: {
            'bluff_count': 0,          # 诈唬次数
            'honest_count': 0,         # 诚实次数
            'challenge_aggressive': 0, # 质疑攻击性
            'play_conservative': 0,    # 出牌保守性
            'total_interactions': 0,   # 总交互次数
            'bluff_rate': 0.0,         # 诈唬率
            'challenge_rate': 0.0,     # 质疑率
            'recent_behavior': []      # 最近行为记录（最多10个）
        })

    def print_status(self) -> None:
        """打印玩家状态"""
        if self.showDetails:
            print(
                f"{self.name} - 手牌: {', '.join(self.hand)} - "
                f"子弹位置: {self.bullet_position} - 当前弹舱位置: {self.current_bullet_position}"
            )

    def init_opinions(self, other_players: List["BasePlayer"]) -> None:
        """初始化对其他玩家的看法"""
        self.opinions = {
            player.name: "还不了解这个玩家"
            for player in other_players
            if player.name != self.name
        }
        
        # 初始化心理侧写
        for player in other_players:
            if player.name != self.name:
                _ = self.psych_profiles[player.name]  # 确保有默认值

    def update_psych_profile(self, opponent_name: str, behavior_type: str, **kwargs):
        """
        更新对特定玩家的心理侧写
        
        Args:
            opponent_name: 对手名称
            behavior_type: 行为类型 ('bluff', 'honest', 'challenge', 'conservative', 'aggressive')
            **kwargs: 额外参数，如intensity
        """
        if opponent_name == self.name:
            return
            
        profile = self.psych_profiles[opponent_name]
        profile['total_interactions'] += 1
        
        if behavior_type == 'bluff':
            profile['bluff_count'] += 1
        elif behavior_type == 'honest':
            profile['honest_count'] += 1
        elif behavior_type == 'challenge':
            profile['challenge_aggressive'] += kwargs.get('intensity', 1)
        elif behavior_type == 'conservative':
            profile['play_conservative'] += 1
            
        # 计算比率
        total_plays = profile['bluff_count'] + profile['honest_count']
        if total_plays > 0:
            profile['bluff_rate'] = profile['bluff_count'] / total_plays
            
        # 记录最近行为（最多保存10个）
        profile['recent_behavior'].append(behavior_type)
        if len(profile['recent_behavior']) > 10:
            profile['recent_behavior'].pop(0)
            
    def get_psych_summary(self, opponent_name: str) -> str:
        """
        获取对特定玩家的心理侧写摘要
        """
        if opponent_name == self.name:
            return "自己"
            
        profile = self.psych_profiles[opponent_name]
        
        if profile['total_interactions'] == 0:
            return f"对{opponent_name}的了解：未知"
            
        summary_parts = []
        
        if profile['total_interactions'] >= 3:
            if profile['bluff_rate'] > 0.7:
                summary_parts.append(f"频繁诈唬（{profile['bluff_rate']*100:.1f}%概率）")
            elif profile['bluff_rate'] < 0.3:
                summary_parts.append(f"通常诚实")
                
            challenge_ratio = profile['challenge_aggressive'] / max(1, profile['total_interactions'])
            if challenge_ratio > 0.6:
                summary_parts.append(f"质疑攻击性强")
            elif challenge_ratio < 0.3:
                summary_parts.append(f"质疑谨慎")
                
            conservative_ratio = profile['play_conservative'] / max(1, profile['total_interactions'])
            if conservative_ratio > 0.7:
                summary_parts.append(f"出牌保守")
            elif conservative_ratio < 0.3:
                summary_parts.append(f"出牌激进")
                
        if summary_parts:
            return f"对{opponent_name}的了解：" + "，".join(summary_parts)
        else:
            return f"对{opponent_name}的了解：还在观察中（已互动{profile['total_interactions']}次）"
            
    def get_psych_features(self, opponent_name: str) -> List[float]:
        """
        获取心理侧写的数值特征（用于RL状态）
        返回5维向量：[诈唬率，质疑攻击性，出牌保守性，了解程度，最近行为模式]
        """
        if opponent_name == self.name:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
            
        profile = self.psych_profiles[opponent_name]
        
        features = [
            profile['bluff_rate'],  # 诈唬率
            profile['challenge_aggressive'] / max(1, profile['total_interactions']),  # 质疑攻击性
            profile['play_conservative'] / max(1, profile['total_interactions']),  # 出牌保守性
            min(profile['total_interactions'] / 10.0, 1.0)  # 了解程度（0-1）
        ]
        
        # 添加最近行为模式（使用滑动窗口统计）
        if profile['recent_behavior']:
            recent_bluff = profile['recent_behavior'].count('bluff') / len(profile['recent_behavior'])
            features.append(recent_bluff)
        else:
            features.append(0.0)
            
        return features

    @abstractmethod
    def choose_cards_to_play(
        self,
        round_base_info: str,
        round_action_info: str,
        play_decision_info: str,
    ) -> Tuple[Dict, str]:
        """
        选择出什么牌。

        返回：(result_dict, reasoning)
        - result_dict 必须包含:
          - "played_cards": List[str]
          - "behavior": str
          - "play_reason": str
        - reasoning: 额外"思考过程"，可以是空字符串
        """
        ...
        

    @abstractmethod
    def decide_challenge(
        self,
        round_base_info: str,
        round_action_info: str,
        challenge_decision_info: str,
        challenging_player_performance: str,
        extra_hint: str,
    ) -> Tuple[Dict, str]:
        """
        决定是否质疑上一位玩家。

        返回：(result_dict, reasoning)
        - result_dict 必须包含:
          - "was_challenged": bool
          - "challenge_reason": str
        - reasoning: 额外"思考过程"，可以是空字符串
        """
        ...

    def reflect(
        self,
        alive_players: List[str],
        round_base_info: str,
        round_action_info: str,
        round_result: str,
    ) -> None:
        """
        轮次结束后的反思过程。

        默认实现：什么都不做。
        - LinearQPlayer 可以在这里更新 Q 表
        """
        pass

    def process_penalty(self) -> bool:
        """处理惩罚：俄罗斯轮盘赌逻辑"""
        if self.showDetails:
            print(f"玩家 {self.name} 执行射击惩罚：")
        self.print_status()
        if self.bullet_position == self.current_bullet_position:
            if self.showDetails:
                print(f"{self.name} 中枪死亡！")
            self.alive = False
        else:
            if self.showDetails:
                print(f"{self.name} 幸免于难！")
        self.current_bullet_position = (self.current_bullet_position + 1) % 6
        return self.alive


# ================== 1. 手动玩家 ==================


class ManualPlayer(BasePlayer):
    """
    手动操作的玩家（命令行）。
    用于 baseline 或 debug。
    """

    def choose_cards_to_play(
        self,
        round_base_info: str,
        round_action_info: str,
        play_decision_info: str,
    ) -> Tuple[Dict, str]:
        print("============== 手动玩家出牌阶段 ==============")
        print(round_base_info)
        if round_action_info:
            print(round_action_info)
        if play_decision_info:
            print(play_decision_info)
        print(f"你的手牌：{self.hand}")

        while True:
            raw = input(
                f"[{self.name}] 输入要出的牌，用逗号分隔（例如 'Q,Q' 或留空表示不出）："
            ).strip()
            if not raw:
                played_cards: List[str] = []
            else:
                played_cards = [c.strip() for c in raw.split(",") if c.strip()]

            # 校验：都是自己手牌
            if all(card in self.hand for card in played_cards):
                break
            print("出牌不合法：有牌不在你的手牌中，请重新输入。")

        # 从手牌中移除已出的牌
        for card in played_cards:
            self.hand.remove(card)

        behavior = input(f"[{self.name}] 简要描述一下你的表演/行为（可空）：").strip() or "无"
        reason = input(f"[{self.name}] 为什么这么出牌？（可空）：").strip() or "无"

        result = {
            "played_cards": played_cards,
            "behavior": behavior,
            "play_reason": reason,
        }
        return result, ""  # reasoning 留空

    def decide_challenge(
        self,
        round_base_info: str,
        round_action_info: str,
        challenge_decision_info: str,
        challenging_player_performance: str,
        extra_hint: str,
    ) -> Tuple[Dict, str]:
        print("============== 手动玩家质疑阶段 ==============")
        print(round_base_info)
        if round_action_info:
            print(round_action_info)
        if challenge_decision_info:
            print(challenge_decision_info)
        if challenging_player_performance:
            print(challenging_player_performance)
        if extra_hint:
            print(extra_hint)
        print(f"你的手牌：{self.hand}")

        while True:
            choice = input(f"[{self.name}] 是否质疑上一位玩家？(y/n)：").strip().lower()
            if choice in ("y", "n"):
                break
            print("请输入 y 或 n。")

        if choice == "y":
            reason = input(f"[{self.name}] 质疑理由（可空）：").strip() or "直觉"
            result = {"was_challenged": True, "challenge_reason": reason}
        else:
            reason = input(f"[{self.name}] 不质疑的理由（可空）：").strip() or "觉得他说的是真的"
            result = {"was_challenged": False, "challenge_reason": reason}

        return result, ""  # reasoning 留空

# ================== 2. 强化学习玩家 ==================

# 在player.py中修改RLPlayer类

class RLPlayer(BasePlayer):
    """
    使用强化学习训练的AI玩家，集成轻量心理侧写
    """
    
    def __init__(self, name: str, showDetails: bool, agent: LinearQAgent | DQNAgent, is_training: bool = True):
        super().__init__(name, showDetails)
        self.agent = agent
        self.is_training = is_training
        self.prev_state = None
        self.prev_action = None
        self.state_encoder = StateEncoder()
        self.action_decoder = ActionDecoder()
        self.current_reward = 0
        
        # 轻量心理侧写系统
        self.opponent_profiles = {}  # 对手名 -> 心理侧写
        
    def _init_opponent_profile(self, opponent_name: str):
        """初始化对手心理侧写"""
        if opponent_name not in self.opponent_profiles:
            self.opponent_profiles[opponent_name] = {
                'bluff_count': 0,          # 诈唬次数
                'honest_count': 0,         # 诚实次数
                'challenge_aggressive': 0, # 质疑次数
                'total_interactions': 0,   # 总交互次数
                'recent_outcomes': [],     # 最近5次交互结果
                'trust_score': 0.5,        # 信任度 (0-1)
                'aggression_score': 0.5,   # 攻击性 (0-1)
            }
    
    def update_opponent_profile(self, opponent_name: str, interaction_type: str, outcome: str):
        """
        更新对手心理侧写
        
        Args:
            opponent_name: 对手名称
            interaction_type: 交互类型 ('play', 'challenge')
            outcome: 结果 ('success', 'failure', 'bluff', 'honest')
        """
        self._init_opponent_profile(opponent_name)
        profile = self.opponent_profiles[opponent_name]
        profile['total_interactions'] += 1
        
        if interaction_type == 'challenge' and outcome == 'success':
            profile['challenge_aggressive'] += 1
            # 质疑成功增加攻击性分数
            profile['aggression_score'] = min(1.0, profile['aggression_score'] + 0.1)
        elif interaction_type == 'play':
            if outcome == 'bluff':
                profile['bluff_count'] += 1
                profile['trust_score'] = max(0.0, profile['trust_score'] - 0.15)
            elif outcome == 'honest':
                profile['honest_count'] += 1
                profile['trust_score'] = min(1.0, profile['trust_score'] + 0.1)
        
        # 更新最近结果记录
        profile['recent_outcomes'].append(f"{interaction_type}:{outcome}")
        if len(profile['recent_outcomes']) > 5:
            profile['recent_outcomes'].pop(0)
            
        # 计算综合信任度（考虑最近趋势）
        total_plays = profile['bluff_count'] + profile['honest_count']
        if total_plays > 0:
            base_trust = 1.0 - (profile['bluff_count'] / total_plays)
            # 加权平均：70%基于历史，30%基于最近趋势
            recent_bluff = sum(1 for r in profile['recent_outcomes'] if 'bluff' in r) / len(profile['recent_outcomes'])
            profile['trust_score'] = 0.7 * base_trust + 0.3 * (1.0 - recent_bluff)
    
    def get_psych_features(self, opponent_name: str):
        """获取对手的心理特征向量（用于状态编码）"""
        self._init_opponent_profile(opponent_name)
        profile = self.opponent_profiles[opponent_name]
        
        features = [
            profile['trust_score'],  # 信任度
            profile['aggression_score'],  # 攻击性
            min(profile['total_interactions'] / 20.0, 1.0),  # 熟悉度
        ]
        
        # 添加趋势特征
        if len(profile['recent_outcomes']) >= 3:
            recent_trend = sum(1 for r in profile['recent_outcomes'][-3:] if 'bluff' in r or 'success' in r)
            features.append(recent_trend / 3.0)  # 最近攻击性
        else:
            features.append(0.5)  # 默认值
        
        return features
    
    def choose_cards_to_play(self, round_base_info, round_action_info, play_decision_info):
        # 解析下家名称
        import re
        next_player_match = re.search(r'([^ ]+)是你的下家', play_decision_info)
        next_player_name = next_player_match.group(1) if next_player_match else "Unknown"
        
        # 获取心理特征
        psych_features = self.get_psych_features(next_player_name)
        
        # 编码当前状态（修改StateEncoder以接收心理特征）
        state = self.state_encoder.encode_play_state(
            round_base_info,
            round_action_info,
            play_decision_info,
            self.hand,
            self.target_card,
            self.current_bullet_position,
            psych_features  # 添加心理特征
        )
        
        # 获取动作掩码
        N = self.action_decoder.num_play_actions()
        mask = self.action_decoder.get_total_action_mask("play", self.hand)
        
        # 更新Q网络
        if self.prev_state is not None and self.prev_action is not None:
            self.agent.update(self.prev_state, self.prev_action, self.current_reward, 
                            state, mask, done=False, is_training=self.is_training)
        
        # 选择动作（基于心理特征调整探索率）
        original_epsilon = self.agent.epsilon
        if next_player_name != "Unknown":
            profile = self.opponent_profiles[next_player_name]
            # 面对高攻击性对手，降低探索，更依赖学习策略
            if profile['aggression_score'] > 0.7:
                self.agent.epsilon = max(0.01, original_epsilon * 0.5)
        
        action_idx = self.agent.choose_action(state, mask, self.is_training)
        
        # 恢复epsilon
        self.agent.epsilon = original_epsilon
        
        # 解码动作
        action = self.action_decoder.decode_play_action(action_idx, self.hand)
        
        # 记录状态和动作
        self.prev_state = state
        self.prev_action = action_idx
        
        # 从手牌中移除已出的牌
        for card in action["played_cards"]:
            self.hand.remove(card)
        
        # 记录出牌信息（用于后续心理侧写更新）
        self.last_play_info = {
            'cards': action["played_cards"],
            'target_card': self.target_card,
            'next_player': next_player_name
        }
        
        result = {
            "played_cards": action["played_cards"],
            "behavior": "RL出牌(含心理侧写)",
            "play_reason": "RL决策",
        }
        
        self.current_reward = 0
        return result, f"RL选择出牌: {action['played_cards']}"
    
    def decide_challenge(self, round_base_info, round_action_info, challenge_decision_info,
                        challenging_player_performance, extra_hint):
        # 解析上家名称
        import re
        prev_player_match = re.search(r'^([^ ]+)', challenging_player_performance)
        prev_player_name = prev_player_match.group(1) if prev_player_match else "Unknown"
        
        # 获取心理特征
        psych_features = self.get_psych_features(prev_player_name)
        
        # 编码当前状态
        state = self.state_encoder.encode_challenge_state(
            round_base_info,
            round_action_info,
            challenge_decision_info,
            challenging_player_performance,
            self.hand,
            self.target_card,
            self.current_bullet_position,
            psych_features  # 添加心理特征
        )
        
        # 获取动作掩码
        N = self.action_decoder.num_play_actions()
        mask = self.action_decoder.get_total_action_mask("challenge", self.hand)
        
        # 更新Q网络
        if self.prev_state is not None and self.prev_action is not None:
            self.agent.update(self.prev_state, self.prev_action, self.current_reward, 
                            state, mask, done=False, is_training=self.is_training)
        
        # 选择动作（基于心理特征调整决策）
        action_idx = self.agent.choose_action(state, mask, self.is_training)
        was_challenged = (action_idx == N + 1)
        
        # 如果对手诈唬率高，增加质疑倾向
        if prev_player_name != "Unknown":
            profile = self.opponent_profiles[prev_player_name]
            if profile['trust_score'] < 0.3 and not was_challenged:
                # 低信任度对手，强制质疑（有一定概率）
                if random.random() < 0.7:
                    was_challenged = True
                    action_idx = N + 1
        
        # 记录状态和动作
        self.prev_state = state
        self.prev_action = action_idx
        
        result = {
            "was_challenged": was_challenged,
            "challenge_reason": f"RL决策(信任度:{self.opponent_profiles.get(prev_player_name, {}).get('trust_score', 0.5):.2f})",
        }
        
        self.current_reward = 0
        return result, f"RL选择{'质疑' if was_challenged else '不质疑'}"
    
    def reflect(self, alive_players, round_base_info, round_action_info, round_result):
        """重写反思方法，更新心理侧写"""
        # 解析游戏记录，更新对手心理侧写
        self._update_profiles_from_history(round_action_info, round_result)
        
        # 原有奖励计算逻辑
        if "死亡" in round_result and "你" in round_result:
            self.current_reward = -100
        elif "活着" in round_result and "你" in round_result:
            self.current_reward = -50
        elif "死亡" in round_result and "你" not in round_result:
            self.current_reward = 100
        elif "质疑成功" in round_action_info and "你" not in round_result:
            self.current_reward = 20
        elif "质疑失败" in round_action_info and "你" in round_result:
            self.current_reward = -20
        
        # RL更新
        if not self.alive or len(alive_players) == 1:
            if self.prev_state is not None and self.prev_action is not None:
                self.agent.update(self.prev_state, self.prev_action, self.current_reward, 
                                None, [], done=True, is_training=self.is_training)
                self.prev_state = None
                self.prev_action = None
    
    def _update_profiles_from_history(self, round_action_info: str, round_result: str):
        """从游戏历史更新心理侧写"""
        lines = round_action_info.split('\n')
        
        for i, line in enumerate(lines):
            # 解析出牌记录
            if '宣称打出' in line and '轮到你出牌' not in line:
                match = re.search(r'轮到(\w+)出牌，(\w+)宣称打出(\d+)张', line)
                if match:
                    player_name = match.group(2)
                    
                    # 查找后续的质疑结果
                    for j in range(i, min(i+3, len(lines))):
                        if '质疑' in lines[j]:
                            if '质疑成功' in lines[j]:
                                # 质疑成功，说明该玩家诈唬
                                self.update_opponent_profile(player_name, 'play', 'bluff')
                            elif '质疑失败' in lines[j]:
                                # 质疑失败，说明该玩家诚实
                                self.update_opponent_profile(player_name, 'play', 'honest')
                            break
            
            # 解析质疑行为
            if '选择质疑' in line and '你选择质疑' not in line:
                match = re.search(r'(\w+)选择质疑(\w+)', line)
                if match:
                    challenger = match.group(1)
                    challenged = match.group(2)
                    
                    # 更新质疑者的攻击性
                    self.update_opponent_profile(challenger, 'challenge', 'success')
    
# ================== 3. 简单策略玩家 ==================

class SimpleStrategyPlayer(BasePlayer):
    """
    使用最简单策略的玩家：
    - 出牌阶段：只出真牌（目标牌或Joker），如果没有则不出牌
    - 质疑阶段：总是质疑
    """

    def choose_cards_to_play(
        self,
        round_base_info: str,
        round_action_info: str,
        play_decision_info: str,
    ) -> Tuple[Dict, str]:
        # 选择所有符合目标牌的牌（目标牌或Joker）
        valid_cards = [card for card in self.hand if card == self.target_card or card == 'Joker']
        
        # 从手牌中移除已出的牌
        for card in valid_cards:
            self.hand.remove(card)

        result = {
            "played_cards": valid_cards,
            "behavior": "简单策略：只出真牌",
            "play_reason": f"出{len(valid_cards)}张真牌",
        }
        return result, f"简单策略：出真牌 {valid_cards}"

    def decide_challenge(
        self,
        round_base_info: str,
        round_action_info: str,
        challenge_decision_info: str,
        challenging_player_performance: str,
        extra_hint: str,
    ) -> Tuple[Dict, str]:
        # 总是质疑
        result = {
            "was_challenged": True,
            "challenge_reason": "简单策略：总是质疑",
        }
        return result, "简单策略：总是质疑"
    
# ================== 4. smarter玩家 ==================

class SmarterStrategyPlayer(BasePlayer):
    """
    使用更智能策略的玩家：
    - 出牌阶段：优先出Joker，其次出目标牌
    - 质疑阶段：以一个随机数作为阈值，超过阈值则质疑，否则不质疑
    """
    def choose_cards_to_play(
        self,
        round_base_info: str,
        round_action_info: str,
        play_decision_info: str,
    ) -> Tuple[Dict, str]:
        # 优先出Joker，其次出目标牌
        valid_cards = [card for card in self.hand if card == 'Joker'] + [card for card in self.hand if card == self.target_card]
        
        # 从手牌中移除已出的牌
        for card in valid_cards:
            self.hand.remove(card)

        result = {
            "played_cards": valid_cards,
            "behavior": "更智能策略：优先出Joker，其次出目标牌",
            "play_reason": f"出{len(valid_cards)}张牌",
        }
        return result, f"更智能策略：出牌 {valid_cards}"
    
    def decide_challenge(
        self,
        round_base_info: str,
        round_action_info: str,
        challenge_decision_info: str,
        challenging_player_performance: str,
        extra_hint: str,
    ) -> Tuple[Dict, str]:
        # 以一个随机数作为阈值，超过阈值则质疑，否则不质疑
        threshold = random.uniform(0, 1)
        result = {
            "was_challenged": threshold > 0.5,
            "challenge_reason": "更智能策略：以随机数作为阈值，超过阈值则质疑，否则不质疑",
        }
        return result, f"更智能策略：{'质疑' if result['was_challenged'] else '不质疑'}"