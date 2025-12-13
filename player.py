# player.py

import random
import numpy as np
from rl_agent import RLAgent
from rl_utils import StateEncoder, ActionDecoder
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import itertools
import math


class BasePlayer(ABC):
    """
    所有玩家（手动 / Minimax / QLearning）的统一接口。
    负责维护：
    - 基本状态（手牌、存活、子弹位置等）
    - 统一的决策函数签名
    """

    def __init__(self, name: str):
        self.name = name
        self.hand: List[str] = []
        self.alive: bool = True
        self.bullet_position: int = random.randint(0, 5)  # 子弹所在位置
        self.current_bullet_position: int = 0             # 当前弹仓位置
        self.opinions: Dict[str, str] = {}                # 对其他玩家的印象
        self.target_card: Optional[str] = None            # 当前局的目标牌（由 Game 设置）

    def print_status(self) -> None:
        """打印玩家状态"""
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
        - reasoning: 额外“思考过程”，可以是空字符串
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
        - reasoning: 额外“思考过程”，可以是空字符串
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
        - QLearningPlayer 可以在这里更新 Q 表
        """
        pass

    def process_penalty(self) -> bool:
        """处理惩罚：俄罗斯轮盘赌逻辑"""
        print(f"玩家 {self.name} 执行射击惩罚：")
        self.print_status()
        if self.bullet_position == self.current_bullet_position:
            print(f"{self.name} 中枪死亡！")
            self.alive = False
        else:
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

# ================== 2. 简单策略玩家 ==================


class SimpleStrategyPlayer(BasePlayer):
    """
    使用最简单策略的AI玩家：
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

# ================== 3. 强化学习玩家 ==================

class RLPlayer(BasePlayer):
    """
    使用强化学习训练的AI玩家
    """

    def __init__(self, name: str, agent: RLAgent, is_training: bool = True):
        super().__init__(name)
        self.agent = agent
        self.is_training = is_training
        self.prev_state = None
        self.prev_action = None
        self.state_encoder = StateEncoder()
        self.action_decoder = ActionDecoder()
        self.current_reward = 0  # 当前步骤的奖励

    def choose_cards_to_play(
        self,
        round_base_info: str,
        round_action_info: str,
        play_decision_info: str,
    ) -> Tuple[Dict, str]:
        """
        使用RL代理选择出牌
        """
        # 编码当前状态
        state = self.state_encoder.encode_play_state(
            round_base_info,
            round_action_info,
            play_decision_info,
            self.hand,
            self.target_card,
            self.current_bullet_position
        )

        # 记录前一个状态和动作（用于更新）
        if self.prev_state is not None and self.prev_action is not None:
            self.agent.update(self.prev_state, self.prev_action, self.current_reward, state, self.is_training)

        # 获取合法动作空间
        legal_actions = self.action_decoder.get_legal_play_actions(self.hand)
        
        # 选择动作
        action_idx = self.agent.choose_action(state, legal_actions, self.is_training)
        action = self.action_decoder.decode_play_action(action_idx, self.hand)
        
        # 记录当前状态和动作
        self.prev_state = state
        self.prev_action = action_idx
        
        # 从手牌中移除已出的牌
        for card in action["played_cards"]:
            self.hand.remove(card)

        result = {
            "played_cards": action["played_cards"],
            "behavior": "RL出牌",
            "play_reason": "RL决策",
        }
        
        # 重置当前奖励
        self.current_reward = 0
        
        return result, f"RL选择出牌: {action['played_cards']}"

    def decide_challenge(
        self,
        round_base_info: str,
        round_action_info: str,
        challenge_decision_info: str,
        challenging_player_performance: str,
        extra_hint: str,
    ) -> Tuple[Dict, str]:
        """
        使用RL代理决定是否质疑
        """
        # 编码当前状态
        state = self.state_encoder.encode_challenge_state(
            round_base_info,
            round_action_info,
            challenge_decision_info,
            challenging_player_performance,
            self.hand,
            self.target_card,
            self.current_bullet_position,
            
        )

        # 记录前一个状态和动作（用于更新）
        if self.prev_state is not None and self.prev_action is not None:
            self.agent.update(self.prev_state, self.prev_action, self.current_reward, state, self.is_training)

        # 获取合法动作空间（质疑或不质疑）
        legal_actions = [0, 1]  # 0: 不质疑, 1: 质疑
        
        # 选择动作
        action_idx = self.agent.choose_action(state, legal_actions, self.is_training)
        was_challenged = bool(action_idx)
        
        # 记录当前状态和动作
        self.prev_state = state
        self.prev_action = action_idx

        result = {
            "was_challenged": was_challenged,
            "challenge_reason": "RL决策",
        }
        
        # 重置当前奖励
        self.current_reward = 0
        
        return result, f"RL选择{'质疑' if was_challenged else '不质疑'}"

    def reflect(
        self,
        alive_players: List[str],
        round_base_info: str,
        round_action_info: str,
        round_result: str,
    ) -> None:
        """
        处理奖励和学习
        对局信息示例：
        000000000000round_base_info: 现在是第2轮，目标牌：K，本轮玩家：RL_Player、Opponent，从玩家Opponent开始

        111111111111round_action_info: 轮到Opponent出牌，Opponent宣称打出1张'K'，剩余手牌4张
        Opponent 的表现：简单策略：只出真牌
        你选择不质疑Opponent
        轮到你出牌，你打出2张牌，出牌：K、K，剩余手牌：Q、A、K
        你的表现：RL出牌
        Opponent选择质疑你，你打出的牌是：K、K，质疑失败

        222222222222round_result: Opponent开枪！没有命中，Opponent还活着
        """
        print(f'000000000000round_base_info: {round_base_info}')
        print(f"111111111111round_action_info: {round_action_info}")
        print(f"222222222222round_result: {round_result}")
        # 计算奖励
        if "死亡" in round_result and "你" in round_result:
            self.current_reward = -100  # 玩家死亡，给予大惩罚
        elif "活着" in round_result and "你" in round_result:
            self.current_reward = -50  # xx玩家存活，给予奖励xx ---------> 你在奖励什么？？？出现这个情况，说明你开枪了，都已经输了一半了哥们，还搁那奖励呢
        elif "死亡" in round_result and "你" not in round_result:
            self.current_reward = 100  # 对手死亡，给予大奖励
        # elif "活着" in round_result and "你" not in round_result:  这个应该不需要惩罚，因为出现这个情况，说明你已经获利
        #     self.current_reward = -50  # 对手存活，给予惩罚
        elif "质疑成功" in round_action_info and "你" not in round_result:
            self.current_reward = 20  # 质疑成功，给予奖励
        elif "质疑失败" in round_action_info and "你" in round_result:
            self.current_reward = -20  # 质疑失败，给予惩罚

        print(f"Player {self.name} received reward: {self.current_reward}")
        
        # 如果游戏结束，进行最终更新
        if not self.alive or len(alive_players) == 1:
            if self.prev_state is not None and self.prev_action is not None:
                # 最终状态设置为None表示游戏结束
                self.agent.update(self.prev_state, self.prev_action, self.current_reward, None)
                self.prev_state = None
                self.prev_action = None