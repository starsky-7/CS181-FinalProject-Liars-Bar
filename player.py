# player.py

import random
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


