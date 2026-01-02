# player.py

import random
import numpy as np
from LinearQAgent import LinearQAgent
from DQNAgent import DQNAgent
from rl_utils import StateEncoder, ActionDecoder
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import itertools
import math


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

class RLPlayer(BasePlayer):
    """
    使用强化学习训练的AI玩家
    """

    def __init__(self, name: str, showDetails: bool, agent: LinearQAgent | DQNAgent, is_training: bool = True, stats_manager=None):
        super().__init__(name, showDetails)
        self.agent = agent
        self.is_training = is_training
        self.stats_manager = stats_manager  # 新增：对手统计管理器
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
        # 编码当前状态（传递stats_manager，暂时不使用）
        state = self.state_encoder.encode_play_state(
            round_base_info,
            round_action_info,
            play_decision_info,
            self.hand,
            self.target_card,
            self.current_bullet_position,
            stats_manager=self.stats_manager  # 新增参数
        )
        # 获取全局动作空间（play phase）
        N = self.action_decoder.num_play_actions()
        mask = self.action_decoder.get_total_action_mask("play", self.hand)  # shape = (N+2,)
        
        # # 过滤出合法动作
        # legal_actions = [i for i, m in enumerate(mask) if m > 0.5]

        # 记录前一个状态和动作（用于更新）
        if self.prev_state is not None and self.prev_action is not None:
            self.agent.update(self.prev_state, self.prev_action, self.current_reward, state, mask, done=False, is_training=self.is_training)
  
        # 选择动作
        action_idx = self.agent.choose_action(state, mask, self.is_training)
        # assert action_idx < N, "play 阶段不应选择 challenge 动作（mask 应该已屏蔽）"
        action = self.action_decoder.decode_play_action(action_idx, self.hand)

        
        # 记录当前状态和动作 (注意：现在的prev_action存的是全局动作索引)
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
        # 编码当前状态（传递stats_manager，暂时不使用）
        state = self.state_encoder.encode_challenge_state(
            round_base_info,
            round_action_info,
            challenge_decision_info,
            challenging_player_performance,
            self.hand,
            self.target_card,
            self.current_bullet_position,
            stats_manager=self.stats_manager  # 新增参数
        )
        # 获取全局动作空间（challenge phase）
        N = self.action_decoder.num_play_actions()
        mask = self.action_decoder.get_total_action_mask("challenge", self.hand)  # shape = (N+2,)
        
        # # 过滤出合法动作
        # legal_actions = [i for i, m in enumerate(mask) if m > 0.5]
        # 为保持接口统一，只传入mask

        # 记录前一个状态和动作（用于更新）
        if self.prev_state is not None and self.prev_action is not None:
            self.agent.update(self.prev_state, self.prev_action, self.current_reward, state, mask, done=False, is_training=self.is_training)
        
        # 选择动作
        action_idx = self.agent.choose_action(state, mask, self.is_training)
        # assert action_idx in (N, N+1), "challenge 阶段只应选择 N(不质疑) 或 N+1(质疑)"
        was_challenged = (action_idx == N + 1)
                
        # 记录当前状态和动作 (注意：现在的prev_action存的是全局动作索引)
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
        # print(f'000000000000round_base_info: {round_base_info}')
        # print(f"111111111111round_action_info: {round_action_info}")
        # print(f"222222222222round_result: {round_result}")
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

        # print(f"Player {self.name} received reward: {self.current_reward}")
        
        # 如果游戏结束，进行最终更新
        if not self.alive or len(alive_players) == 1:
            if self.prev_state is not None and self.prev_action is not None:
                # 最终状态设置为None表示游戏结束
                self.agent.update(self.prev_state, self.prev_action, self.current_reward, None, [], done=True, is_training=self.is_training)
                self.prev_state = None
                self.prev_action = None

# ================== 2. 简单策略玩家 ==================

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
    
# ================== 3. smarter玩家 ==================

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
    
# ================== 4. human-like玩家 ==================

class HumanLikeStrategyPlayer(BasePlayer):
    """
    使用类人策略的玩家：
    - 出牌阶段：读取当前游戏状态，根据游戏规则和玩家策略出牌
    - 质疑阶段：根据当前游戏状态和玩家策略，判断是否质疑
    - 具体来说：
        1. 基于统计学实时更新其他玩家的特征：出假牌概率、质疑概率
        2. 出牌时：如果下家质疑概率高，出真牌；否则，有探索倾向出假牌
        3. 质疑时：如果上家出假牌概率高，质疑；否则，有探索倾向质疑
    """

    def __init__(self, name: str, showDetails: bool = True):
        super().__init__(name, showDetails)
        # 针对每个对手维护统计信息
        self.opponent_stats = {}
        self.total_rounds = 0
        self.max_rounds = 500  # 假设一局最多20轮，可根据实际情况调整
        self.init_opinions([])  # 需要在Game初始化后再次调用

    def init_opinions(self, other_players: List["BasePlayer"]):
        super().init_opinions(other_players)
        # 不重置 opponent_stats，只在第一次初始化时创建
        if not hasattr(self, 'opponent_stats'):
            self.opponent_stats = {}
        for player in other_players:
            if player.name != self.name:
                self.opponent_stats[player.name] = {
                    'total_plays': 0,
                    'fake_plays': 0,
                    'total_challenges': 0,
                    'challenges_made': 0
                }

    def get_exploration_rate(self):
        # 分阶段递减探索率，前期高后期低
        progress = min(self.total_rounds / self.max_rounds, 1.0)
        return 0.5 * (1 - progress) + 0.05  # 0.55->0.05线性递减

    def choose_cards_to_play(
        self,
        round_base_info: str,
        round_action_info: str,
        play_decision_info: str,
    ) -> Tuple[Dict, str]:
        self.total_rounds += 1
        exploration_rate = self.get_exploration_rate()

        # 方法1：从游戏信息中解析真实玩家顺序
        def parse_player_order_from_info(info: str):
            """尝试从游戏信息中解析玩家顺序"""
            lines = info.split('\n')
            for line in lines:
                # 尝试查找包含玩家顺序的信息
                if '玩家顺序' in line or '顺序' in line or 'players' in line.lower():
                    # 尝试提取方括号中的内容
                    import re
                    match = re.search(r'\[(.*?)\]', line)
                    if match:
                        players = [p.strip() for p in match.group(1).split(',')]
                        return players
                    # 或者提取圆括号中的内容
                    match = re.search(r'\((.*?)\)', line)
                    if match:
                        players = [p.strip() for p in match.group(1).split(',')]
                        return players
            return None

        # 尝试获取真实玩家顺序
        player_order = parse_player_order_from_info(round_base_info)
        
        if not player_order:
            # 方法2：如果无法解析，使用创建时的默认顺序
            # 假设创建顺序是固定的
            player_order = ['RL_Player', 'Opponent1', 'Opponent2', 'Opponent3']
            # print(f"无法从游戏信息解析顺序，使用默认顺序: {player_order}")
        else:
            print(f"从游戏信息解析的玩家顺序: {player_order}")

        # 找到自己在顺序中的位置
        if self.name in player_order:
            current_index = player_order.index(self.name)
            # 计算真实下家
            next_index = (current_index + 1) % len(player_order)
            next_player = player_order[next_index]
            
            # 如果下家是自己（不应该发生，但安全起见）
            if next_player == self.name and len(player_order) > 1:
                next_index = (next_index + 1) % len(player_order)
                next_player = player_order[next_index]
        else:
            # 方法3：如果自己不在顺序中，使用旧逻辑
            print(f"警告：{self.name} 不在玩家顺序中，使用旧逻辑")
            next_players = [n for n in self.opinions.keys() if self.opponent_stats.get(n)]
            next_player = next_players[0] if next_players else None

        # 计算下家的质疑概率
        challenge_prob = 0.0
        if next_player and next_player in self.opponent_stats:
            stats = self.opponent_stats[next_player]
            challenge_prob = stats['challenges_made'] / max(1, stats['total_challenges'])
        
        # print(f"玩家: {self.name}, 真实下家: {next_player}, 质疑概率: {challenge_prob:.4f}")
        # print(f"DEBUG - opinions.keys(): {list(self.opinions.keys())}")
        # print(f"DEBUG - opponent_stats.keys(): {list(self.opponent_stats.keys())}")

        # 决定是否出真牌
        if challenge_prob > 0.5:
            # 下家质疑概率高，出真牌
            valid_cards = [card for card in self.hand if card == self.target_card or card == 'Joker']
            # print(f"决策：下家质疑概率高({challenge_prob:.2f}>0.5)，出真牌")
        else:
            if random.random() < exploration_rate:
                # 探索：尝试出假牌
                valid_cards = [card for card in self.hand if card != self.target_card and card != 'Joker']
                if not valid_cards:
                    # 没有假牌可出，出真牌
                    valid_cards = [card for card in self.hand if card == self.target_card or card == 'Joker']
                    # print(f"决策：探索出假牌但无假牌可出，出真牌")
                # else:
                    # print(f"决策：探索出假牌，探索率{exploration_rate:.2f}")
            else:
                # 利用：出真牌
                valid_cards = [card for card in self.hand if card == self.target_card or card == 'Joker']
                # print(f"决策：利用出真牌，探索率{exploration_rate:.2f}")

        # 从手牌中移除选择的牌
        selected_cards = valid_cards.copy()  # 先复制
        for card in selected_cards:
            self.hand.remove(card)

        result = {
            "played_cards": selected_cards,
            "behavior": "类人策略：基于下家质疑概率决策",
            "play_reason": f"出{len(selected_cards)}张牌，下家({next_player})质疑概率{challenge_prob:.2f}，探索率{exploration_rate:.2f}",
        }
        return result, f"类人策略：出牌 {selected_cards}"

    def decide_challenge(
        self,
        round_base_info: str,
        round_action_info: str,
        challenge_decision_info: str,
        challenging_player_performance: str,
        extra_hint: str,
    ) -> Tuple[Dict, str]:
        exploration_rate = self.get_exploration_rate()
        
        # 同样解析玩家顺序
        def parse_player_order_from_info(info: str):
            lines = info.split('\n')
            for line in lines:
                if '玩家顺序' in line or '顺序' in line:
                    import re
                    match = re.search(r'\[(.*?)\]', line)
                    if match:
                        players = [p.strip() for p in match.group(1).split(',')]
                        return players
            return None
        
        player_order = parse_player_order_from_info(round_base_info)
        if not player_order:
            player_order = ['RL_Player', 'Opponent1', 'Opponent2', 'Opponent3']
        
        # 找到自己和上家
        if self.name in player_order:
            current_index = player_order.index(self.name)
            # 计算上家（前一个玩家）
            prev_index = (current_index - 1) % len(player_order)
            prev_player = player_order[prev_index]
            
            # 如果上家是自己，找前前一个
            if prev_player == self.name and len(player_order) > 1:
                prev_index = (prev_index - 1) % len(player_order)
                prev_player = player_order[prev_index]
        else:
            # 使用旧逻辑
            prev_players = [n for n in self.opinions.keys() if self.opponent_stats.get(n)]
            prev_player = prev_players[-1] if prev_players else None
        
        fake_prob = 0.0
        if prev_player and prev_player in self.opponent_stats:
            stats = self.opponent_stats[prev_player]
            fake_prob = stats['fake_plays'] / max(1, stats['total_plays'])
            # print(f"上家: {prev_player}, 出假牌概率: {fake_prob:.4f}")
        else:
            print(f"警告：无法找到上家或上家({prev_player})无统计数据")

        # 决定是否质疑
        if fake_prob > 0.5:
            was_challenged = True
            # print(f"决策：上家出假牌概率高({fake_prob:.2f}>0.5)，质疑")
        else:
            was_challenged = random.random() < exploration_rate
            # if was_challenged:
            #     print(f"决策：探索性质疑，探索率{exploration_rate:.2f}")
            # else:
            #     print(f"决策：不质疑，上家出假牌概率{fake_prob:.2f}")

        result = {
            "was_challenged": was_challenged,
            "challenge_reason": f"类人策略：基于上家({prev_player})出假牌概率{fake_prob:.2f}决策，探索率{exploration_rate:.2f}",
        }
        return result, f"类人策略：{'质疑' if was_challenged else '不质疑'}"

    def reflect(
        self,
        alive_players: List[str],
        round_base_info: str,
        round_action_info: str,
        round_result: str,
    ) -> None:
        """
        更新所有对手的统计数据
        以小局为单位，统计整个小局中的事件
        """
        # 更新每个对手的出牌统计：计数该对手在小局中出牌的次数
        for opp in self.opponent_stats.keys():
            play_count = round_action_info.count(f"轮到{opp}出牌")
            self.opponent_stats[opp]['total_plays'] += play_count


        # 更新每个对手的质疑统计：计数该对手可质疑的轮数和实际质疑的次数
        for opp in self.opponent_stats.keys():
            not_challenge_count = round_action_info.count(f"你选择不质疑{opp}")
            challenge_count = round_action_info.count(f"{opp}选择质疑你")
            self.opponent_stats[opp]['total_challenges'] += not_challenge_count + challenge_count
            self.opponent_stats[opp]['challenges_made'] += challenge_count


        # 更新假牌统计：如果质疑成功，被质疑者出了假牌
        # 假设一个小局中只有一个质疑成功的情况
        if "质疑成功" in round_action_info:
            lines = round_action_info.split('\n')
            last_player = None
            for line in reversed(lines):
                if "轮到" in line and "出牌" in line:
                    parts = line.split("轮到")[1].split("出牌")[0].strip()
                    last_player = parts
                    break
            if last_player and last_player in self.opponent_stats:
                self.opponent_stats[last_player]['fake_plays'] += 1

            