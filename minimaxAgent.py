"""
从与simpleAgent对战的测试结果来看，minimax表现并不是很理想，我定义了比较复杂的评估函数和行为(四种行为在迭代时已经麻了)，才使得minimax不至于被simpleAgent完胜。。。。（simpleagent恐怖如斯）
当然理论上来说，只要对minimaxagent提供足够多样的评估函数以及行为定义，包括无脑加深深度都可以优化其表现，但越来越冗余的定义和计算也说明了，对于骗子酒吧来说miniMax算法
完全算不上最优解法，毕竟它的核心思想是穷举搜索，而在复杂的博弈游戏中，状态空间和行为空间都非常庞大，穷举搜索的计算量会呈指数级增长，导致计算资源和时间消耗过大。
"""

'''
至于RLAgent，MinimaxAgent只赢第一把:)
'''

from typing import List
import re

class GameState:
    def __init__(self, hand: List[str], opponent_hand_size: int, target_card: str, is_my_turn: bool, depth: int):
        self.hand = hand
        self.opponent_hand_size = opponent_hand_size
        self.target_card = target_card
        self.is_my_turn = is_my_turn
        self.depth = depth

    def is_terminal(self) -> bool:
        """判断是否为终止状态"""
        return self.depth == 0 or len(self.hand) == 0 or self.opponent_hand_size == 0


    def evaluate(self) -> int:
        real_card_count = len([c for c in self.hand if c == self.target_card or c == "Joker"])
        bluff_risk_penalty = 5 if real_card_count == 0 else 0
        hand_size_bonus = len(self.hand)
        is_my_turn_bonus = 5 if self.is_my_turn else 0

        score = (
            real_card_count * 10
            - self.opponent_hand_size * 6
            + is_my_turn_bonus
            - bluff_risk_penalty
            + hand_size_bonus * 1
        )
        return score



    def generate_actions(self) -> List[List[str]]:
        """生成当前状态下可能的出牌动作"""
        actions = [[]]  # 可以选择不出牌
        real_cards = [c for c in self.hand if c == self.target_card or c == "Joker"]
        bluff_cards = [c for c in self.hand if c not in real_cards]

        if real_cards:
            actions.append([real_cards[0]])
            if len(real_cards) >= 2:
                actions.append(real_cards[:2])
        if bluff_cards:
            actions.append([bluff_cards[0]])
        return actions

    def simulate(self, action: List[str]) -> 'GameState':
        """模拟出牌后的新状态"""
        new_hand = self.hand.copy()
        for card in action:
            if card in new_hand:
                new_hand.remove(card)
        # 简单模拟对手也会出一张牌
        new_opponent_size = self.opponent_hand_size - (0 if self.is_my_turn else 1)
        return GameState(
            hand=new_hand,
            opponent_hand_size=new_opponent_size,
            target_card=self.target_card,
            is_my_turn=not self.is_my_turn,
            depth=self.depth - 1
        )


def minimax_decision(state: GameState) -> List[str]:
    """从当前状态出发，使用 Minimax 算法计算最优出牌动作"""
    best_score = float('-inf')
    best_action = []
    for action in state.generate_actions():
        new_state = state.simulate(action)
        score = min_value(new_state)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action

def max_value(state: GameState) -> int:
    if state.is_terminal():
        return state.evaluate()
    v = float('-inf')
    for action in state.generate_actions():
        v = max(v, min_value(state.simulate(action)))
        '''这里将min()替换为expecti_value()就可以实现Expectimax算法，不过对战表现区别也不大。。。'''
        # v = max(v, expecti_value(state.simulate(action)))
    return v


def min_value(state: GameState) -> int:
    if state.is_terminal():
        return state.evaluate()
    v = float('inf')
    for action in state.generate_actions():
        v = min(v, max_value(state.simulate(action)))
    return v

def expecti_value(state: GameState) -> float:
    if state.is_terminal():
        return state.evaluate()
    actions = state.generate_actions()
    total = 0
    for action in actions:
        new_state = state.simulate(action)
        total += max_value(new_state)
    return total / len(actions) if actions else 0



def extract_opponent_hand_size(info: str, opponent_name: str) -> int:
    """
    从 action_info 中提取指定玩家的手牌数（剩余手牌X张）
    """
    pattern = fr"{opponent_name}.*?剩余手牌(\d+)张"
    match = re.search(pattern, info)
    if match:
        return int(match.group(1))
    return 5  # 默认值

def infer_opponent_name(round_base_info: str, self_name: str) -> str:
    """
    从 round_base_info 中识别对手玩家名称
    """
    match = re.search(r"本轮玩家：(.*?)[，,]从玩家", round_base_info)
    if match:
        players = match.group(1).replace(" ", "").split("、")
        for name in players:
            if name != self_name:
                return name
    return "Opponent"