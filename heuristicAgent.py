from typing import List, Tuple, Dict

class HeuristicAgent:
    def __init__(self, name: str, hand: List[str], target_card: str):
        self.name = name
        self.hand = hand
        self.target_card = target_card
        self.bullet_pos = 0
        self.chamber_pos = 0

    def update_state(self, hand: List[str], target_card: str, bullet_pos: int, chamber_pos: int):
        self.hand = hand
        self.target_card = target_card
        self.bullet_pos = bullet_pos
        self.chamber_pos = chamber_pos

    def choose_cards_to_play(self) -> Tuple[Dict, str]:
        candidate_actions = self._generate_actions()
        best_score = float("-inf")
        best_action = []

        for action in candidate_actions:
            score = self._evaluate_action(action)
            if score > best_score:
                best_score = score
                best_action = action

        for card in best_action:
            if card in self.hand:
                self.hand.remove(card)

        result = {
            "played_cards": best_action,
            "behavior": "子弹规避 + 启发式出牌",
            "play_reason": f"评分={best_score:.2f}，基于目标牌、Joker、风险规避"
        }
        return result, f"HeuristicAgent出牌: {best_action}"

    # def decide_challenge(self, opponent_action_info: str) -> Tuple[bool, str]:
    #     # 可扩展，目前默认质疑
    #     return True, "默认质疑"

    def _generate_actions(self) -> List[List[str]]:
        real_cards = [c for c in self.hand if c == self.target_card or c == "Joker"]
        bluff_cards = [c for c in self.hand if c != self.target_card and c != "Joker"]
        actions = [[]]

        if real_cards:
            actions.append([real_cards[0]])
            if len(real_cards) >= 2:
                actions.append(real_cards[:2])
        if bluff_cards:
            actions.append([bluff_cards[0]])

        return actions

    def _evaluate_action(self, action: List[str]) -> float:
        if not action:
            return 0.0  # 不出牌，风险为零

        score = 0.0
        for card in action:
            if card == self.target_card:
                score += 10
            elif card == "Joker":
                score += 8
            else:
                score -= 4  # 出非目标牌惩罚

        # 子弹风险惩罚（比如在子弹位置前一两格）
        risk_distance = (self.bullet_pos - self.chamber_pos + 6) % 6
        if risk_distance in [0, 1]:  # 当前或下一个就是子弹
            score -= 10

        return score
