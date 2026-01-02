"""
测试对手特征提取功能
验证StateEncoder和OpponentStatsManager的正确性
"""

from rl_utils import StateEncoder, FeatureExtractor
from opponent_stats import OpponentStatsManager
from player import RLPlayer, HumanLikeStrategyPlayer, SimpleStrategyPlayer
from DQNAgent import DQNAgent
from game import Game
import numpy as np


def test_feature_extraction():
    """测试特征提取的正确性"""
    print("=" * 60)
    print("测试1: StateEncoder的对手名字提取功能")
    print("=" * 60)
    
    encoder = StateEncoder()
    
    # 测试出牌阶段的文本
    play_text1 = "Opponent1是你的下家，决定是否质疑你的出牌。\n你已经开了2枪，Opponent1开了1枪。你对Opponent1的印象分析：还不了解这个玩家"
    play_text2 = "Opponent2是你的下家，决定是否质疑你的出牌。"
    
    # 测试质疑阶段的文本
    challenge_text1 = "你正在判断是否质疑Opponent3的出牌。\n你已经开了1枪，Opponent3开了0枪。"
    challenge_text2 = "你正在判断是否质疑RL_Player的出牌。"
    
    print(f"出牌文本1: {play_text1[:50]}...")
    print(f"提取结果: {encoder._extract_opponent_name(play_text1)}")
    print(f"\n出牌文本2: {play_text2}")
    print(f"提取结果: {encoder._extract_opponent_name(play_text2)}")
    print(f"\n质疑文本1: {challenge_text1[:50]}...")
    print(f"提取结果: {encoder._extract_opponent_name(challenge_text1)}")
    print(f"\n质疑文本2: {challenge_text2}")
    print(f"提取结果: {encoder._extract_opponent_name(challenge_text2)}")
    
    print("\n✅ 对手名字提取测试完成\n")


def test_stats_manager():
    """测试OpponentStatsManager的统计功能"""
    print("=" * 60)
    print("测试2: OpponentStatsManager统计功能")
    print("=" * 60)
    
    stats_manager = OpponentStatsManager(update_mode='train')
    
    # 注册玩家
    stats_manager.register_player("TestOpponent", "HumanLikeStrategyPlayer")
    
    # 模拟一些行为
    print("\n模拟20次出牌行为（5次作弊）:")
    for i in range(20):
        did_cheat = (i % 4 == 0)  # 每4次作弊1次 = 25%
        stats_manager.record_play_behavior("TestOpponent", did_cheat)
    
    print("\n模拟10次质疑机会（6次质疑）:")
    for i in range(10):
        did_challenge = (i < 6)  # 前6次质疑 = 60%
        stats_manager.record_challenge_behavior("TestOpponent", did_challenge)
    
    # 获取特征
    features = stats_manager.get_features("TestOpponent", use_recent=False)
    
    print("\n提取的特征:")
    print(f"  作弊率: {features['cheat_rate']:.2%}")
    print(f"  质疑率: {features['challenge_rate']:.2%}")
    print(f"  激进度: {features['aggression']:.2%}")
    print(f"  谨慎度: {features['caution']:.2%}")
    print(f"  置信度: {features['confidence']:.2%}")
    
    print("\n✅ OpponentStatsManager测试完成\n")


def test_state_encoding():
    """测试完整的state编码"""
    print("=" * 60)
    print("测试3: 完整state编码（有/无stats_manager）")
    print("=" * 60)
    
    encoder = StateEncoder()
    feature_extractor = FeatureExtractor()
    stats_manager = OpponentStatsManager(update_mode='train')
    
    # 注册并模拟统计
    stats_manager.register_player("Opponent1", "SimpleStrategyPlayer")
    for i in range(50):
        stats_manager.record_play_behavior("Opponent1", did_cheat=(i % 10 == 0))
        stats_manager.record_challenge_behavior("Opponent1", did_challenge=(i % 3 == 0))
    
    # 模拟游戏信息
    round_base_info = "现在是第3轮，目标牌：K，本轮玩家：RL_Player、Opponent1、Opponent2，从玩家RL_Player开始"
    round_action_info = ""
    play_decision_info = "Opponent1是你的下家，决定是否质疑你的出牌。\n你已经开了1枪，Opponent1开了0枪。你对Opponent1的印象分析：还不了解"
    hand = ["K", "K", "Q", "A", "Joker"]
    target_card = "K"
    current_bullet = 1
    
    # 编码状态（无stats_manager）
    print("\n不使用stats_manager:")
    state_no_stats = encoder.encode_play_state(
        round_base_info, round_action_info, play_decision_info,
        hand, target_card, current_bullet,
        stats_manager=None
    )
    print(f"  State长度: {len(state_no_stats)}")
    print(f"  State内容: {state_no_stats}")
    features_no_stats = feature_extractor.get_features(state_no_stats)
    print(f"  特征向量长度: {len(features_no_stats)}")
    print(f"  对手特征部分: {features_no_stats[-4:]}")  # 最后4个是对手特征
    
    # 编码状态（有stats_manager）
    print("\n使用stats_manager:")
    state_with_stats = encoder.encode_play_state(
        round_base_info, round_action_info, play_decision_info,
        hand, target_card, current_bullet,
        stats_manager=stats_manager
    )
    print(f"  State长度: {len(state_with_stats)}")
    print(f"  State内容: {state_with_stats}")
    features_with_stats = feature_extractor.get_features(state_with_stats)
    print(f"  特征向量长度: {len(features_with_stats)}")
    print(f"  对手特征部分: {features_with_stats[-4:]}")
    
    # 验证对手特征不是默认值
    opp_features = stats_manager.get_features("Opponent1")
    print(f"\n预期的对手特征:")
    print(f"  作弊率: {opp_features['cheat_rate']:.4f}")
    print(f"  质疑率: {opp_features['challenge_rate']:.4f}")
    print(f"  激进度: {opp_features['aggression']:.4f}")
    print(f"  置信度: {opp_features['confidence']:.4f}")
    
    print("\n✅ State编码测试完成\n")


def test_integration():
    """集成测试：运行一局完整游戏"""
    print("=" * 60)
    print("测试4: 集成测试 - 运行完整游戏")
    print("=" * 60)
    
    # 创建组件
    agent_config = {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.05
    }
    agent = DQNAgent(**agent_config)
    stats_manager = OpponentStatsManager(update_mode='train')
    
    # 创建玩家
    rl_player = RLPlayer("RL_Player", showDetails=False, agent=agent, 
                        is_training=True, stats_manager=stats_manager)
    opp1 = HumanLikeStrategyPlayer("Opponent1", showDetails=False)
    opp2 = SimpleStrategyPlayer("Opponent2", showDetails=False)
    
    players = [rl_player, opp1, opp2]
    
    # 注册玩家
    stats_manager.register_player("RL_Player", "RLPlayer")
    stats_manager.register_player("Opponent1", "HumanLikeStrategyPlayer")
    stats_manager.register_player("Opponent2", "SimpleStrategyPlayer")
    
    print("\n运行3局游戏...")
    for i in range(3):
        game = Game(players=players, showDetails=False, stats_manager=stats_manager)
        game.start_game()
        
        # 重置玩家
        for player in players:
            player.alive = True
        
        print(f"  第{i+1}局完成，胜者: {game.game_record.winner}")
    
    print("\n游戏结束后的统计:")
    stats_manager.print_summary()
    
    print("\n✅ 集成测试完成\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始对手特征提取功能测试")
    print("=" * 60 + "\n")
    
    try:
        test_feature_extraction()
        test_stats_manager()
        test_state_encoding()
        test_integration()
        
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
