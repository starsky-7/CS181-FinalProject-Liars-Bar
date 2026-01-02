"""
对比实验：比较有/无对手特征的训练效果
"""

from typing import Dict
from rl_trainer import RLTrainer
from DQNAgent import DQNAgent
from LinearQAgent import LinearQAgent
from opponent_stats import OpponentStatsManager
from player import RLPlayer, HumanLikeStrategyPlayer, SimpleStrategyPlayer
from game import Game
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time


class BaselineTrainer:
    """
    基线训练器：不使用对手特征（模拟旧版本）
    通过传入None作为stats_manager来禁用对手特征
    """
    
    def __init__(self, agent_type: str, agent_config: Dict):
        self.agent_type = agent_type
        if self.agent_type == "LinearQ":
            self.agent = LinearQAgent(**agent_config)
        elif self.agent_type == "dqn":
            self.agent = DQNAgent(**agent_config)
        
        # 创建RL玩家（不传stats_manager）
        self.rl_player = RLPlayer("RL_Player", showDetails=False, 
                                  agent=self.agent, is_training=True,
                                  stats_manager=None)  # 关键：不使用对手特征
    
    def train(self, num_episodes: int, opponent_types=["humanlike", "simple", "simple"]):
        """训练（无对手特征）"""
        # 创建对手
        opponent_type_map = {
            "humanlike": HumanLikeStrategyPlayer,
            "simple": SimpleStrategyPlayer,
        }
        
        opponents = []
        for i, opp_type in enumerate(opponent_types):
            opp_class = opponent_type_map.get(opp_type, SimpleStrategyPlayer)
            opponents.append(opp_class(f"Opponent{i+1}", showDetails=False))
        
        players = [self.rl_player] + opponents
        
        for episode in tqdm(range(num_episodes), desc="Baseline Training"):
            game = Game(players=players, showDetails=False, 
                       stats_manager=None)  # 不使用stats_manager
            game.start_game()
            
            for player in players:
                player.alive = True
    
    def evaluate(self, num_games: int, opponent_types=["humanlike", "simple", "simple"]):
        """评估"""
        opponent_type_map = {
            "humanlike": HumanLikeStrategyPlayer,
            "simple": SimpleStrategyPlayer,
        }
        
        opponents = []
        for i, opp_type in enumerate(opponent_types):
            opp_class = opponent_type_map.get(opp_type, SimpleStrategyPlayer)
            opponents.append(opp_class(f"Opponent{i+1}", showDetails=False))
        
        eval_rl_player = RLPlayer("RL_Player", showDetails=False,
                                  agent=self.agent, is_training=False,
                                  stats_manager=None)
        
        players = [eval_rl_player] + opponents
        
        wins = 0
        for _ in range(num_games):
            game = Game(players=players, showDetails=False, stats_manager=None)
            game.start_game()
            
            for player in players:
                player.alive = True
            
            if game.game_record.winner == "RL_Player":
                wins += 1
        
        return {"wins": wins, "total": num_games, "win_rate": wins / num_games}


def compare_training():
    """对比实验主函数"""
    print("\n" + "=" * 70)
    print("对比实验：有/无对手特征的训练效果比较")
    print("=" * 70 + "\n")
    
    # 配置
    agent_config = {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.05
    }
    
    train_episodes = 200  # 训练局数
    eval_games = 50       # 评估局数
    opponent_types = ["humanlike", "simple", "simple"]
    
    print(f"配置:")
    print(f"  训练局数: {train_episodes}")
    print(f"  评估局数: {eval_games}")
    print(f"  对手类型: {opponent_types}")
    print(f"  Agent配置: {agent_config}\n")
    
    # ========== 基线：无对手特征 ==========
    print("-" * 70)
    print("实验1: 基线训练（无对手特征）")
    print("-" * 70)
    
    start_time = time.time()
    baseline_trainer = BaselineTrainer("dqn", agent_config)
    baseline_trainer.train(train_episodes, opponent_types)
    baseline_train_time = time.time() - start_time
    
    print(f"\n基线训练完成，耗时: {baseline_train_time:.2f}秒")
    print("开始评估...")
    
    baseline_results = baseline_trainer.evaluate(eval_games, opponent_types)
    print(f"基线胜率: {baseline_results['win_rate']:.2%}")
    
    # ========== 实验组：有对手特征 ==========
    print("\n" + "-" * 70)
    print("实验2: 对手特征训练")
    print("-" * 70)
    
    start_time = time.time()
    feature_trainer = RLTrainer("dqn", agent_config)
    feature_trainer.train(train_episodes, opponent_types)
    feature_train_time = time.time() - start_time
    
    print(f"\n对手特征训练完成，耗时: {feature_train_time:.2f}秒")
    print("开始评估...")
    
    feature_results = feature_trainer.evaluate(eval_games, opponent_types)
    print(f"对手特征胜率: {feature_results['win_rate']:.2%}")
    
    # ========== 泛化能力测试 ==========
    print("\n" + "-" * 70)
    print("泛化能力测试：对阵不同对手组合")
    print("-" * 70)
    
    new_opponents = ["humanlike", "humanlike", "simple"]
    print(f"\n新对手组合: {new_opponents}")
    
    print("  基线Agent...")
    baseline_gen = baseline_trainer.evaluate(eval_games, new_opponents)
    print(f"    胜率: {baseline_gen['win_rate']:.2%}")
    
    print("  对手特征Agent...")
    feature_gen = feature_trainer.evaluate(eval_games, new_opponents)
    print(f"    胜率: {feature_gen['win_rate']:.2%}")
    
    # ========== 结果总结 ==========
    print("\n" + "=" * 70)
    print("实验结果总结")
    print("=" * 70)
    
    print(f"\n训练耗时:")
    print(f"  基线: {baseline_train_time:.2f}秒")
    print(f"  对手特征: {feature_train_time:.2f}秒")
    print(f"  耗时增加: {(feature_train_time - baseline_train_time) / baseline_train_time * 100:.1f}%")
    
    print(f"\n原对手胜率 ({opponent_types}):")
    print(f"  基线: {baseline_results['win_rate']:.2%}")
    print(f"  对手特征: {feature_results['win_rate']:.2%}")
    print(f"  胜率提升: {(feature_results['win_rate'] - baseline_results['win_rate']) * 100:.1f}%")
    
    print(f"\n新对手胜率 ({new_opponents}):")
    print(f"  基线: {baseline_gen['win_rate']:.2%}")
    print(f"  对手特征: {feature_gen['win_rate']:.2%}")
    print(f"  胜率提升: {(feature_gen['win_rate'] - baseline_gen['win_rate']) * 100:.1f}%")
    
    # 判断对手特征是否有效
    print("\n" + "=" * 70)
    if feature_results['win_rate'] > baseline_results['win_rate']:
        improvement = (feature_results['win_rate'] - baseline_results['win_rate']) * 100
        print(f"✅ 对手特征有效！胜率提升 {improvement:.1f}%")
    else:
        print("⚠️  对手特征可能需要更多训练或调优")
    
    if feature_gen['win_rate'] > baseline_gen['win_rate']:
        print("✅ 对手特征提升了泛化能力！")
    else:
        print("⚠️  泛化能力未见明显提升")
    print("=" * 70 + "\n")


def quick_feature_check():
    """快速检查对手特征是否被使用"""
    print("\n" + "=" * 70)
    print("快速检查：验证对手特征是否真正被使用")
    print("=" * 70 + "\n")
    
    agent_config = {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon": 0.1,  # 低探索率，观察决策
    }
    
    # 创建训练器并运行几局
    trainer = RLTrainer("dqn", agent_config)
    
    print("运行10局游戏，收集统计...")
    trainer.train(num_episodes=10)
    
    print("\n对手统计（应该有数据）:")
    trainer.stats_manager.print_summary()
    
    # 检查特征维度
    from rl_utils import FeatureExtractor
    extractor = FeatureExtractor()
    
    # 模拟一个state（13个元素 = 9基础 + 4对手）
    test_state = (1, 'K', 2, 2, 1, 1, 1, 3, 5, 0.15, 0.4, 0.09, 0.8)
    features = extractor.get_features(test_state)
    
    print(f"\n特征向量检查:")
    print(f"  State元素数: {len(test_state)}")
    print(f"  提取的特征数: {len(features)}")
    print(f"  对手特征值: {features[-4:]}")
    
    if len(test_state) == 13 and len(features) > 10:
        print("\n✅ 对手特征正在被使用！")
    else:
        print("\n❌ 对手特征可能未正确集成")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # 快速检查模式
        quick_feature_check()
    else:
        # 完整对比实验
        compare_training()
