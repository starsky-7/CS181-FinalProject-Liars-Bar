"""
最终评估脚本：全面测试RL Agent的性能
包括：
1. 基础性能测试
2. 泛化能力测试（不同对手组合）
3. 鲁棒性测试（位置打乱）
4. 对手特征有效性验证
"""

from advanced_trainer import AdvancedRLTrainer
from rl_trainer import RLTrainer
from player import RLPlayer, HumanLikeStrategyPlayer, SimpleStrategyPlayer, SmarterStrategyPlayer
from game import Game
from opponent_stats import OpponentStatsManager
import numpy as np
import random
from tqdm import tqdm
import json
from datetime import datetime


class FinalEvaluator:
    """最终评估器"""
    
    def __init__(self, model_path: str, stats_path: str = None):
        """
        Args:
            model_path: 训练好的模型路径
            stats_path: 对手统计数据路径（可选）
        """
        self.model_path = model_path
        self.stats_path = stats_path
        
        # 加载模型
        from DQNAgent import DQNAgent
        self.agent = DQNAgent()
        self.agent.load_model(model_path)
        
        # 加载对手统计
        self.stats_manager = OpponentStatsManager(update_mode='eval')
        if stats_path:
            self.stats_manager.load_stats(stats_path)
        
        self.results = {}
    
    def test_basic_performance(self, num_games: int = 100) -> dict:
        """基础性能测试：对阵训练时的对手"""
        print("\n" + "="*70)
        print("测试1: 基础性能（对阵训练对手）")
        print("="*70)
        
        opponent_types = ["humanlike", "simple", "simple"]
        results = self._run_games(num_games, opponent_types, shuffle=False)
        
        print(f"\n对手组合: {opponent_types}")
        print(f"胜率: {results['win_rate']:.2%} ({results['wins']}/{results['total']})")
        print(f"平均轮数: {results['avg_rounds']:.1f}")
        
        self.results['basic_performance'] = results
        return results
    
    def test_generalization(self, num_games: int = 50) -> dict:
        """泛化能力测试：对阵不同对手组合"""
        print("\n" + "="*70)
        print("测试2: 泛化能力（不同对手组合）")
        print("="*70)
        
        test_cases = [
            ["simple", "simple", "simple"],       # 全弱
            ["humanlike", "humanlike", "humanlike"],  # 全强
            ["smarter", "smarter", "simple"],     # 混合1
            ["humanlike", "smarter", "smarter"],  # 混合2
        ]
        
        gen_results = {}
        for opponents in test_cases:
            results = self._run_games(num_games, opponents, shuffle=False)
            gen_results[str(opponents)] = results
            
            print(f"\n对手: {opponents}")
            print(f"  胜率: {results['win_rate']:.2%}")
            print(f"  平均轮数: {results['avg_rounds']:.1f}")
        
        # 计算平均泛化胜率
        avg_win_rate = np.mean([r['win_rate'] for r in gen_results.values()])
        print(f"\n平均泛化胜率: {avg_win_rate:.2%}")
        
        self.results['generalization'] = gen_results
        return gen_results
    
    def test_position_robustness(self, num_games: int = 100) -> dict:
        """鲁棒性测试：随机打乱位置"""
        print("\n" + "="*70)
        print("测试3: 位置鲁棒性（随机打乱玩家顺序）")
        print("="*70)
        
        opponent_types = ["humanlike", "simple", "simple"]
        
        # 固定位置
        print("\n固定位置测试...")
        fixed_results = self._run_games(num_games, opponent_types, shuffle=False)
        print(f"胜率: {fixed_results['win_rate']:.2%}")
        
        # 随机位置
        print("\n随机位置测试...")
        shuffled_results = self._run_games(num_games, opponent_types, shuffle=True)
        print(f"胜率: {shuffled_results['win_rate']:.2%}")
        
        # 对比
        diff = shuffled_results['win_rate'] - fixed_results['win_rate']
        print(f"\n胜率差异: {diff*100:+.1f}%")
        
        if abs(diff) < 0.05:
            print("✅ 位置鲁棒性良好（差异<5%）")
        else:
            print("⚠️  位置鲁棒性一般（差异≥5%）")
        
        self.results['position_robustness'] = {
            'fixed': fixed_results,
            'shuffled': shuffled_results,
            'difference': diff
        }
        
        return self.results['position_robustness']
    
    def test_opponent_feature_impact(self, num_games: int = 50) -> dict:
        """对手特征影响测试：对比有/无对手特征"""
        print("\n" + "="*70)
        print("测试4: 对手特征有效性")
        print("="*70)
        
        opponent_types = ["humanlike", "simple", "simple"]
        
        # 有对手特征
        print("\n使用对手特征...")
        with_features = self._run_games(num_games, opponent_types, 
                                       use_features=True, shuffle=False)
        print(f"胜率: {with_features['win_rate']:.2%}")
        
        # 无对手特征
        print("\n不使用对手特征...")
        without_features = self._run_games(num_games, opponent_types, 
                                          use_features=False, shuffle=False)
        print(f"胜率: {without_features['win_rate']:.2%}")
        
        # 对比
        improvement = with_features['win_rate'] - without_features['win_rate']
        print(f"\n对手特征带来的提升: {improvement*100:+.1f}%")
        
        if improvement > 0.03:
            print("✅ 对手特征显著有效（提升>3%）")
        elif improvement > 0:
            print("✓ 对手特征有一定效果")
        else:
            print("⚠️  对手特征效果不明显")
        
        self.results['feature_impact'] = {
            'with_features': with_features,
            'without_features': without_features,
            'improvement': improvement
        }
        
        return self.results['feature_impact']
    
    def _run_games(self, num_games: int, opponent_types: list, 
                   shuffle: bool = False, use_features: bool = True) -> dict:
        """运行多局游戏并收集统计"""
        # 创建对手
        opponent_map = {
            "humanlike": HumanLikeStrategyPlayer,
            "simple": SimpleStrategyPlayer,
            "smarter": SmarterStrategyPlayer,
        }
        
        opponents = [opponent_map[t](f"Opponent{i+1}", showDetails=False) 
                    for i, t in enumerate(opponent_types)]
        
        # 创建RL玩家
        stats = self.stats_manager if use_features else None
        rl_player = RLPlayer("RL_Player", showDetails=False, 
                            agent=self.agent, is_training=False,
                            stats_manager=stats)
        
        wins = 0
        total_rounds = []
        
        for _ in tqdm(range(num_games), desc="Running games", leave=False):
            players = [rl_player] + opponents
            
            # 是否打乱位置
            if shuffle:
                random.shuffle(players)
            
            # 运行游戏
            game = Game(players=players, showDetails=False, stats_manager=stats)
            game.start_game()
            
            # 统计结果
            if game.game_record.winner == "RL_Player":
                wins += 1
            total_rounds.append(game.round_count)
            
            # 重置
            for p in players:
                p.alive = True
        
        return {
            'wins': wins,
            'total': num_games,
            'win_rate': wins / num_games,
            'avg_rounds': np.mean(total_rounds),
        }
    
    def generate_report(self, output_path: str = None):
        """生成评估报告"""
        print("\n" + "="*70)
        print("最终评估报告")
        print("="*70)
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': self.model_path,
            'stats_path': self.stats_path,
            'results': self.results,
        }
        
        # 打印摘要
        print(f"\n模型: {self.model_path}")
        
        if 'basic_performance' in self.results:
            print(f"\n基础胜率: {self.results['basic_performance']['win_rate']:.2%}")
        
        if 'generalization' in self.results:
            avg = np.mean([r['win_rate'] for r in self.results['generalization'].values()])
            print(f"平均泛化胜率: {avg:.2%}")
        
        if 'position_robustness' in self.results:
            diff = self.results['position_robustness']['difference']
            print(f"位置鲁棒性差异: {diff*100:+.1f}%")
        
        if 'feature_impact' in self.results:
            imp = self.results['feature_impact']['improvement']
            print(f"对手特征提升: {imp*100:+.1f}%")
        
        # 保存报告
        if output_path is None:
            output_path = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n报告已保存: {output_path}")
        print("="*70 + "\n")
        
        return report
    
    def run_full_evaluation(self):
        """运行完整评估流程"""
        print("\n" + "="*70)
        print("开始完整评估流程")
        print("="*70)
        
        self.test_basic_performance(num_games=100)
        self.test_generalization(num_games=50)
        self.test_position_robustness(num_games=100)
        self.test_opponent_feature_impact(num_games=50)
        
        self.generate_report()


def quick_eval():
    """快速评估（小样本）"""
    print("快速评估模式")
    print("注意：需要先训练模型！")
    print("\n使用示例:")
    print("  python -c \"from train_configs import medium_training; medium_training()\"")
    print("  然后运行此脚本\n")


def evaluate_latest_model():
    """评估最新模型"""
    import os
    import glob
    
    # 查找最新的实验
    exp_dirs = glob.glob("experiments/*/models/agent_final.pkl")
    
    if not exp_dirs:
        print("❌ 未找到训练好的模型！")
        print("请先运行训练：python train_configs.py medium")
        return
    
    # 使用最新的模型
    latest_model = max(exp_dirs, key=os.path.getctime)
    latest_stats = latest_model.replace("agent_final.pkl", "opponent_stats_final.pkl")
    
    print(f"找到模型: {latest_model}")
    
    if not os.path.exists(latest_stats):
        print(f"⚠️  未找到对应的统计文件，将不使用对手特征")
        latest_stats = None
    
    # 创建评估器
    evaluator = FinalEvaluator(latest_model, latest_stats)
    
    # 运行完整评估
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 指定模型路径
        model_path = sys.argv[1]
        stats_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        evaluator = FinalEvaluator(model_path, stats_path)
        evaluator.run_full_evaluation()
    else:
        # 自动评估最新模型
        evaluate_latest_model()
