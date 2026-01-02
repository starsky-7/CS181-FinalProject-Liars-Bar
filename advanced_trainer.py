"""
增强型训练器：支持训练监控、可视化、断点续训
"""

from typing import List, Dict, Optional
from game import Game
from player import RLPlayer, HumanLikeStrategyPlayer, SmarterStrategyPlayer, SimpleStrategyPlayer
from DQNAgent import DQNAgent
from LinearQAgent import LinearQAgent
from opponent_stats import OpponentStatsManager
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime


class AdvancedRLTrainer:
    """
    增强型强化学习训练器
    新增功能：
    - 训练过程监控（胜率、epsilon、损失等）
    - 自动可视化
    - 定期评估
    - 断点续训
    - 训练日志
    """
    
    def __init__(self, agent_type: str, agent_config: Dict, experiment_name: Optional[str] = None):
        self.agent_type = agent_type
        self.agent_config = agent_config
        
        if self.agent_type == "LinearQ":
            self.agent = LinearQAgent(**agent_config)
        elif self.agent_type == "dqn":
            self.agent = DQNAgent(**agent_config)
        
        # 创建对手统计管理器
        self.stats_manager = OpponentStatsManager(update_mode='train')
        
        # 创建RL玩家
        self.rl_player = RLPlayer("RL_Player", showDetails=False, agent=self.agent, 
                                  is_training=True, stats_manager=self.stats_manager)
        
        # 实验名称
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建实验目录
        self.exp_dir = f"experiments/{self.experiment_name}"
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(f"{self.exp_dir}/models", exist_ok=True)
        os.makedirs(f"{self.exp_dir}/plots", exist_ok=True)
        
        # 训练指标记录
        self.metrics = {
            'episodes': [],
            'train_wins': [],
            'eval_wins': [],
            'epsilon': [],
            'opponent_cheat_rates': [],
            'opponent_challenge_rates': []
        }
        
        # 保存配置
        self._save_config(agent_config)
    
    def _save_config(self, agent_config: Dict):
        """保存实验配置"""
        config = {
            'agent_type': self.agent_type,
            'agent_config': agent_config,
            'experiment_name': self.experiment_name,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(f"{self.exp_dir}/config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _create_opponents(self, opponent_types: List[str]) -> List:
        """根据类型创建对手"""
        opponent_type_map = {
            "humanlike": HumanLikeStrategyPlayer,
            "simple": SimpleStrategyPlayer,
            "smarter": SmarterStrategyPlayer
        }
        
        opponents = []
        for i, opp_type in enumerate(opponent_types):
            opp_class = opponent_type_map.get(opp_type, SimpleStrategyPlayer)
            opponents.append(opp_class(f"Opponent{i+1}", showDetails=False))
        
        return opponents
    
    def _register_players(self, opponent_types: List[str]):
        """注册玩家到统计管理器"""
        opponent_type_map = {
            "humanlike": "HumanLikeStrategyPlayer",
            "simple": "SimpleStrategyPlayer",
            "smarter": "SmarterStrategyPlayer"
        }
        
        if "RL_Player" not in self.stats_manager.player_stats:
            self.stats_manager.register_player("RL_Player", "RLPlayer")
        
        for i, opp_type in enumerate(opponent_types):
            player_name = f"Opponent{i+1}"
            if player_name not in self.stats_manager.player_stats:
                class_name = opponent_type_map.get(opp_type, "SimpleStrategyPlayer")
                self.stats_manager.register_player(player_name, class_name)
    
    def train(self, 
              num_episodes: int, 
              opponent_types: List[str] = ["humanlike", "simple", "simple"],
              eval_interval: int = 100,
              eval_games: int = 20,
              save_interval: int = 1000,
              plot_interval: int = 100,
              shuffle_positions: bool = True) -> None:
        """
        训练RL代理（增强版）
        
        Args:
            num_episodes: 训练局数
            opponent_types: 对手类型列表
            eval_interval: 评估间隔（每N局评估一次）
            eval_games: 每次评估的游戏数
            save_interval: 保存间隔（每N局保存一次）
            plot_interval: 绘图间隔（每N局绘制一次）
            shuffle_positions: 是否每局随机打乱玩家顺序（提升位置鲁棒性）
        """
        opponents = self._create_opponents(opponent_types)
        self._register_players(opponent_types)
        players = [self.rl_player] + opponents
        
        print(f"\n{'='*70}")
        print(f"开始训练实验: {self.experiment_name}")
        print(f"{'='*70}")
        print(f"配置:")
        print(f"  总训练局数: {num_episodes}")
        print(f"  对手类型: {opponent_types}")
        print(f"  评估间隔: {eval_interval}局")
        print(f"  保存间隔: {save_interval}局")
        print(f"  随机打乱位置: {'是' if shuffle_positions else '否'}")
        print(f"  实验目录: {self.exp_dir}")
        print(f"{'='*70}\n")
        
        # 训练窗口内的胜率统计
        recent_wins = []
        window_size = 50
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            # 随机打乱玩家顺序（提升位置鲁棒性）
            if shuffle_positions:
                import random
                random.shuffle(players)
            
            # 创建游戏
            game = Game(players=players, showDetails=False, 
                       stats_manager=self.stats_manager)
            game.start_game()
            
            # 记录本局胜负
            won = (game.game_record.winner == "RL_Player")
            recent_wins.append(1 if won else 0)
            if len(recent_wins) > window_size:
                recent_wins.pop(0)
            
            # 重置玩家状态
            for player in players:
                player.alive = True
            
            # 定期评估
            if (episode + 1) % eval_interval == 0:
                eval_result = self._evaluate(eval_games, opponent_types)
                
                # 记录指标
                self.metrics['episodes'].append(episode + 1)
                self.metrics['train_wins'].append(np.mean(recent_wins))
                self.metrics['eval_wins'].append(eval_result['win_rate'])
                self.metrics['epsilon'].append(self.agent.epsilon)
                
                # 记录对手统计
                opp_stats = []
                opp_challenge = []
                for i in range(len(opponent_types)):
                    features = self.stats_manager.get_features(f"Opponent{i+1}")
                    opp_stats.append(features['cheat_rate'])
                    opp_challenge.append(features['challenge_rate'])
                
                self.metrics['opponent_cheat_rates'].append(np.mean(opp_stats))
                self.metrics['opponent_challenge_rates'].append(np.mean(opp_challenge))
                
                # 打印进度
                print(f"\n[Episode {episode+1}/{num_episodes}]")
                print(f"  训练胜率(最近{window_size}局): {np.mean(recent_wins):.2%}")
                print(f"  评估胜率: {eval_result['win_rate']:.2%}")
                print(f"  Epsilon: {self.agent.epsilon:.4f}")
                print(f"  对手平均作弊率: {np.mean(opp_stats):.2%}")
                print(f"  对手平均质疑率: {np.mean(opp_challenge):.2%}")
            
            # 定期绘图
            if (episode + 1) % plot_interval == 0 and len(self.metrics['episodes']) > 0:
                self._plot_metrics()
            
            # 定期保存
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode + 1)
        
        # 训练结束
        print(f"\n{'='*70}")
        print(f"训练完成！")
        print(f"{'='*70}\n")
        
        # 最终评估
        final_eval = self._evaluate(eval_games * 2, opponent_types)
        print(f"最终评估胜率: {final_eval['win_rate']:.2%}")
        
        # 保存最终模型和指标
        self._save_checkpoint(num_episodes, final=True)
        self._plot_metrics(final=True)
        self._save_metrics()
        
        # 打印对手统计摘要
        print("\n最终对手统计:")
        self.stats_manager.print_summary()
    
    def _evaluate(self, num_games: int, opponent_types: List[str]) -> Dict:
        """评估当前模型"""
        self.stats_manager.set_mode('eval')
        
        opponents = self._create_opponents(opponent_types)
        eval_rl_player = RLPlayer("RL_Player", showDetails=False, 
                                  agent=self.agent, is_training=False,
                                  stats_manager=self.stats_manager)
        players = [eval_rl_player] + opponents
        
        wins = 0
        for _ in range(num_games):
            game = Game(players=players, showDetails=False,
                       stats_manager=self.stats_manager)
            game.start_game()
            
            for player in players:
                player.alive = True
            
            if game.game_record.winner == "RL_Player":
                wins += 1
        
        self.stats_manager.set_mode('train')
        return {"wins": wins, "total": num_games, "win_rate": wins / num_games}
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """保存检查点"""
        suffix = "final" if final else f"ep{episode}"
        
        # 保存模型
        model_path = f"{self.exp_dir}/models/agent_{suffix}.pkl"
        self.agent.save_model(model_path)
        
        # 保存对手统计
        stats_path = f"{self.exp_dir}/models/opponent_stats_{suffix}.pkl"
        self.stats_manager.save_stats(stats_path)
        
        if not final:
            print(f"  → 检查点已保存: episode {episode}")
    
    def _save_metrics(self):
        """保存训练指标"""
        metrics_path = f"{self.exp_dir}/metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def _plot_metrics(self, final: bool = False):
        """绘制训练曲线"""
        if len(self.metrics['episodes']) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Metrics - {self.experiment_name}', fontsize=16)
        
        episodes = self.metrics['episodes']
        
        # 1. 胜率曲线
        ax1 = axes[0, 0]
        ax1.plot(episodes, self.metrics['train_wins'], label='Train Win Rate', marker='o')
        ax1.plot(episodes, self.metrics['eval_wins'], label='Eval Win Rate', marker='s')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Win Rate over Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Epsilon衰减
        ax2 = axes[0, 1]
        ax2.plot(episodes, self.metrics['epsilon'], color='red', marker='o')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Epsilon')
        ax2.set_title('Epsilon Decay')
        ax2.grid(True, alpha=0.3)
        
        # 3. 对手作弊率
        ax3 = axes[1, 0]
        ax3.plot(episodes, self.metrics['opponent_cheat_rates'], color='orange', marker='o')
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Cheat Rate')
        ax3.set_title('Average Opponent Cheat Rate')
        ax3.grid(True, alpha=0.3)
        
        # 4. 对手质疑率
        ax4 = axes[1, 1]
        ax4.plot(episodes, self.metrics['opponent_challenge_rates'], color='green', marker='o')
        ax4.set_xlabel('Episodes')
        ax4.set_ylabel('Challenge Rate')
        ax4.set_title('Average Opponent Challenge Rate')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        suffix = "final" if final else "latest"
        plot_path = f"{self.exp_dir}/plots/metrics_{suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if final:
            print(f"\n训练曲线已保存: {plot_path}")
    
    def load_checkpoint(self, checkpoint_path: str, stats_path: Optional[str] = None):
        """加载检查点续训"""
        print(f"加载检查点: {checkpoint_path}")
        self.agent.load_model(checkpoint_path)
        
        if stats_path and os.path.exists(stats_path):
            print(f"加载对手统计: {stats_path}")
            self.stats_manager.load_stats(stats_path)


def run_long_training():
    """运行长时间训练实验"""
    print("\n" + "="*70)
    print("长时间训练实验")
    print("="*70 + "\n")
    
    # 配置
    agent_config = {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,  # 更慢的衰减
        "epsilon_min": 0.05,
        "batch_size": 64,
        "warmup_steps": 500,
    }
    
    # 创建训练器
    trainer = AdvancedRLTrainer(
        agent_type="dqn",
        agent_config=agent_config,
        experiment_name="long_training_with_opponent_features"
    )
    
    # 训练
    trainer.train(
        num_episodes=5000,           # 5000局训练
        opponent_types=["humanlike", "simple", "simple"],
        eval_interval=100,           # 每100局评估
        eval_games=30,               # 每次评估30局
        save_interval=500,           # 每500局保存
        plot_interval=100            # 每100局更新图表
    )
    
    print(f"\n实验完成！结果保存在: {trainer.exp_dir}")


def resume_training():
    """从检查点恢复训练"""
    print("\n从检查点恢复训练功能")
    print("用法示例:")
    print("  trainer = AdvancedRLTrainer('dqn', config, 'my_experiment')")
    print("  trainer.load_checkpoint('path/to/model.pkl', 'path/to/stats.pkl')")
    print("  trainer.train(...)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        resume_training()
    else:
        run_long_training()
