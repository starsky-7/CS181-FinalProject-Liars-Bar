from typing import List, Dict
from game import Game
from player import RLPlayer, HumanLikeStrategyPlayer, SmarterStrategyPlayer, SimpleStrategyPlayer
from LinearQAgent import LinearQAgent
from DQNAgent import DQNAgent
from opponent_stats import OpponentStatsManager
from tqdm import tqdm
import random

class RLTrainer:
    """
    强化学习训练器
    """
    
    def __init__(self, agent_type: str, agent_config: Dict):
        self.agent_type = agent_type
        self.agent_config = agent_config
        if self.agent_type == "LinearQ":
            self.agent = LinearQAgent(**agent_config)
        elif self.agent_type == "dqn":
            self.agent = DQNAgent(**agent_config)

        # 创建对手统计管理器（训练模式）
        self.stats_manager = OpponentStatsManager(update_mode='train')
        
        # 创建RL玩家（传递stats_manager）
        self.rl_player = RLPlayer("RL_Player", showDetails=False, agent=self.agent, 
                                  is_training=True, stats_manager=self.stats_manager)
        
        # 注意：不在这里创建对手，而是在train/evaluate中动态创建
        # 这样可以灵活切换对手类型
    
    def train(self, num_episodes: int, opponent_types: List[str] = ["humanlike", "simple", "simple"]) -> None:
        """
        训练RL代理
        
        Args:
            num_episodes: 训练局数
            opponent_types: 对手类型列表，支持 "humanlike", "simple", "smarter"
        """
        # 根据opponent_types创建对手
        opponents = []
        opponent_type_map = {
            "humanlike": HumanLikeStrategyPlayer,
            "simple": SimpleStrategyPlayer,
            "smarter": SmarterStrategyPlayer
        }
        
        for i, opp_type in enumerate(opponent_types):
            opp_class = opponent_type_map.get(opp_type, SimpleStrategyPlayer)
            opponents.append(opp_class(f"Opponent{i+1}", showDetails=False))
        
        # 注册所有玩家到统计管理器
        self.stats_manager.register_player("RL_Player", "RLPlayer")
        for i, opp_type in enumerate(opponent_types):
            class_name = opponent_type_map[opp_type].__name__
            self.stats_manager.register_player(f"Opponent{i+1}", class_name)
        
        # 组合玩家列表
        players = [self.rl_player] + opponents
        
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            # 随机打乱玩家顺序（提升位置鲁棒性）
            random.shuffle(players)
            
            # 创建游戏实例（直接传入players和stats_manager）
            game = Game(players=players, showDetails=False, 
                       stats_manager=self.stats_manager)
            
            # 启动游戏
            game.start_game()
            
            # 重置所有玩家状态（为下一局做准备）
            for player in players:
                player.alive = True
            
            # 定期打印统计摘要
            if (episode + 1) % 100 == 0:
                print(f"\n===== Episode {episode + 1} 统计摘要 =====")
                self.stats_manager.print_summary()
            
            # 保存模型（每5000个回合）
            if (episode + 1) % 5000 == 0:
                self.agent.save_model(f"rl_models/agent_episode_{episode + 1}.pkl")
                self.stats_manager.save_stats(f"rl_models/opponent_stats_ep{episode + 1}.pkl")
                print(f"Model and stats saved at episode {episode + 1}")
    
    def evaluate(self, num_games: int, opponent_types: List[str] = ["humanlike", "simple", "simple"]) -> Dict:
        """
        评估RL代理
        
        Args:
            num_games: 评估局数
            opponent_types: 对手类型列表
        """
        # 切换到评估模式（不再更新统计）
        self.stats_manager.set_mode('eval')
        
        # 根据opponent_types创建对手
        opponents = []
        opponent_type_map = {
            "humanlike": HumanLikeStrategyPlayer,
            "simple": SimpleStrategyPlayer,
            "smarter": SmarterStrategyPlayer
        }
        
        for i, opp_type in enumerate(opponent_types):
            opp_class = opponent_type_map.get(opp_type, SimpleStrategyPlayer)
            opponents.append(opp_class(f"Opponent{i+1}", showDetails=False))
        
        # 创建评估用的RL玩家（非训练模式）
        eval_rl_player = RLPlayer("RL_Player", showDetails=False, 
                                  agent=self.agent, is_training=False,
                                  stats_manager=self.stats_manager)
        
        # 组合玩家列表
        players = [eval_rl_player] + opponents
        
        results = {
            "wins": 0,
            "losses": 0,
            "total_games": num_games
        }

        for game_num in range(num_games):
            print(f"Evaluation Game {game_num + 1}/{num_games}")
            
            # 创建游戏实例（传入stats_manager）
            game = Game(players=players, showDetails=False,
                       stats_manager=self.stats_manager)
            
            # 启动游戏
            game.start_game()
            
            # 重置玩家状态
            for player in players:
                player.alive = True
            
            # 记录结果
            if game.game_record.winner == "RL_Player":
                results["wins"] += 1
            else:
                results["losses"] += 1
        
        results["win_rate"] = results["wins"] / results["total_games"]
        
        # 恢复训练模式
        self.stats_manager.set_mode('train')
        
        return results

if __name__ == "__main__":
    # 训练配置
    agent_type = "dqn"  # "LinearQ" 或 "dqn"

    agent_config = {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon": 0.5,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01
    }
    
    # 创建训练器
    trainer = RLTrainer(agent_type, agent_config)
    
    # 训练代理（现在opponent_types参数真正有用了）
    trainer.train(num_episodes=500, opponent_types=["humanlike", "simple", "simple"])
    
    # 评估代理（可以测试不同对手组合）
    print("\n===== 评估：对阵相同对手 =====")
    results = trainer.evaluate(num_games=100, opponent_types=["humanlike", "simple", "simple"])
    print(f"Evaluation Results: {results}")
    
    print("\n===== 评估：对阵全新对手组合 =====")
    results2 = trainer.evaluate(num_games=100, opponent_types=["humanlike", "humanlike", "smarter"])
    print(f"Evaluation Results (new opponents): {results2}")
