from typing import List, Dict
from game import Game
from player import RLPlayer
from rl_agent import RLAgent
from tqdm import tqdm

class RLTrainer:
    """
    强化学习训练器
    """
    
    def __init__(self, agent_config: Dict):
        self.agent_config = agent_config
        self.agent = RLAgent(**agent_config)
    
    def train(self, num_episodes: int, opponent_type: str = "simple") -> None:
        """
        训练RL代理
        """
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            
            # 创建玩家配置
            player_configs = [
                {"name": "RL_Player", "type": "rl", "agent": self.agent, "is_training": True},
                {"name": "Opponent", "type": opponent_type}
            ]
            
            # 创建游戏实例
            game = Game(player_configs, showDetails=False)
            
            # 启动游戏
            game.start_game()
            # self.agent.print_q_table_summary(output_file=f"q_table_summary.txt")
            
            # 保存模型（每100个回合）
            if (episode + 1) % 5000 == 0:
                self.agent.save_model(f"rl_models/agent_episode_{episode + 1}.pkl")
                print(f"Model saved at episode {episode + 1}")
                self.agent.print_q_table_summary(output_file=f"q_table_summary.txt")
    
    def evaluate(self, num_games: int, opponent_type: str = "simple") -> Dict:
        """
        评估RL代理
        """
        results = {
            "wins": 0,
            "losses": 0,
            "total_games": num_games
        }
        
        for game_num in range(num_games):
            print(f"Evaluation Game {game_num + 1}/{num_games}")
            
            # 创建玩家配置
            player_configs = [
                {"name": "RL_Player", "type": "rl", "agent": self.agent, "is_training": False},
                {"name": "Opponent", "type": opponent_type}
            ]
            
            # 创建游戏实例
            game = Game(player_configs, showDetails=True)
            
            # 启动游戏
            game.start_game()
            
            # 记录结果
            if game.game_record.winner == "RL_Player":
                results["wins"] += 1
            else:
                results["losses"] += 1
        
        results["win_rate"] = results["wins"] / results["total_games"]
        print(f"评估完成。Q表摘要:")
        self.agent.print_q_table_summary()
        return results

if __name__ == "__main__":
    # 训练配置
    agent_config = {
        "learning_rate": 0.01,
        "discount_factor": 0.95,
        "epsilon": 0.5,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01
    }
    
    # 创建训练器
    trainer = RLTrainer(agent_config)
    
    # 训练代理
    trainer.train(num_episodes=50000)
    
    # 评估代理
    results = trainer.evaluate(num_games=100)
    print(f"Evaluation Results: {results}")
