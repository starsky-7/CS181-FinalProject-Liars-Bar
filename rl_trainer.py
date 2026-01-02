from typing import List, Dict
from game import Game
from player import RLPlayer
from LinearQAgent import LinearQAgent
from DQNAgent import DQNAgent
from tqdm import tqdm

# 在你的 rl_trainer.py 中
#from tuning_results.Best _config import Best _CONFIG

# 在rl_trainer.py开头添加
from external_monitor import ExternalTrainingMonitor

class RLTrainer:
    def __init__(self, agent_type: str, agent_config: Dict):
        self.agent_type = agent_type
        self.agent_config = agent_config
        if self.agent_type == "LinearQ":
            self.agent = LinearQAgent(**agent_config)
        elif self.agent_type == "dqn":
            self.agent = DQNAgent(**agent_config)
        
        # 添加外部Monitoring器
        self.monitor = ExternalTrainingMonitor()
    
    def train(self, num_episodes: int, opponent_types: List[str] = ["simple", "smarter"]) -> None:
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            # 记录回合开始
            self.monitor.log_episode_start(episode)
            
            # 原有的Training代码...
            player_configs = [
                {"name": "RL_Player", "type": self.agent_type, "agent": self.agent, "is_training": True},
                {"name": "Opponent1", "type": opponent_types[0]},
                {"name": "Opponent2", "type": opponent_types[1]},
                {"name": "Opponent3", "type": opponent_types[2]}
            ]
            
            game = Game(player_configs, showDetails=False)
            game.start_game()
            
            # 这里的问题：我们无法直接获取loss和epsilon
            # 但可以从agent的属性获取
            if hasattr(self.agent, 'epsilon'):
                # 记录epsilon（每回合记录一次）
                self.monitor.log_training_step(epsilon=self.agent.epsilon)
            
            # 每100回合评估一次
            if (episode + 1) % 100 == 0:
                eval_results = self.evaluate(num_games=20, opponent_types=opponent_types)
                win_rate = eval_results["win_rate"]
                self.monitor.log_episode_result(win_rate=win_rate)
        
        # Training结束后生成Chart
        print("\nTraining完成，生成Chart...")
        self.monitor.create_plots()
        ###添加完成

agent_config = {
    "learning_rate": 0.000626,
    "discount_factor": 0.95,
    "epsilon": 1.0,
    "epsilon_decay": 0.99,
    "epsilon_min": 0.01,
    "batch_size": 128,
    "buffer_size": 20000,
    "warmup_steps": 500,
    "target_update_every": 500,
}
agent_config["num_episodes"] = 5000  # 调整TrainingEpisodes

class RLTrainer:
    """
    强化学习Training器
    """
    
    def __init__(self, agent_type: str, agent_config: Dict):
        self.agent_type = agent_type
        self.agent_config = agent_config
        if self.agent_type == "LinearQ":
            self.agent = LinearQAgent(**agent_config)
        elif self.agent_type == "dqn":
            self.agent = DQNAgent(**agent_config)
    
    def train(self, num_episodes: int, opponent_types: List[str] = ["simple", "smarter"]) -> None:
        """
        TrainingRL代理
        """
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            
            # 创建玩家配置
            player_configs = [
                {"name": "RL_Player", "type": self.agent_type, "agent": self.agent, "is_training": True},
                {"name": "Opponent1", "type": opponent_types[0]},
                {"name": "Opponent2", "type": opponent_types[1]},
                {"name": "Opponent3", "type": opponent_types[2]}
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
                # self.agent.print_q_table_summary(output_file=f"q_table_summary.txt")
                
                # 每100个episode评估一次
                if (episode + 1) % 100 == 0:
                    print(f"\nTraining进度: {episode+1}/{num_episodes} 回合")
                    
                    # 这里先不进行评估，只打印信息
                    if hasattr(self.agent, 'visualizer'):
                        try:
                            # 只保存Training数据，不进行评估（评估需要单独运行）
                            self.agent.visualizer.save_training_data(f"training_data_{episode+1}")
                            print(f"✅ Training数据已保存")
                        except:
                            pass
            # ====== 暂时注释掉评估部分，先让Training跑起来 ======
            # if (episode + 1) % 100 == 0:
            #     print(f"\n回合 {episode+1}/{num_episodes}")
            #     
            #     # 这里会有问题，先注释
            #     # results = trainer.agent.evaluate(num_games=20)
        
        
                
                # 记录到可视化器
                trainer.agent.visualizer.add_evaluation_point(
                    episode=episode,
                    win_rate=win_rate
                )
                
                # 每500个episode生成一次Chart
                if (episode + 1) % 500 == 0:
                    trainer.agent.visualizer.create_essential_plots()
    
    def evaluate(self, num_games: int, opponent_types: List[str]) -> Dict:
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
                {"name": "RL_Player", "type": self.agent_type, "agent": self.agent, "is_training": False},
                {"name": "Opponent1", "type": opponent_types[0]},
                {"name": "Opponent2", "type": opponent_types[1]},
                {"name": "Opponent3", "type": opponent_types[2]}
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
        # print(f"评估完成。Q表摘要:")
        # self.agent.print_q_table_summary()
        return results

if __name__ == "__main__":
    # Training配置
    agent_type = "dqn"  # "LinearQ" 或 "dqn"

    agent_config = {
        "learning_rate": 0.01,
        "discount_factor": 0.95,
        "epsilon": 0.5,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01
    }
    
    # 创建Training器
    trainer = RLTrainer(agent_type, agent_config)
    
    # Training代理
    trainer.train(num_episodes=500, opponent_types=["simple", "smarter", "smarter"])
    
    # 评估代理
    results = trainer.evaluate(num_games=200, opponent_types=["simple", "smarter", "smarter"])
    print(f"Evaluation Results: {results}")