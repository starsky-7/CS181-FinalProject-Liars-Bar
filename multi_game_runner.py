from game import Game
from typing import Dict, List
import argparse

class MultiGameRunner:
    def __init__(self, player_configs: List[Dict[str, str]], num_games: int = 10):
        """初始化多局游戏运行器
        
        Args:
            player_configs: 玩家配置列表
            num_games: 要运行的游戏局数
        """
        self.player_configs = player_configs
        self.num_games = num_games

    def run_games(self) -> None:
        """运行指定数量的游戏"""
        results = {
            "participants": [config["name"] for config in self.player_configs],
            "player to evaluate": self.player_configs[0]["name"],
            "wins": 0,
            "losses": 0,
            "total_games": self.num_games
        }
        for game_num in range(1, self.num_games + 1):
            print(f"\n=== 开始第 {game_num}/{self.num_games} 局游戏 ===")
            
            # 创建并运行新游戏
            game = Game(self.player_configs)
            game.start_game()
            
            print(f"第 {game_num} 局游戏结束")
            
            # 记录结果(只记录第一方玩家的胜利)
            if game.game_record.winner == self.player_configs[0]["name"]:
                results["wins"] += 1
            else:
                results["losses"] += 1
        
        return results
    
    def win_rate(self, results: Dict) -> float:
        """计算胜率
        
        Args:
            results: 包含游戏结果的字典
            
        Returns:
            胜率（0到1之间的浮点数）
        """
        return results["wins"] / results["total_games"]
        
        

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='运行多局AI对战游戏',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-n', '--num-games',
        type=int,
        default=10,
        help='要运行的游戏局数 (默认: 10)'
    )
    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 配置玩家信息, 其中model为你通过API调用的模型名称
    player_configs = [
        {"name": "simple", "type": "simple"},
        {"name": "smarter", "type": "smarter"}
        # {"name": "DeepSeek", "model": "deepseek-r1"},
        # {"name": "ChatGPT", "model": "o3-mini"},
        # {"name": "Claude", "model": "claude-3.7-sonnet"},
        # {"name": "Gemini", "model": "gemini-2.0-flash-thinking"}
    ]
    
    # 创建并运行多局游戏
    runner = MultiGameRunner(player_configs, num_games=args.num_games)
    results = runner.run_games()

    # 评估智能体
    results["win_rate"] = runner.win_rate(results)
    print("\n=== 评估结果 ===")
    print(f"参与游戏的玩家: {', '.join(results['participants'])}")
    print(f"玩家 {results['player to evaluate']} 胜率: {results['win_rate']:.2%}")
    print(f"总游戏数: {results['total_games']}")
    print(f"胜利数: {results['wins']}")
    print(f"失败数: {results['losses']}")