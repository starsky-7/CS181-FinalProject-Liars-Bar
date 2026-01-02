"""
简单的对手行为统计模块
使用贝叶斯平滑的统计概率预测对手行为
"""

from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class PlayerBehaviorStats:
    """单个玩家的行为统计"""
    
    player_name: str
    player_type: str
    
    # 先验概率（基于玩家类型）
    prior_cheat_rate: float = 0.0
    prior_challenge_rate: float = 0.0
    
    # 观察计数
    total_plays: int = 0
    cheat_plays: int = 0
    total_challenge_opportunities: int = 0
    challenges_made: int = 0
    
    # 滑动窗口（可选，保留最近N次行为）
    recent_window_size: int = 100
    recent_plays: list = field(default_factory=list)  # [True/False]
    recent_challenges: list = field(default_factory=list)  # [True/False]
    
    def __post_init__(self):
        """根据玩家类型设置先验概率"""
        type_priors = {
            'SimpleStrategyPlayer': {
                'cheat_rate': 0.0,
                'challenge_rate': 0.3
            },
            'SmarterStrategyPlayer': {
                'cheat_rate': 0.1,
                'challenge_rate': 0.5
            },
            'HumanLikeStrategyPlayer': {
                'cheat_rate': 0.2,
                'challenge_rate': 0.6
            },
            'RLPlayer': {
                'cheat_rate': 0.15,
                'challenge_rate': 0.4
            }
        }
        
        if self.player_type in type_priors:
            self.prior_cheat_rate = type_priors[self.player_type]['cheat_rate']
            self.prior_challenge_rate = type_priors[self.player_type]['challenge_rate']
    
    def record_play(self, did_cheat: bool):
        """记录一次出牌行为"""
        self.total_plays += 1
        if did_cheat:
            self.cheat_plays += 1
        
        # 更新滑动窗口
        self.recent_plays.append(did_cheat)
        if len(self.recent_plays) > self.recent_window_size:
            self.recent_plays.pop(0)
    
    def record_challenge_opportunity(self, did_challenge: bool):
        """记录一次质疑机会"""
        self.total_challenge_opportunities += 1
        if did_challenge:
            self.challenges_made += 1
        
        # 更新滑动窗口
        self.recent_challenges.append(did_challenge)
        if len(self.recent_challenges) > self.recent_window_size:
            self.recent_challenges.pop(0)
    
    def get_cheat_rate(self, use_recent: bool = False, min_samples: int = 50) -> float:
        """
        获取作弊率估计（贝叶斯平滑）
        
        Args:
            use_recent: 是否只使用最近的观察
            min_samples: 最小样本数阈值（样本少时更依赖先验）
        
        Returns:
            估计的作弊概率
        """
        if use_recent and len(self.recent_plays) > 0:
            observed_rate = sum(self.recent_plays) / len(self.recent_plays)
            n_samples = len(self.recent_plays)
        elif self.total_plays > 0:
            observed_rate = self.cheat_plays / self.total_plays
            n_samples = self.total_plays
        else:
            return self.prior_cheat_rate
        
        # 贝叶斯平滑：样本少时更依赖先验
        if n_samples < min_samples:
            weight = n_samples / min_samples
            return (1 - weight) * self.prior_cheat_rate + weight * observed_rate
        
        return observed_rate
    
    def get_challenge_rate(self, use_recent: bool = False, min_samples: int = 50) -> float:
        """
        获取质疑率估计（贝叶斯平滑）
        
        Args:
            use_recent: 是否只使用最近的观察
            min_samples: 最小样本数阈值
        
        Returns:
            估计的质疑概率
        """
        if use_recent and len(self.recent_challenges) > 0:
            observed_rate = sum(self.recent_challenges) / len(self.recent_challenges)
            n_samples = len(self.recent_challenges)
        elif self.total_challenge_opportunities > 0:
            observed_rate = self.challenges_made / self.total_challenge_opportunities
            n_samples = self.total_challenge_opportunities
        else:
            return self.prior_challenge_rate
        
        # 贝叶斯平滑
        if n_samples < min_samples:
            weight = n_samples / min_samples
            return (1 - weight) * self.prior_challenge_rate + weight * observed_rate
        
        return observed_rate
    
    def get_features(self, use_recent: bool = False) -> Dict[str, float]:
        """
        获取用于RL的特征
        
        Returns:
            包含各种统计特征的字典
        """
        import math
        
        cheat_rate = self.get_cheat_rate(use_recent)
        challenge_rate = self.get_challenge_rate(use_recent)
        
        # 使用对数缩放计算置信度，让置信度增长更平滑
        # 约50样本->62%, 100样本->73%, 200样本->84%, 500样本->100%
        samples = self.total_plays + self.total_challenge_opportunities
        confidence = min(1.0, math.log(samples + 1) / (math.log(501)*1.2)) if samples > 0 else 0.0
        
        return {
            'cheat_rate': cheat_rate,
            'challenge_rate': challenge_rate,
            'aggression': cheat_rate * (1 - challenge_rate),  # 激进度（敢作弊但不爱质疑）
            'caution': (1 - cheat_rate) * challenge_rate,     # 谨慎度（不作弊但爱质疑）
            'sample_size': samples,
            'confidence': confidence
        }


class OpponentStatsManager:
    """管理所有对手的统计信息"""
    
    def __init__(self, update_mode: str = 'train'):
        """
        Args:
            update_mode: 'train' - 训练时动态更新
                        'eval' - 评估时使用固定统计
        """
        self.update_mode = update_mode
        self.player_stats: Dict[str, PlayerBehaviorStats] = {}
    
    def register_player(self, player_name: str, player_type: str):
        """注册一个新玩家"""
        if player_name not in self.player_stats:
            self.player_stats[player_name] = PlayerBehaviorStats(
                player_name=player_name,
                player_type=player_type
            )
    
    def record_play_behavior(self, player_name: str, did_cheat: bool):
        """记录出牌行为（仅训练模式）"""
        if self.update_mode == 'train' and player_name in self.player_stats:
            self.player_stats[player_name].record_play(did_cheat)
    
    def record_challenge_behavior(self, player_name: str, did_challenge: bool):
        """记录质疑行为（仅训练模式）"""
        if self.update_mode == 'train' and player_name in self.player_stats:
            self.player_stats[player_name].record_challenge_opportunity(did_challenge)
    
    def get_player_stats(self, player_name: str) -> Optional[PlayerBehaviorStats]:
        """获取指定玩家的统计"""
        return self.player_stats.get(player_name)
    
    def get_features(self, player_name: str, use_recent: bool = False) -> Dict[str, float]:
        """获取玩家的特征向量"""
        if player_name in self.player_stats:
            return self.player_stats[player_name].get_features(use_recent)
        
        # 未知玩家返回默认值
        return {
            'cheat_rate': 0.1,
            'challenge_rate': 0.4,
            'aggression': 0.06,
            'caution': 0.36,
            'sample_size': 0,
            'confidence': 0.0
        }
    
    def set_mode(self, mode: str):
        """切换模式：train 或 eval"""
        assert mode in ['train', 'eval']
        self.update_mode = mode
    
    def save_stats(self, filepath: str):
        """保存统计数据"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.player_stats, f)
        print(f"对手统计数据已保存到: {filepath}")
    
    def load_stats(self, filepath: str):
        """加载统计数据"""
        import pickle
        with open(filepath, 'rb') as f:
            self.player_stats = pickle.load(f)
        print(f"对手统计数据已加载: {len(self.player_stats)} 个玩家")
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n========== 对手行为统计摘要 ==========")
        for name, stats in self.player_stats.items():
            features = stats.get_features()
            print(f"\n{name} ({stats.player_type}):")
            print(f"  样本数: {stats.total_plays} 次出牌, {stats.total_challenge_opportunities} 次质疑机会")
            print(f"  作弊率: {features['cheat_rate']:.2%}")
            print(f"  质疑率: {features['challenge_rate']:.2%}")
            print(f"  激进度: {features['aggression']:.2%}")
            print(f"  谨慎度: {features['caution']:.2%}")
            print(f"  置信度: {features['confidence']:.2%}")
        print("=" * 40 + "\n")
