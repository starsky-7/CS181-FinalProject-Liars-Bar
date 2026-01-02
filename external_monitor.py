# external_monitor.py
"""
外部Monitoring器 - 不修改DQNAgent，通过hook方式收集数据
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

class ExternalTrainingMonitor:
    """外部TrainingMonitoring器，不修改原代码"""
    
    def __init__(self, save_dir="external_monitor_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training数据
        self.data = {
            'episodes': [],
            'steps': [],
            'losses': [],
            'epsilons': [],
            'win_rates': [],
            'rewards': []
        }
        
        self.current_episode = 0
        self.current_step = 0
        
        print(f"[外部Monitoring] 初始化完成，数据目录: {save_dir}")
    
    def log_episode_start(self, episode):
        """记录回合开始"""
        self.current_episode = episode
        print(f"[外部Monitoring] 开始第{episode}回合")
    
    def log_training_step(self, loss=None, epsilon=None, q_value=None):
        """记录Training步骤（从外部调用）"""
        self.current_step += 1
        
        if loss is not None:
            self.data['losses'].append(loss)
            self.data['steps'].append(self.current_step)
        
        if epsilon is not None:
            self.data['epsilons'].append(epsilon)
    
    def log_episode_result(self, win_rate=None, reward=None):
        """记录回合结果"""
        self.data['episodes'].append(self.current_episode)
        
        if win_rate is not None:
            self.data['win_rates'].append(win_rate)
            print(f"[外部Monitoring] 第{self.current_episode}Episodes Win Rate: {win_rate:.2%}")
        
        if reward is not None:
            self.data['rewards'].append(reward)
    
    def create_plots(self):
        """生成可视化Chart"""
        print(f"\n[Exteral Monitoring] 生成Chart...")
        print(f"Total steps: {self.current_step}")
        print(f"Total Episodes: {self.current_episode}")
        
        # 检查是否有数据
        if len(self.data['losses']) == 0:
            print("⚠️  警告：没有Training数据")
            self._create_demo_plots()
            return
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Loss Curve
        if self.data['losses']:
            ax1 = plt.subplot(2, 2, 1)
            steps = self.data['steps'][:len(self.data['losses'])]
            ax1.plot(steps, self.data['losses'], 'b-', alpha=0.5, linewidth=1)
            
            # 滑动平均
            if len(self.data['losses']) > 50:
                window = min(100, len(self.data['losses']) // 10)
                loss_series = pd.Series(self.data['losses'])
                loss_smooth = loss_series.rolling(window=window).mean()
                ax1.plot(steps, loss_smooth, 'r-', linewidth=2, label=f'{window}steps in average')
                ax1.legend()
            
            ax1.set_title('Loss Curve')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
        
        # 2. 探索率
        if self.data['epsilons']:
            ax2 = plt.subplot(2, 2, 2)
            eps_steps = self.data['steps'][:len(self.data['epsilons'])]
            ax2.plot(eps_steps, self.data['epsilons'], 'orange', linewidth=2)
            ax2.set_title('Exploration Rate Decay')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Exploration Rate (ε)')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        # 3. Win Rate Curve
        if self.data['win_rates']:
            ax3 = plt.subplot(2, 2, 3)
            ax3.plot(self.data['episodes'], self.data['win_rates'], 'g-', marker='o', linewidth=2)
            ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='基准线')
            ax3.fill_between(self.data['episodes'], 0, self.data['win_rates'], alpha=0.3, color='green')
            ax3.set_title('Evaluation Win Rate')
            ax3.set_xlabel('TrainingEpisodes')
            ax3.set_ylabel('Win Rate')
            ax3.set_ylim([0, 1.1])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 数据统计
        ax4 = plt.subplot(2, 2, 4)
        
        stats_text = []
        if self.data['losses']:
            stats_text.append(f"Total Training Steps: {self.current_step}")
            stats_text.append(f"Final Loss: {self.data['losses'][-1]:.4f}")
            stats_text.append(f"Loss Range: [{min(self.data['losses']):.4f}, {max(self.data['losses']):.4f}]")
        
        if self.data['epsilons']:
            stats_text.append(f"Final Exploration Rate: {self.data['epsilons'][-1]:.4f}")
        
        if self.data['win_rates']:
            stats_text.append(f"Best Win Rate: {max(self.data['win_rates']):.2%}")
            stats_text.append(f"Final Win Rate: {self.data['win_rates'][-1]:.2%}")
        
        if not stats_text:
            stats_text = ["暂无数据"]
        
        ax4.text(0.1, 0.5, '\n'.join(stats_text), 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')
        ax4.set_title('Training Statistics')
        
        plt.suptitle(f'External Monitoring - {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=16)
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.save_dir, f'external_monitor_{timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 外部MonitoringChart已保存: {plot_path}")
        
        # 保存数据
        self.save_data()
        
        return plot_path
    
    def _create_demo_plots(self):
        """创建演示Chart（当没有数据时）"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        demo_text = "请连接Monitoring数据\n\n使用方法:\n1. 在Training循环中调用:\n   monitor.log_training_step(loss, epsilon)\n2. 在评估时调用:\n   monitor.log_episode_result(win_rate)"
        
        for ax in axes.flat:
            ax.text(0.5, 0.5, demo_text, 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=11)
            ax.axis('off')
        
        plt.suptitle('外部Monitoring器 - 等待数据输入', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, 'demo_waiting_for_data.png')
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"📋 演示Chart已保存: {plot_path}")
    
    def save_data(self):
        """保存数据到文件"""
        data_path = os.path.join(self.save_dir, 'training_data.json')
        
        # 只保存有效长度的数据
        save_data = {}
        for key, values in self.data.items():
            if values:
                save_data[key] = values
        
        with open(data_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"💾 Training数据已保存: {data_path}")
        return data_path

# ============= 使用示例 =============

def simulate_training():
    """模拟Training过程"""
    print("模拟Training过程...")
    
    monitor = ExternalTrainingMonitor()
    
    # 模拟100回合Training
    for episode in range(1, 101):
        monitor.log_episode_start(episode)
        
        # 模拟每回合100步Training
        for step in range(100):
            # 模拟loss和epsilon
            base_loss = 0.5 * np.exp(-episode/20)
            loss = base_loss + np.random.normal(0, 0.05)
            epsilon = max(0.01, 0.5 * np.exp(-episode/50))
            
            # 记录Training步骤
            if step % 10 == 0:  # 每10步记录一次
                monitor.log_training_step(loss=loss, epsilon=epsilon)
        
        # 每10回合评估一次
        if episode % 10 == 0:
            win_rate = min(0.95, 0.2 + 0.7 * (episode/100))
            monitor.log_episode_result(win_rate=win_rate)
    
    # 生成Chart
    monitor.create_plots()
    
    return monitor

if __name__ == "__main__":
    print("External Monitoring Test")
    print("="*60)
    
    # 运行模拟
    monitor = simulate_training()