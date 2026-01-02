# visualization/generate_charts.py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def create_all_visualizations():
    """生成所有图表，即使数据不完整"""
    
    # 确保输出目录存在
    output_dir = "visualization/charts"
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试读取数据
    try:
        with open('logs/dqn_monitor_logs.json', 'r') as f:
            data = json.load(f)
        print("✓ 找到监控数据")
    except FileNotFoundError:
        print("✗ 未找到监控数据，创建示例数据用于演示")
        data = create_sample_data()
    
    # 生成四个图表
    create_q_value_chart(data, output_dir)
    create_action_heatmap(data, output_dir)
    create_replay_buffer_chart(data, output_dir)
    create_training_summary(data, output_dir)
    
    print(f"\n✅ 图表已生成到: {output_dir}/")
    print("1. q_value_analysis.png - Q-Value分析")
    print("2. action_strategy.png - 动作策略")
    print("3. replay_buffer.png - 经验分析")
    print("4. training_summary.png - 训练总结")

def create_sample_data():
    """创建示例数据用于演示图表"""
    return {
        "q_values_stats": {
            "mean": list(np.random.uniform(-1, 1, 100)),
            "std": list(np.random.uniform(0, 0.5, 100)),
            "max": list(np.random.uniform(0.5, 2, 100)),
            "min": list(np.random.uniform(-2, -0.5, 100))
        },
        "action_choices": {
            "play": {"action_0": 150, "action_1": 120, "action_2": 80},
            "challenge": {"challenge_yes": 45, "challenge_no": 65}
        },
        "state_action_records": [
            {
                "target_count": np.random.randint(0, 4),
                "bullet_pos": np.random.randint(0, 6),
                "action_type": np.random.choice(["challenge_yes", "challenge_no"]),
                "phase": "challenge"
            }
            for _ in range(50)
        ],
        "reward_distribution": {
            "positive": 25,
            "negative": 15,
            "zero": 60
        },
        "done_distribution": {
            "done": 8,
            "not_done": 42
        }
    }

def create_q_value_chart(data, output_dir):
    """创建Q-Value分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    stats = data["q_values_stats"]
    steps = range(len(stats["mean"])) if stats["mean"] else []
    
    # 图表1: Q-Value均值趋势
    if steps:
        axes[0, 0].plot(steps, stats["mean"], 'b-', linewidth=2)
        axes[0, 0].set_title('Q-Value Mean Trend', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Q-Value均值')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].fill_between(steps, 
                               np.array(stats["mean"]) - np.array(stats["std"]),
                               np.array(stats["mean"]) + np.array(stats["std"]),
                               alpha=0.2, color='blue')
    else:
        axes[0, 0].text(0.5, 0.5, '无Q-Value数据\n请检查监控是否启用',
                       ha='center', va='center', fontsize=12)
        axes[0, 0].set_title('Q-Value Mean Trend', fontsize=12, fontweight='bold')
    
    # 图表2: Q-Value波动性
    if stats["std"]:
        axes[0, 1].plot(steps, stats["std"], 'r-', linewidth=2)
        axes[0, 1].set_title('Q-Value Standard Deviation', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 图表3: Q-Value范围
    if stats["max"] and stats["min"]:
        axes[1, 0].plot(steps, stats["max"], 'g-', label='最大值', alpha=0.7)
        axes[1, 0].plot(steps, stats["min"], 'r-', label='最小值', alpha=0.7)
        axes[1, 0].fill_between(steps, stats["min"], stats["max"], alpha=0.1, color='gray')
        axes[1, 0].set_title('Q-Value Range Variation', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Q-Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 图表4: 收敛指示器
    if len(steps) > 10:
        # 计算滑动平均
        window_size = max(5, len(steps) // 20)
        if len(stats["mean"]) >= window_size:
            mean_ma = np.convolve(stats["mean"], np.ones(window_size)/window_size, mode='valid')
            steps_ma = steps[window_size-1:]
            
            axes[1, 1].plot(steps_ma, mean_ma, 'purple', linewidth=3, label=f'{window_size}步滑动平均')
            axes[1, 1].plot(steps, stats["mean"], 'b-', alpha=0.3, label='Original number')
            axes[1, 1].set_title(f'Q-Value Convergence Trend (Moving average)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Q-Value mean')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/q_value_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_action_heatmap(data, output_dir):
    """创建动作策略热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    records = data.get("state_action_records", [])
    
    if records and len(records) > 10:
        # 提取挑战阶段的记录
        challenge_records = [r for r in records if r.get("phase") == "challenge"]
        
        if challenge_records:
            # 创建决策矩阵
            max_target = max(r.get("target_count", 0) for r in challenge_records)
            max_bullet = max(r.get("bullet_pos", 0) for r in challenge_records)
            
            challenge_count = np.zeros((max_target + 1, max_bullet + 1))
            total_count = np.zeros((max_target + 1, max_bullet + 1))
            
            for record in challenge_records:
                target = record.get("target_count", 0)
                bullet = record.get("bullet_pos", 0)
                
                if 0 <= target <= max_target and 0 <= bullet <= max_bullet:
                    total_count[target, bullet] += 1
                    if record.get("action_type") == "challenge_yes":
                        challenge_count[target, bullet] += 1
            
            # 计算Challenge Probability
            challenge_prob = np.zeros_like(challenge_count, dtype=float)
            for i in range(challenge_prob.shape[0]):
                for j in range(challenge_prob.shape[1]):
                    if total_count[i, j] > 0:
                        challenge_prob[i, j] = challenge_count[i, j] / total_count[i, j]
            
            # 热力图
            im = axes[0].imshow(challenge_prob, cmap='RdYlBu_r', aspect='auto')
            axes[0].set_title('Challenge Decision Heatmap', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Bullet Position')
            axes[0].set_ylabel('Target Card Number')
            plt.colorbar(im, ax=axes[0])
            
            # 添加数值标注
            for i in range(challenge_prob.shape[0]):
                for j in range(challenge_prob.shape[1]):
                    if total_count[i, j] > 0:
                        text = axes[0].text(j, i, f'{challenge_prob[i, j]:.2f}',
                                           ha="center", va="center", 
                                           color="black", fontsize=8)
    
    # 动作选择分布饼图
    action_choices = data.get("action_choices", {})
    if action_choices.get("challenge"):
        challenge_data = action_choices["challenge"]
        labels = list(challenge_data.keys())
        sizes = list(challenge_data.values())
        
        wedges, texts, autotexts = axes[1].pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=['#ff9999', '#66b3ff'],
            startangle=90
        )
        axes[1].set_title('Challenge Decision Distribution', fontsize=12, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, '无质疑决策数据\n请收集更多游戏数据',
                    ha='center', va='center', fontsize=10)
        axes[1].set_title('Challenge Decision Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/action_strategy.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_replay_buffer_chart(data, output_dir):
    """创建经验回放分析图表"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Reward Distribution
    reward_dist = data.get("reward_distribution", {})
    if reward_dist:
        labels = ['Positive Reward', 'Negative Reward', 'Zero Reward']
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        # 映射键名
        mapping = {'positive': 'Positive Reward', 'negative': 'Negative Reward', 'zero': 'Zero Reward'}
        values = [reward_dist.get(k, 0) for k in ['positive', 'negative', 'zero']]
        
        axes[0].pie(values, labels=labels, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
        axes[0].set_title('Reward Distribution', fontsize=12, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'Non-Reward data', ha='center', va='center', fontsize=10)
        axes[0].set_title('Reward Distribution', fontsize=12, fontweight='bold')
    
    # 2. Terminal State Distribution
    done_dist = data.get("done_distribution", {})
    if done_dist:
        labels = ['Terminal', '非Terminal']
        values = [done_dist.get('done', 0), done_dist.get('not_done', 0)]
        colors = ['#e74c3c', '#3498db']
        
        bars = axes[1].bar(labels, values, color=colors)
        axes[1].set_title('Terminal State Distribution', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Sample Number')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val}', ha='center', va='bottom')
    else:
        axes[1].text(0.5, 0.5, 'Non-Terminal Data', ha='center', va='center', fontsize=10)
        axes[1].set_title('Terminal State Distribution', fontsize=12, fontweight='bold')
    
    # 3. Action Phase Distribution
    action_choices = data.get("action_choices", {})
    if action_choices:
        play_count = sum(action_choices.get("play", {}).values())
        challenge_count = sum(action_choices.get("challenge", {}).values())
        
        labels = ['Play Phase', 'Challenge Phase']
        values = [play_count, challenge_count]
        colors = ['#9b59b6', '#f1c40f']
        
        bars = axes[2].bar(labels, values, color=colors)
        axes[2].set_title('Action Phase Distribution', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Action number')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val}', ha='center', va='bottom')
    else:
        axes[2].text(0.5, 0.5, 'No Action Choose Data', ha='center', va='center', fontsize=10)
        axes[2].set_title('Action Phase Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/replay_buffer.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_training_summary(data, output_dir):
    """Train Summaru Chart"""
    fig = plt.figure(figsize=(10, 8))
    
    # 添加标题
    plt.suptitle('DQN Training Analysis Report', fontsize=16, fontweight='bold', y=0.95)
    
    # 创建4个子图
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    # 1. Data Collection Status
    record_count = len(data.get("state_action_records", []))
    ax1.bar(['Status-action record'], [record_count], color='skyblue')
    ax1.set_title('Data Collection Status')
    ax1.set_ylabel('Quantity Record')
    ax1.text(0, record_count, str(record_count), ha='center', va='bottom')
    
    # 2. Training Steps
    steps = len(data.get("q_values_stats", {}).get("mean", []))
    ax2.bar(['Q-Value Step Record'], [steps], color='lightgreen')
    ax2.set_title('Training Progress')
    ax2.set_ylabel('Step Record')
    ax2.text(0, steps, str(steps), ha='center', va='bottom')
    
    # 3. 数据质量指示器
    has_q_data = steps > 0
    has_action_data = record_count > 0
    has_reward_data = bool(data.get("reward_distribution", {}))
    
    indicators = ['Q-Value data', 'Action Data', 'Reward Data']
    values = [int(has_q_data), int(has_action_data), int(has_reward_data)]
    colors = ['green' if v else 'red' for v in values]
    
    ax3.bar(indicators, values, color=colors)
    ax3.set_title('Data Quality Check')
    ax3.set_ylim(0, 1)
    
    # 4. 建议
    suggestions = []
    if steps == 0:
        suggestions.append("1. Use Q-Value monitoring")
    if record_count == 0:
        suggestions.append("2. Collect More Data")
    if not has_reward_data:
        suggestions.append("3. Record Reward Distribution")
    
    if not suggestions:
        suggestions = ["✓ Data collection normal", "✓ Continue training for more episodes"]
    
    ax4.text(0.1, 0.5, '\n'.join(suggestions), 
            fontsize=10, va='center', linespacing=1.5)
    ax4.set_title('Improvement Suggestions')
    ax4.axis('off')
    
    # 添加时间戳
    plt.figtext(0.02, 0.02, f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
               fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_all_visualizations()