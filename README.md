# CS181-FinalProject-Liars-Bar

## 项目概述

本项目实现了一个完整的"骗子酒馆"游戏框架，支持多种玩家类型和AI算法。玩家可以手动参与游戏，也可以让AI代理自动游戏。项目特别关注强化学习在这种具有欺骗和推理元素的游戏中的应用。

## 游戏规则

### 基本设置
- 游戏使用一副牌，包含Q、K、A和Joker四种类型的牌
- 每轮游戏随机选择Q、K或A作为目标牌
- 玩家轮流进行游戏，初始时每位玩家获得相同数量的手牌
- 游戏采用俄罗斯轮盘赌机制进行惩罚（6发子弹中1发实弹）

### 游戏流程
1. **发牌阶段**：每位玩家获得相同数量的手牌
2. **选择目标牌**：随机选择Q、K或A作为本轮目标牌
3. **出牌阶段**：当前玩家轮流出牌，必须声明打出的是目标牌
4. **质疑阶段**：下一位玩家可以选择质疑上一位玩家的出牌
5. **惩罚阶段**：如果质疑成功，出牌玩家接受惩罚；如果质疑失败，质疑玩家接受惩罚
6. **胜利条件**：最后存活的玩家获胜

### 关键机制
- **出牌规则**：玩家可以选择出目标牌或Joker，但必须声明为目标牌
- **质疑机制**：下一位玩家可以选择质疑，根据实际出牌决定惩罚对象
- **惩罚机制**：被惩罚玩家转动俄罗斯轮盘，有1/6的概率被淘汰

## 安装说明

### 环境要求
- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- tqdm

## 核心组件

### 主要文件

- **`game.py`**：实现游戏核心逻辑，包括游戏流程控制、规则检查和状态管理
- **`player.py`**：定义各种玩家类型，包括手动玩家、策略玩家和强化学习玩家
- **`DQNAgent.py`**：实现基于深度Q网络的强化学习代理
- **`advanced_trainer.py`**：增强型训练器，支持训练监控、可视化和断点续训
- **`compare_experiments.py`**：用于比较不同实验结果的工具
- **`multi_game_runner.py`**：用于批量运行多轮游戏的工具

### 玩家类型

- **手动玩家**：由人类手动操作的玩家
- **简单策略玩家**：基于固定简单策略的AI玩家
- **更智能策略玩家**：基于更复杂启发式策略的AI玩家
- **DQN玩家**：基于深度Q网络的强化学习玩家

## 使用方法

### 运行单局游戏

```python
from game import Game

# 配置玩家
player_configs = [
    {"name": "Human1", "type": "manual"},
    {"name": "SimpleAgent", "type": "simple"},
    {"name": "DQNAgent", "type": "dqn", "agent": DQNAgent(), "is_training": False}
]

# 创建游戏实例并开始游戏
game = Game(player_configs)
game.start_game()
```

### 训练强化学习代理

```python
from advanced_trainer import AdvancedRLTrainer

# 配置训练参数
agent_config = {
    "learning_rate": 1e-3,
    "discount_factor": 0.95,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.05,
    "buffer_size": 20000,
    "batch_size": 64
}

# 创建训练器并开始训练
trainer = AdvancedRLTrainer(agent_type="dqn", agent_config=agent_config, experiment_name="my_experiment")
trainer.train(num_episodes=1000, opponent_types=["simple", "smarter"])
```

### 批量运行游戏

```python
from multi_game_runner import MultiGameRunner

# 配置玩家
player_configs = [
    {"name": "Player1", "type": "simple"},
    {"name": "Player2", "type": "smarter"}
]

# 创建并运行多局游戏
runner = MultiGameRunner(player_configs, num_games=100)
results = runner.run_games()
```

## 项目结构

```
CS181-FinalProject-Liars-Bar/
├── converted_game_records/  # 转换后的游戏记录
├── demo_records/            # 演示游戏记录
│   ├── converted_game_records/  # 转换后的演示记录
│   ├── game_records/            # 原始演示记录
│   └── matchup_records/         # 对战记录
├── docs/                    # 项目文档
│   └── img/                 # 文档图片
├── experiments/             # 实验结果
│   ├── medium_training/     # 中等规模训练结果
│   └── position_test_fixed/ # 位置测试结果
├── DQNAgent.py              # DQN强化学习代理
├── LinearQAgent.py          # 线性Q学习代理
├── advanced_trainer.py      # RL训练器
├── compare_experiments.py   # 实验比较工具
├── game.py                  # 游戏核心逻辑
├── game_record.py           # 游戏记录模块
├── multi_game_runner.py     # 多局游戏运行器
├── opponent_stats.py        # 对手统计模块
├── player.py                # 玩家类型定义
├── rl_utils.py              # 强化学习工具
└── README.md                # 项目说明文档
```

## 状态空间

游戏状态空间由以下组件组成：
1. 当前玩家的回合
2. 当前玩家数量
3. 牌堆中剩余的牌数
4. 每位玩家拥有的不同类型牌的数量
5. 每位玩家采取的行动历史
6. 当前目标牌
7. 每位玩家的生命值状态

## 强化学习实现

### DQN代理

项目实现了一个基于深度Q网络的强化学习代理，具有以下特点：
- 支持经验回放
- 使用目标网络稳定训练
- 实现了ε-贪婪探索策略
- 支持Huber损失函数

### 训练流程

1. 初始化强化学习代理和对手
2. 进行多轮游戏，收集经验
3. 使用经验回放训练神经网络
4. 定期评估代理性能
5. 保存训练结果和模型

## 实验与评估

项目提供了完整的实验框架，可以：
- 训练不同配置的强化学习代理
- 比较不同算法的性能
- 可视化训练过程和结果
- 分析玩家行为和策略


## 许可证

本项目采用MIT许可证。

---

感谢您对CS181-FinalProject-Liars-Bar项目的关注和支持！
        