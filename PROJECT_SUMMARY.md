# 骗子酒馆 RL Agent 项目总结

## 项目概述

本项目开发了一个基于深度强化学习（DQN）的骗子酒馆游戏AI，核心创新在于**对手行为建模**，使Agent能够根据不同对手的心理特征调整策略。

---

## 核心问题与解决方案

### 问题1：RL Agent忽略对手信息

**发现的问题**：
- 原始实现中，Agent的state只包含自己的手牌信息
- 完全不知道在和谁对局
- 无法针对不同对手调整策略
- 位置打乱后策略失效

**解决方案**：
1. **上帝视角数据收集**：`Game`类记录每次出牌/质疑的真实情况
2. **对手统计建模**：`OpponentStatsManager`统计每个对手的作弊率、质疑率等
3. **特征工程**：在state中加入4个对手特征
4. **受限视角决策**：Agent只看统计特征，不看真实手牌

### 问题2：特征维度变化

**原始特征**（9个）：
```
(round_id, target_card, bullet_pos, q_count, k_count, a_count, 
 joker_count, target_count, hand_size)
```

**新特征**（13个）：
```
原9个 + (opponent_cheat_rate, opponent_challenge_rate, 
         opponent_aggression, opponent_confidence)
```

**影响**：
- 需要重新训练模型
- FeatureExtractor需要更新解析逻辑
- 出牌/质疑两阶段特征维度不同

---

## 技术架构

### 1. 对手建模模块 (`opponent_stats.py`)

```python
class OpponentStatsManager:
    """
    功能：
    - 记录对手出牌/质疑行为（上帝视角）
    - 计算统计特征（作弊率、质疑率等）
    - 贝叶斯平滑（样本少时依赖先验）
    - 支持训练/评估模式切换
    """
```

**关键特征**：
- `cheat_rate`: 对手作弊概率
- `challenge_rate`: 对手质疑概率
- `aggression`: 激进度（敢作弊但不爱质疑）
- `confidence`: 统计置信度

### 2. 状态编码器 (`rl_utils.py`)

```python
class StateEncoder:
    def encode_play_state(..., stats_manager):
        # 提取对手名字
        opponent_name = self._extract_opponent_name(play_decision_info)
        
        # 获取对手特征
        features = stats_manager.get_features(opponent_name)
        
        # 合并到state
        return (*base_state, *opponent_features)
```

### 3. 增强型训练器 (`advanced_trainer.py`)

**新功能**：
- 自动评估和监控
- 训练曲线可视化
- 断点续训
- 实验管理

**监控指标**：
- 训练胜率（滑动窗口）
- 评估胜率
- Epsilon衰减
- 对手统计收敛

---

## 实验设计

### 对比实验

| 组别 | 配置 | 预期结果 |
|------|------|----------|
| 基线 | 无对手特征 | 胜率 ~25-30% |
| 实验组 | 有对手特征 | 胜率提升 5-10% |

### 泛化测试

| 对手组合 | 难度 | 测试目的 |
|----------|------|----------|
| [simple, simple, simple] | 低 | 基础能力 |
| [humanlike, simple, simple] | 中 | 训练配置 |
| [humanlike, humanlike, humanlike] | 高 | 极限挑战 |
| [smarter, smarter, simple] | 中高 | 泛化能力 |

### 鲁棒性测试

- **固定位置**：玩家顺序不变
- **随机位置**：每局随机打乱
- **判断标准**：胜率差异<5%为良好

---

## 关键发现

### 1. 对手特征的重要性

**理论预期**：
- 骗子酒馆依赖心理博弈
- 不同对手有不同策略
- 对手建模应该有效

**验证方法**：
```python
# 运行对比实验
python compare_experiments.py
```

### 2. 训练时长需求

| 训练局数 | 预期效果 |
|----------|----------|
| 100-500 | 验证代码正确性 |
| 1000-2000 | 观察学习趋势 |
| 5000+ | 达到最优性能 |

### 3. 超参数敏感性

**关键参数**：
- `epsilon_decay`: 探索-利用平衡
- `learning_rate`: 学习速度
- `warmup_steps`: 统计稳定性

---

## 使用指南

### 快速开始

```bash
# 1. 快速测试（5-10分钟）
python train_configs.py quick

# 2. 查看训练曲线
# 打开 experiments/quick_test/plots/metrics_final.png

# 3. 运行评估
python final_evaluation.py
```

### 中等规模训练

```bash
# 训练1000局（20-40分钟）
python train_configs.py medium

# 评估
python final_evaluation.py
```

### 长时间训练

```bash
# 训练5000局（2-3小时）
python train_configs.py long

# 完整评估
python final_evaluation.py
```

### 对比实验

```bash
# 有/无对手特征对比
python compare_experiments.py
```

---

## 文件结构

```
CS181-FinalProject-Liars-Bar/
├── opponent_stats.py           # 对手统计模块 ⭐新增
├── rl_utils.py                 # 状态编码（已修改）
├── player.py                   # RLPlayer（已修改）
├── game.py                     # Game类（已修改）
├── rl_trainer.py               # 基础训练器（已修改）
├── advanced_trainer.py         # 增强训练器 ⭐新增
├── train_configs.py            # 快速启动脚本 ⭐新增
├── compare_experiments.py      # 对比实验 ⭐新增
├── test_opponent_features.py   # 单元测试 ⭐新增
├── final_evaluation.py         # 最终评估 ⭐新增
├── TRAINING_GUIDE.md           # 训练指南 ⭐新增
└── experiments/                # 实验结果目录 ⭐新增
    └── {exp_name}/
        ├── config.json
        ├── metrics.json
        ├── models/
        └── plots/
```

---

## 后续优化方向

### 1. 模型架构
- [ ] 使用LSTM处理历史序列
- [ ] 添加Attention机制关注关键对手
- [ ] 尝试Multi-agent RL

### 2. 特征工程
- [ ] 添加更多对手特征（风险偏好、手牌管理风格等）
- [ ] 使用Embedding表示对手身份
- [ ] 引入游戏阶段特征（早期/中期/后期）

### 3. 训练策略
- [ ] 课程学习（从简单对手到困难对手）
- [ ] 对抗训练（自我博弈）
- [ ] 迁移学习（预训练 + 微调）

### 4. 评估完善
- [ ] 与人类玩家对战
- [ ] 分析决策可解释性
- [ ] 统计不同场景下的策略差异

---

## 成果总结

### 技术贡献
1. ✅ 实现了对手行为统计和建模
2. ✅ 成功将对手特征集成到RL框架
3. ✅ 建立了完整的训练和评估流程
4. ✅ 提供了可复现的实验框架

### 预期成果
1. RL Agent胜率提升5-10%
2. 对不同对手有针对性策略
3. 良好的泛化和鲁棒性

### 学习收获
1. 强化学习在不完全信息游戏中的应用
2. 对手建模的重要性
3. 特征工程对性能的影响
4. 实验设计和结果分析

---

## 参考资料

### 相关论文
- DQN: Mnih et al. "Playing Atari with Deep Reinforcement Learning" (2013)
- Opponent Modeling: He et al. "Opponent Modeling in Deep Reinforcement Learning" (2016)

### 工具和框架
- PyTorch: 深度学习框架
- tqdm: 进度条
- matplotlib: 可视化

---

## 联系方式

如有问题，请查看：
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 训练指南
- [README.md](README.md) - 项目说明

---

**最后更新**: 2026-01-03
