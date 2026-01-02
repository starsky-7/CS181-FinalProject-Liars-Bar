# 训练优化指南

## 超参数调优建议

### 1. Epsilon衰减策略

**当前问题**：固定衰减率可能不适合所有训练阶段

**优化方案**：
```python
# 阶段性衰减
if episode < 1000:
    epsilon_decay = 0.999   # 前期快速探索
elif episode < 3000:
    epsilon_decay = 0.9995  # 中期平衡
else:
    epsilon_decay = 0.9999  # 后期精细调整
```

### 2. 学习率调度

**建议**：使用学习率衰减
```python
# 在DQNAgent中添加
if self.grad_updates % 10000 == 0:
    for param_group in self.optimizer.param_groups:
        param_group['lr'] *= 0.9
```

### 3. 对手多样性

**策略1：动态对手组合**
```python
opponent_pools = [
    ["humanlike", "simple", "simple"],
    ["humanlike", "humanlike", "simple"],
    ["simple", "simple", "simple"],
    ["humanlike", "smarter", "simple"],
]

# 每1000局切换对手
opponent_types = opponent_pools[episode // 1000 % len(opponent_pools)]
```

**策略2：随机对手**
```python
import random
opponent_types = random.choice(opponent_pools)
```

### 4. 奖励塑形

**当前**：只在游戏结束时给奖励

**优化**：添加中间奖励
```python
# 在player.py的reflect方法中
if "质疑成功" in round_result:
    self.current_reward += 0.5
elif "成功骗过对手" in round_result:
    self.current_reward += 0.3
```

### 5. 经验回放优化

**优先级经验回放**：优先采样重要的经验
```python
# 在replay buffer中记录TD-error
# 采样时按TD-error加权
```

## 训练监控

### 关键指标

1. **胜率曲线**
   - 训练胜率应该稳定上升
   - 评估胜率滞后于训练胜率（正常现象）
   - 如果胜率不增长，检查epsilon和学习率

2. **Epsilon值**
   - 应该从1.0逐渐降到0.05
   - 如果衰减太快，agent没有充分探索
   - 如果衰减太慢，训练时间过长

3. **对手统计**
   - 作弊率应该收敛到稳定值
   - 置信度应该逐渐增加
   - 如果波动很大，增加统计样本数

### 异常诊断

**问题1：胜率不增长**
- 检查epsilon是否太高（过度探索）
- 检查学习率是否合适
- 检查网络是否过拟合（增加正则化）

**问题2：胜率突然下降**
- 可能是灾难性遗忘
- 降低学习率
- 增加replay buffer大小

**问题3：对手特征不收敛**
- 增加warmup_steps
- 使用贝叶斯平滑（已实现）
- 检查对手策略是否真的稳定

## 性能优化

### 1. 加速训练

**并行化**（需要修改代码）：
```python
from multiprocessing import Pool

def train_episode(args):
    # 单局训练
    pass

with Pool(4) as p:
    results = p.map(train_episode, range(num_episodes))
```

**GPU加速**（已支持）：
- 确保PyTorch使用GPU
- 增加batch_size利用GPU

### 2. 减少内存使用

```python
# 限制replay buffer大小
buffer_size = 10000  # 而不是20000

# 定期清理
if episode % 1000 == 0:
    import gc
    gc.collect()
```

### 3. 快速实验

```python
# 小规模验证
quick_config = {
    "num_episodes": 100,
    "eval_interval": 20,
    "eval_games": 10,
}

# 验证通过后再长时间训练
```

## 推荐训练流程

### Phase 1: 快速验证（100-500局）
- 目标：验证代码正确性
- Epsilon: 1.0 -> 0.1
- 检查对手统计是否正常

### Phase 2: 中等规模（1000-2000局）
- 目标：观察学习趋势
- 调整超参数
- 对比有/无对手特征

### Phase 3: 长时间训练（5000-10000局）
- 目标：达到最优性能
- 使用最佳超参数
- 测试泛化能力

### Phase 4: 测试和部署
- 对阵各种对手组合
- 测试位置打乱的鲁棒性
- 保存最佳模型

## 实验记录模板

```
实验编号: exp_001
日期: 2026-01-02
目标: 验证对手特征有效性

配置:
- Agent: DQN
- Learning Rate: 0.001
- Epsilon Decay: 0.9995
- 对手: [humanlike, simple, simple]
- 训练局数: 2000

结果:
- 最终胜率: 35%
- 对手作弊率收敛值: 18%
- 训练时间: 45分钟

结论:
- 对手特征有效，胜率提升8%
- 需要更长训练时间
- 下一步：增加到5000局

```

## 常见问题

**Q: 训练需要多久？**
A: 1000局约10-20分钟，5000局约1-2小时（取决于硬件）

**Q: 如何判断训练完成？**
A: 当评估胜率连续多次评估不再增长时

**Q: 模型保存在哪里？**
A: `experiments/{experiment_name}/models/`

**Q: 如何恢复训练？**
A: 使用`load_checkpoint()`方法

**Q: 对手特征是否一定有用？**
A: 需要通过对比实验验证，可能需要1000+局才能看到效果
