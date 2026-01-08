### **项目报告大纲：构建具有拟人化策略的“骗子酒馆”AI智能体**

#### **1. Abstract (摘要)**
*   **项目目标：** 在不完全信息博弈游戏《Liars Bar》中构建并系统对比启发式、搜索及强化学习（RL）智能体。
*   **核心方法：** 结合 **Humanlike 玩家建模**，利用 **Linear Q-Learning** 和 **DQN** 进行策略学习。
*   **主要贡献：** 引入**环境随机化（Shuffle）**与**心理特征概率编码**，解决了 DQN 在多玩家对局中的顺序过拟合问题，显著提升了模型的泛化能力。

#### **2. Introduction (引言)**
*   **2.1 Motivation (动机)：** 探讨 AI 在非对称信息、高风险决策（如子弹惩罚）下的心理博弈能力。
*   **2.2 Problem Definition (问题定义)：** 描述《Liars Bar》的回合制对抗、虚张声势（Bluffing）和质疑机制。
*   **2.3 Project Goals (项目目标)：** 实现从“机械规则”到“拟人化适应”的智能体演进。

#### **3. Game Framework & Agent Player Design (游戏框架与玩家设计)**
*   **3.1 Game Rules (游戏规则)：** 详述出牌声明、质疑逻辑及基于子弹位置的惩罚机制。
*   **3.2 Baseline Strategies (基准策略设计)：**
    *   **Simple Agent:** 只会质疑且只出真牌。
    *   **Smarter Agent:** 固定概率质疑（不随局势变化），优先使用 Joker。
*   **3.3 Humanlike Modeling (拟人化建模)：**
    *   **适应性与风险意识：** 根据对手的 `cheat/challenge` 倾向及子弹位置调整激进程度。

#### **4. Methods (方法论)**
*   **4.1 Minimax Search (搜索方法)：**
    *   **动作空间简化：** 4 种模板（不出/1真/2真/1假）。
    *   **近似模拟：** 对对手手牌进行近似减量处理。
    *   **局限性分析：** 不完全信息导致状态推演误差累积，计算复杂度随玩家增加而爆炸。
*   **4.2 Heuristic Search (启发式方法)：**
    *   枚举出牌策略并通过 `heuristic()` 函数进行打分决策。
*   **4.3 Reinforcement Learning (强化学习方法)：**
    *   **4.3.1 State & Feature Engineering (特征工程)：** 详细说明 19 维特征向量，涵盖轮次、子弹位置、目标牌充足度及对手心理画像。
    *   **4.3.2 Action Space (动作空间)：** 出牌模板（N 维）+ 质疑动作（2 维）的 One-Hot 编码。
    *   **4.3.3 Reward Design (奖励函数)：** 基于胜负、存活结果的奖励及针对过度保守行为的 **Reward Shaping**。
    *   **4.3.4 LinearQAgent:** 基于线性函数逼近的行为价值学习，具有较好的可解释性。
    *   **4.3.5 Deep Q-Network (DQN):** 采用**经验回放 (Experience Replay)** 打破相关性，利用**目标网络 (Target Network)** 稳定训练。

#### **5. Optimization: Addressing Overfitting (挑战与优化：解决顺序过拟合)**
*   **5.1 Problem Identification:** 发现 DQN 严重依赖固定玩家顺序，交换顺序后胜率从训练时的水平显著下降。
*   **5.2 Proposed Solution (改进方案)：**
    *   **环境随机化：** 每隔一个 epoch 随机打乱玩家顺序（Shuffle）。
    *   **对手画像 (Psychological Profiling)：** 将全局历史信息编码为 `cheat_rate`、`challenge_rate` 等特征并加入特征集。
*   **5.3 Effectiveness:** 实验显示改进后模型从“记顺序”转为“学行为模式”，泛化性显著增强。

#### **6. Interpretability of AI Strategy (玩家策略的可解释性分析)**
*   **6.1 Q-Function Analysis:** 分析模型学到的 Q 函数，观察其在不同“目标牌充足度”或“子弹位置”下的决策倾向。
*   **6.2 Behavioral Pattern Validation:** 
    *   验证 AI 是否学到了“在风险高（子弹接近爆发）时减少说谎”或“对频繁质疑的对手采取更诚实的策略”等模式。
    *   **专家经验对比：** 评估 AI 行为是否与人类专家在博弈中的直觉（如适时的虚张声势）相符。

#### **7. Experiments & Results Analysis (实验与结果分析)**
*   **7.1 Experimental Setup:** 设置不同智能体组合的对位决战。
*   **7.2 Performance Comparison:** 
    *   展示对位决胜率表（如 Heuristic vs MiniAI, DQN vs SimpleAI）。
    *   **训练指标：** 展示不同设置下的胜率收敛曲线及 $\epsilon$-greedy 衰减图。
*   **7.3 Result Analysis:** 讨论为什么 RL 智能体在应对不同策略对手时表现更优。

#### **8. Conclusion**
*   总结项目成果

#### **9. External Resources & References (外部资源与参考文献)**
*   列出 Python 库（PyTorch, NumPy, PyGame）及参考项目。
