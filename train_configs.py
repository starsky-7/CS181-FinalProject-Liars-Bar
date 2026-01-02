"""
快速启动不同训练配置的脚本
"""

from advanced_trainer import AdvancedRLTrainer


def quick_test():
    """快速测试（100局）"""
    print("🚀 快速测试配置")
    
    config = {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.1,
    }
    
    trainer = AdvancedRLTrainer("dqn", config, "quick_test")
    trainer.train(
        num_episodes=100,
        opponent_types=["simple", "simple", "simple"],
        eval_interval=20,
        eval_games=10,
        save_interval=50,
        plot_interval=20,
        shuffle_positions=True  # 启用位置打乱
    )


def medium_training():
    """中等规模训练（1000局）"""
    print("🎯 中等规模训练")
    
    config = {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.998,
        "epsilon_min": 0.05,
    }
    
    trainer = AdvancedRLTrainer("dqn", config, "medium_training")
    trainer.train(
        num_episodes=1000,
        opponent_types=["humanlike", "simple", "simple"],
        eval_interval=50,
        eval_games=20,
        save_interval=200,
        plot_interval=50,
        shuffle_positions=True  # 启用位置打乱
    )


def long_training():
    """长时间训练（5000局）"""
    print("🔥 长时间训练")
    
    config = {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.05,
        "batch_size": 64,
        "warmup_steps": 500,
    }
    
    trainer = AdvancedRLTrainer("dqn", config, "long_training")
    trainer.train(
        num_episodes=5000,
        opponent_types=["humanlike", "simple", "simple"],
        eval_interval=100,
        eval_games=30,
        save_interval=500,
        plot_interval=100,
        shuffle_positions=True  # 启用位置打乱
    )


def baseline_comparison():
    """基线对比：不使用对手特征"""
    print("📊 基线对比实验")
    
    from rl_trainer import RLTrainer
    
    config = {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.998,
        "epsilon_min": 0.05,
    }
    
    # 注意：使用普通RLTrainer（带对手特征）
    trainer = RLTrainer("dqn", config)
    
    print("\n训练1000局...")
    trainer.train(num_episodes=1000, opponent_types=["humanlike", "simple", "simple"])
    
    print("\n评估...")
    results = trainer.evaluate(num_games=100, opponent_types=["humanlike", "simple", "simple"])
    print(f"最终胜率: {results['win_rate']:.2%}")


def advanced_config():
    """高级配置：最优超参数"""
    print("⚙️ 高级配置训练")
    
    config = {
        "learning_rate": 0.0005,      # 较小的学习率
        "discount_factor": 0.97,       # 更重视长期奖励
        "epsilon": 1.0,
        "epsilon_decay": 0.9997,       # 非常慢的衰减
        "epsilon_min": 0.03,           # 保持3%探索
        "batch_size": 128,             # 更大的batch
        "warmup_steps": 1000,          # 更多warmup
        "buffer_size": 30000,          # 更大的buffer
    }
    
    trainer = AdvancedRLTrainer("dqn", config, "advanced_config")
    trainer.train(
        num_episodes=10000,
        opponent_types=["humanlike", "humanlike", "simple"],  # 更强的对手
        eval_interval=200,
        eval_games=50,
        save_interval=1000,
        plot_interval=200,
        shuffle_positions=True  # 启用位置打乱
    )


def position_robustness_test():
    """位置鲁棒性对比实验"""
    print("🔄 位置鲁棒性测试")
    
    config = {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.998,
        "epsilon_min": 0.05,
    }
    
    # 实验1：不打乱位置
    print("\n实验1: 固定位置训练...")
    trainer_fixed = AdvancedRLTrainer("dqn", config, "position_test_fixed")
    trainer_fixed.train(
        num_episodes=400,
        opponent_types=["humanlike", "smarter", "smarter"],
        eval_interval=50,
        eval_games=20,
        save_interval=500,
        plot_interval=50,
        shuffle_positions=False  # 关键：不打乱
    )
    
    # 实验2：打乱位置
    print("\n实验2: 随机位置训练...")
    trainer_shuffled = AdvancedRLTrainer("dqn", config, "position_test_shuffled")
    trainer_shuffled.train(
        num_episodes=400,
        opponent_types=["humanlike", "smarter", "smarter"],
        eval_interval=50,
        eval_games=20,
        save_interval=500,
        plot_interval=50,
        shuffle_positions=True  # 关键：打乱
    )
    
    print("\n对比结果：")
    print("  查看 experiments/position_test_fixed/plots/")
    print("  查看 experiments/position_test_shuffled/plots/")
    print("  预期：打乱位置的模型在不同位置下表现更稳定")


if __name__ == "__main__":
    import sys
    
    configs = {
        "quick": quick_test,
        "medium": medium_training,
        "long": long_training,
        "baseline": baseline_comparison,
        "advanced": advanced_config,
        "position": position_robustness_test,  # 新增
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in configs:
        configs[sys.argv[1]]()
    else:
        print("\n可用的训练配置:")
        print("  python train_configs.py quick     - 快速测试（100局）")
        print("  python train_configs.py medium    - 中等规模（1000局）")
        print("  python train_configs.py long      - 长时间训练（5000局）")
        print("  python train_configs.py baseline  - 基线对比")
        print("  python train_configs.py advanced  - 高级配置（10000局）")
        print("  python train_configs.py position  - 位置鲁棒性对比实验")
        print("\n默认运行中等规模训练...")
        print("  python train_configs.py advanced  - 高级配置（10000局）")
        print("\n默认运行中等规模训练...")
        medium_training()
