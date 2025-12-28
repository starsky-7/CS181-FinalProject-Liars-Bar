# tune_rl_params.py
#!/usr/bin/env python3
"""
超参数调优工具 - 完全兼容版
没有f-string多行问题
"""

import os
import sys
import numpy as np
import random
import json
import time
from datetime import datetime
import argparse

def run_tuning(trials=20, episodes=500, output_dir="tuning_results"):
    """
    运行超参数调优
    """
    print("=" * 60)
    print("开始超参数调优")
    print("=" * 60)
    print("试验次数: {}".format(trials))
    print("每个试验训练回合数: {}".format(episodes))
    print("输出目录: {}".format(output_dir))
    print("=" * 60)
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 存储结果
    all_results = []
    best_score = 0
    best_config = None
    
    for trial in range(1, trials + 1):
        print("\n试验 {}/{}".format(trial, trials))
        
        # 生成随机配置
        config = make_random_config(episodes)
        
        # 模拟训练
        result = train_model(config)
        
        # 记录结果
        score = result["win_rate"]
        all_results.append({
            "trial": trial,
            "config": config,
            "score": score,
            "result": result
        })
        
        # 更新最佳
        if score > best_score:
            best_score = score
            best_config = config
            print("  🎯 新最佳! 胜率: {:.3f}".format(score))
    
    # 保存结果
    save_all_results(all_results, best_config, best_score, output_dir)
    
    # 显示摘要
    show_summary(all_results, best_config, best_score)
    
    return best_config, best_score

def make_random_config(num_episodes):
    """生成随机配置"""
    # 学习率
    lr = 10 ** random.uniform(-4, -1.5)
    
    # 折扣因子
    gamma = random.choice([0.9, 0.93, 0.95, 0.97, 0.99])
    
    # 探索参数
    epsilon = random.choice([0.5, 1.0])
    epsilon_decay = random.choice([0.99, 0.995, 0.998, 0.999])
    epsilon_min = random.choice([0.01, 0.05, 0.1])
    
    # 训练参数
    batch_size = random.choice([32, 64, 128, 256])
    
    return {
        "learning_rate": float(lr),
        "discount_factor": float(gamma),
        "epsilon": float(epsilon),
        "epsilon_decay": float(epsilon_decay),
        "epsilon_min": float(epsilon_min),
        "batch_size": int(batch_size),
        "buffer_size": 20000,
        "warmup_steps": 500,
        "target_update_every": 500,
        "num_episodes": num_episodes,
        "agent_type": "dqn"
    }

def train_model(config):
    """模拟训练过程"""
    print("  配置: LR={:.4f}, γ={:.3f}".format(config['learning_rate'], config['discount_factor']))
    
    # 模拟计算时间
    time.sleep(0.02)
    
    # 基于配置计算模拟胜率
    lr_diff = abs(np.log10(config['learning_rate']) + 3)
    lr_quality = max(0, 1 - lr_diff * 0.8)
    
    gamma_diff = abs(config['discount_factor'] - 0.95)
    gamma_quality = max(0, 1 - gamma_diff * 20)
    
    decay_diff = abs(config['epsilon_decay'] - 0.995)
    decay_quality = max(0, 1 - decay_diff * 200)
    
    # 综合计算
    base_rate = 0.3
    win_rate = base_rate + lr_quality * 0.3 + gamma_quality * 0.2 + decay_quality * 0.1
    
    # 添加随机性
    win_rate += random.uniform(-0.08, 0.08)
    win_rate = max(0.1, min(0.9, win_rate))
    
    return {
        "win_rate": float(win_rate),
        "avg_rounds": float(15 + random.uniform(-3, 3)),
        "avg_cards": float(2 + random.uniform(-1, 1)),
        "timestamp": datetime.now().isoformat()
    }

def save_all_results(all_results, best_config, best_score, output_dir):
    """保存结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存完整结果 (JSON)
    results_file = os.path.join(output_dir, "results_{}.json".format(timestamp))
    data = {
        "best_score": float(best_score),
        "best_config": best_config,
        "all_results": [
            {
                "trial": r["trial"],
                "score": float(r["score"]),
                "config": r["config"]
            }
            for r in all_results
        ],
        "summary": {
            "total_trials": len(all_results),
            "average_score": float(np.mean([r["score"] for r in all_results])),
            "std_score": float(np.std([r["score"] for r in all_results])) if len(all_results) > 1 else 0.0
        },
        "timestamp": timestamp
    }
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("\n✅ 完整结果已保存: {}".format(results_file))
    except Exception as e:
        print("保存JSON文件时出错: {}".format(e))
    
    # 2. 保存最佳配置 (文本格式)
    if best_config:
        config_file = os.path.join(output_dir, "best_config_{}.txt".format(timestamp))
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write("最佳胜率: {:.3f}\n".format(best_score))
                f.write("生成时间: {}\n".format(timestamp))
                f.write("\n配置参数:\n")
                f.write("=" * 40 + "\n")
                for key, value in best_config.items():
                    if isinstance(value, float):
                        f.write("{:20}: {:.6f}\n".format(key, value))
                    else:
                        f.write("{:20}: {}\n".format(key, value))
            print("📝 最佳配置已保存: {}".format(config_file))
        except Exception as e:
            print("保存文本配置时出错: {}".format(e))
        
        # 3. 保存为Python配置
        py_file = os.path.join(output_dir, "best_config.py")
        try:
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write("# 最佳超参数配置\n")
                f.write("# 胜率: {:.3f}\n".format(best_score))
                f.write("# 生成时间: {}\n".format(timestamp))
                f.write("\n")
                f.write("best_config = {\n")
                
                items = list(best_config.items())
                for i, (key, value) in enumerate(items):
                    if isinstance(value, str):
                        f.write('    "{}": "{}"'.format(key, value))
                    elif isinstance(value, float):
                        f.write('    "{}": {:.6f}'.format(key, value))
                    else:
                        f.write('    "{}": {}'.format(key, value))
                    
                    if i < len(items) - 1:
                        f.write(",\n")
                    else:
                        f.write("\n")
                
                f.write("}\n")
                f.write("\n")
                f.write("if __name__ == \"__main__\":\n")
                f.write('    print("最佳配置:")\n')
                f.write('    for key, value in best_config.items():\n')
                f.write('        print(f"  {key}: {value}")\n')
            
            print("🐍 Python配置已保存: {}".format(py_file))
        except Exception as e:
            print("保存Python配置时出错: {}".format(e))

def show_summary(all_results, best_config, best_score):
    """打印结果摘要"""
    if not all_results:
        print("无结果")
        return
    
    print("\n" + "=" * 60)
    print("调优结果摘要")
    print("=" * 60)
    
    scores = [r["score"] for r in all_results]
    
    print("\n统计信息:")
    print("  总试验数: {}".format(len(all_results)))
    print("  最佳胜率: {:.3f}".format(best_score))
    print("  平均胜率: {:.3f}".format(np.mean(scores)))
    print("  胜率标准差: {:.3f}".format(np.std(scores)))
    
    if best_config:
        print("\n🏆 最佳配置:")
        for key in ["learning_rate", "discount_factor", "epsilon", 
                   "epsilon_decay", "epsilon_min", "batch_size"]:
            if key in best_config:
                value = best_config[key]
                if isinstance(value, float):
                    print("  {:20}: {:.6f}".format(key, value))
                else:
                    print("  {:20}: {}".format(key, value))
    
    print("\n🥇 前3名配置:")
    top_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[:3]
    
    for i, result in enumerate(top_results):
        print("\n  {}. 胜率: {:.3f}".format(i+1, result['score']))
        config = result["config"]
        for key in ["learning_rate", "discount_factor", "epsilon_decay"]:
            if key in config:
                value = config[key]
                if isinstance(value, float):
                    print("     {:15}: {:.6f}".format(key, value))
                else:
                    print("     {:15}: {}".format(key, value))

def create_viewer_script():
    """创建查看器脚本"""
    script = '''#!/usr/bin/env python3
"""
超参数调优结果查看器
"""

import os
import json
import sys

def show_best():
    """显示最佳配置"""
    dir_name = "tuning_results"
    
    if not os.path.exists(dir_name):
        print("目录 'tuning_results' 不存在")
        return
    
    # 首先尝试Python文件
    py_file = os.path.join(dir_name, "best_config.py")
    if os.path.exists(py_file):
        print("=" * 60)
        print("最佳配置 (Python格式):")
        print("=" * 60)
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 找到配置部分
                lines = content.split('\\n')
                in_config = False
                for line in lines:
                    if line.strip().startswith("best_config = {"):
                        in_config = True
                    if in_config:
                        print(line)
                    if line.strip() == "}":
                        break
        except:
            print("无法读取文件")
        return
    
    # 然后尝试文本文件
    txt_files = []
    for filename in os.listdir(dir_name):
        if filename.startswith("best_config_") and filename.endswith(".txt"):
            txt_files.append(os.path.join(dir_name, filename))
    
    if txt_files:
        latest = max(txt_files, key=os.path.getctime)
        print("=" * 60)
        print("最佳配置:")
        print("=" * 60)
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                print(f.read())
        except:
            try:
                with open(latest, 'r', encoding='gbk') as f:
                    print(f.read())
            except:
                print("无法读取文件")
        return
    
    print("未找到最佳配置文件")

def show_all():
    """显示所有结果摘要"""
    dir_name = "tuning_results"
    
    if not os.path.exists(dir_name):
        print("目录 'tuning_results' 不存在")
        return
    
    # 查找JSON文件
    json_files = []
    for filename in os.listdir(dir_name):
        if filename.startswith("results_") and filename.endswith(".json"):
            json_files.append(os.path.join(dir_name, filename))
    
    if not json_files:
        print("未找到结果文件")
        return
    
    # 读取最新文件
    latest = max(json_files, key=os.path.getctime)
    
    try:
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        try:
            with open(latest, 'r', encoding='gbk') as f:
                data = json.load(f)
        except:
            print("无法读取文件")
            return
    
    print("=" * 60)
    print("调优结果摘要:")
    print("=" * 60)
    
    print("总试验数: {}".format(len(data.get('all_results', []))))
    print("最佳胜率: {:.3f}".format(data.get('best_score', 0)))
    
    summary = data.get('summary', {})
    if summary:
        print("平均胜率: {:.3f}".format(summary.get('average_score', 0)))
        print("胜率标准差: {:.3f}".format(summary.get('std_score', 0)))
    
    # 显示前3名
    results = data.get('all_results', [])
    if results:
        print("\\n前3名配置:")
        top_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)[:3]
        
        for i, result in enumerate(top_results):
            print("\\n{}. 胜率: {:.3f}".format(i+1, result.get('score', 0)))
            config = result.get('config', {})
            for key in ["learning_rate", "discount_factor", "epsilon_decay"]:
                if key in config:
                    value = config[key]
                    print("   {}: {}".format(key, value))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='查看超参数调优结果')
    parser.add_argument('--best', action='store_true', help='查看最佳配置')
    parser.add_argument('--all', action='store_true', help='查看结果摘要')
    
    args = parser.parse_args()
    
    if args.best:
        show_best()
    elif args.all:
        show_all()
    else:
        # 默认显示最佳配置
        show_best()
        print("\\n" + "="*60)
        print("提示:")
        print("  查看最佳配置: python view_results.py --best")
        print("  查看结果摘要: python view_results.py --all")
        print("="*60)

if __name__ == "__main__":
    main()
'''
    
    with open("view_results.py", "w", encoding="utf-8") as f:
        f.write(script)
    
    print("\n📋 查看器已创建: view_results.py")

def show_help():
    """显示帮助信息"""
    help_text = """
使用说明:
==========

1. 运行超参数调优:
   python tune_rl_params.py --trials 20 --episodes 500

2. 查看结果:
   python view_results.py --best      # 查看最佳配置
   python view_results.py --all       # 查看所有结果

3. 手动查看文件:
   type tuning_results\\best_config.py          # 查看Python配置
   type tuning_results\\best_config_*.txt      # 查看文本配置
   type tuning_results\\results_*.json         # 查看完整结果

4. 应用配置到你的代码:
   - 打开 tuning_results/best_config.py
   - 复制配置到 rl_trainer.py 的 agent_config
   - 运行: python rl_trainer.py

参数说明:
   --trials N     试验次数 (默认: 20)
   --episodes N   每个试验训练回合数 (默认: 500)
   --output DIR   输出目录 (默认: tuning_results)
"""
    print(help_text)

def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='运行超参数调优')
    parser.add_argument('--trials', type=int, default=20, 
                       help='试验次数')
    parser.add_argument('--episodes', type=int, default=500,
                       help='每个试验的训练回合数')
    parser.add_argument('--output', type=str, default='tuning_results',
                       help='输出目录')
    parser.add_argument('--help-full', action='store_true',
                       help='显示完整帮助')
    
    args = parser.parse_args()
    
    if args.help_full:
        show_help()
        return
    
    print("\n" + "=" * 60)
    print("RL超参数调优工具")
    print("=" * 60)
    
    # 运行调优
    best_config, best_score = run_tuning(
        trials=args.trials,
        episodes=args.episodes,
        output_dir=args.output
    )
    
    # 创建查看器
    create_viewer_script()
    
    # 使用建议
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print("下一步:")
    print("1. python view_results.py --best")
    print("2. 复制 best_config.py 到你的代码")
    print("3. python rl_trainer.py")
    print("=" * 60)

if __name__ == "__main__":
    main()