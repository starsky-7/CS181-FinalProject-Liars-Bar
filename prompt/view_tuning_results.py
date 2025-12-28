# view_tuning_results.py
#!/usr/bin/env python3
"""
查看超参数调优结果 - 独立文件
"""

import os
import json
import sys

def show_best_config():
    """显示最佳配置"""
    dir_name = "tuning_results"
    
    if not os.path.exists(dir_name):
        print("目录 '{}' 不存在".format(dir_name))
        print("请先运行超参数调优")
        return
    
    # 首先尝试Python文件
    py_file = os.path.join(dir_name, "best_config.py")
    if os.path.exists(py_file):
        print("=" * 60)
        print("最佳配置:")
        print("=" * 60)
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.rstrip()
                    if line and not line.startswith("#"):
                        print(line)
        except:
            print("无法读取文件")
        return
    
    # 然后尝试文本文件
    import glob
    txt_files = glob.glob(os.path.join(dir_name, "best_config_*.txt"))
    
    if txt_files:
        latest = max(txt_files, key=os.path.getctime)
        print("=" * 60)
        print("最佳配置:")
        print("=" * 60)
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                print(f.read())
        except:
            print("无法读取文件")
        return
    
    print("未找到最佳配置文件")

def show_summary():
    """显示结果摘要"""
    dir_name = "tuning_results"
    
    if not os.path.exists(dir_name):
        print("目录 '{}' 不存在".format(dir_name))
        return
    
    # 查找JSON文件
    import glob
    json_files = glob.glob(os.path.join(dir_name, "results_*.json"))
    
    if not json_files:
        print("未找到结果文件")
        return
    
    # 读取最新文件
    latest = max(json_files, key=os.path.getctime)
    
    try:
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        print("无法读取文件 {}".format(latest))
        return
    
    print("=" * 60)
    print("调优结果摘要:")
    print("=" * 60)
    
    print("总试验数: {}".format(len(data.get('all_results', []))))
    print("最佳胜率: {:.3f}".format(data.get('best_score', 0)))
    
    summary = data.get('summary', {})
    if summary:
        avg = summary.get('average_score', 0)
        std = summary.get('std_score', 0)
        print("平均胜率: {:.3f}".format(avg))
        print("胜率标准差: {:.3f}".format(std))
    
    # 显示前3名
    results = data.get('all_results', [])
    if results and len(results) > 0:
        print("\n前3名配置:")
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        
        for i in range(min(3, len(sorted_results))):
            result = sorted_results[i]
            score = result.get('score', 0)
            print("\n{}. 胜率: {:.3f}".format(i+1, score))
            
            config = result.get('config', {})
            keys_to_show = ["learning_rate", "discount_factor", "epsilon_decay", "batch_size"]
            
            for key in keys_to_show:
                if key in config:
                    value = config[key]
                    if isinstance(value, float):
                        print("   {:20}: {:.6f}".format(key, value))
                    else:
                        print("   {:20}: {}".format(key, value))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='查看超参数调优结果')
    parser.add_argument('--best', action='store_true', help='查看最佳配置')
    parser.add_argument('--summary', action='store_true', help='查看结果摘要')
    parser.add_argument('--simple', action='store_true', help='简单模式')
    
    args = parser.parse_args()
    
    if args.simple:
        # 简单模式：直接显示最佳配置
        show_best_config()
    elif args.best:
        show_best_config()
    elif args.summary:
        show_summary()
    else:
        # 默认显示帮助
        print("超参数调优结果查看器")
        print("=" * 60)
        print("使用方法:")
        print("  python view_tuning_results.py --best    查看最佳配置")
        print("  python view_tuning_results.py --summary 查看结果摘要")
        print("  python view_tuning_results.py --simple  简单查看")
        print("=" * 60)

if __name__ == "__main__":
    main()