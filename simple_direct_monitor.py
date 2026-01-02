# simple_direct_monitor.py
#!/usr/bin/env python3
"""
简单直接的DQN监控 - 绝对能工作
"""

import json
import os
import time
import numpy as np

class SimpleDirectMonitor:
    """最简单的直接监控"""
    
    def __init__(self):
        print("[简单监控] 初始化...")
        self.data = {
            'steps': [],
            'losses': [],
            'epsilons': [],
            'timestamps': []
        }
        self.data_dir = "simple_monitor_data"
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"[简单监控] 数据目录: {self.data_dir}")
    
    def record(self, step, loss, epsilon):
        """记录最简单必要的数据"""
        try:
            # 记录数据
            self.data['steps'].append(int(step))
            self.data['losses'].append(float(loss.item() if hasattr(loss, 'item') else loss))
            self.data['epsilons'].append(float(epsilon))
            self.data['timestamps'].append(time.time())
            
            # 每10步打印一次
            if step % 10 == 0:
                print(f"[简单监控] Step {step}: Loss={loss.item():.6f}, ε={epsilon:.4f}")
            
            # 每50步保存一次
            if step % 50 == 0:
                self._save_data(step)
                
        except Exception as e:
            print(f"[简单监控错误] 记录失败: {e}")
    
    def _save_data(self, step):
        """保存数据"""
        try:
            filename = os.path.join(self.data_dir, f"step_{step}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'step': step,
                    'data': self.data,
                    'data_points': len(self.data['steps'])
                }, f, indent=2, ensure_ascii=False)
            
            print(f"[简单监控] 数据保存: {filename} ({len(self.data['steps'])} 点)")
            
        except Exception as e:
            print(f"[简单监控错误] 保存失败: {e}")

# 创建全局实例
simple_monitor = SimpleDirectMonitor()