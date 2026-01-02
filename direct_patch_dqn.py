# direct_patch_dqn.py
#!/usr/bin/env python3
"""
直接修改DQNAgent.py添加最简单的监控
"""

import os
import sys

def add_simple_monitor_to_dqn():
    """给DQNAgent添加最简单的监控"""
    
    dqn_file = "DQNAgent.py"
    
    if not os.path.exists(dqn_file):
        print(f"❌ 找不到文件: {dqn_file}")
        return False
    
    # 备份原文件
    backup_file = f"{dqn_file}.backup_{int(time.time())}"
    os.system(f"copy {dqn_file} {backup_file}")
    print(f"✅ 已备份原文件: {backup_file}")
    
    # 读取文件
    try:
        with open(dqn_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(dqn_file, 'r', encoding='gbk') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return False
    
    # 找到loss行和要插入的位置
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # 找到 loss = self.loss_fn(q_sa, y)
        if 'loss = self.loss_fn(q_sa, y)' in line:
            print(f"✅ 找到loss行在第 {i+1} 行")
            
            # 添加简单监控代码
            monitor_code = '''
            # ====== 最简单监控开始 ======
            try:
                # 每10步打印一次
                if self.step % 10 == 0:
                    print(f"[直接监控] Step {self.step}: Loss={loss.item():.6f}, ε={self.epsilon:.4f}")
                
                # 每100步保存到文件
                if self.step % 100 == 0:
                    import json
                    import os
                    import time
                    
                    # 确保目录存在
                    os.makedirs("direct_monitor_data", exist_ok=True)
                    
                    # 创建或加载数据
                    data_file = "direct_monitor_data/training_data.json"
                    if os.path.exists(data_file):
                        try:
                            with open(data_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                        except:
                            data = {"steps": [], "losses": [], "epsilons": []}
                    else:
                        data = {"steps": [], "losses": [], "epsilons": []}
                    
                    # 添加新数据
                    data["steps"].append(self.step)
                    data["losses"].append(float(loss.item()))
                    data["epsilons"].append(float(self.epsilon))
                    
                    # 保存数据
                    with open(data_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    print(f"[直接监控] 数据保存到: {data_file}")
                    
            except Exception as e:
                # 静默失败，不影响训练
                pass
            # ====== 最简单监控结束 ======
            '''
            
            # 添加监控代码
            new_lines.append(monitor_code)
    
    # 写入新文件
    try:
        with open(dqn_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("✅ DQNAgent.py 已成功添加最简单监控代码！")
        return True
    except Exception as e:
        print(f"❌ 写入文件失败: {e}")
        return False

if __name__ == "__main__":
    import time
    
    print("="*60)
    print("直接修改DQNAgent.py添加最简单监控")
    print("="*60)
    
    if add_simple_monitor_to_dqn():
        print("\n✅ 监控代码添加成功！")
        print("\n现在运行训练:")
        print("1. python rl_trainer.py")
        print("2. 查看控制台输出")
        print("3. 查看生成的 direct_monitor_data/training_data.json")
    else:
        print("\n❌ 添加监控代码失败")
    
    print("\n" + "="*60)