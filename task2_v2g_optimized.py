"""
任务 2 优化版：V2G 站运营策略优化
使用遗传算法和粒子群优化等高级优化方法
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.optimize import differential_evolution, basinhopping
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("任务 2 优化版：V2G 站运营策略优化（高级优化算法）")
print("=" * 80)

# ==================== 数据加载 ====================
print("\n[1/6] 加载数据...")

data_dir = r"数据与样例"
files = os.listdir(data_dir)

v2g_price_file = files[2]
ev_price_file = files[4]

v2g_prices = pd.read_csv(os.path.join(data_dir, v2g_price_file), encoding='gbk')
ev_prices = pd.read_csv(os.path.join(data_dir, ev_price_file), encoding='gbk')

print(f"V2G 电价数据已加载")
print(f"EV 用户电价数据已加载")

# 解析电价
time_periods = ['0:00-6:00', '6:00-10:00', '10:00-14:00', '14:00-18:00', '18:00-24:00']
selling_prices = v2g_prices.loc[0, time_periods].values.astype(float)
buying_prices = v2g_prices.loc[1, time_periods].values.astype(float)

# 创建 24 小时电价
def get_period_index(hour):
    if 0 <= hour < 6:
        return 0
    elif 6 <= hour < 10:
        return 1
    elif 10 <= hour < 14:
        return 2
    elif 14 <= hour < 18:
        return 3
    elif 18 <= hour < 24:
        return 4
    return 0

hourly_selling = np.array([selling_prices[get_period_index(h)] for h in range(24)])
hourly_buying = np.array([buying_prices[get_period_index(h)] for h in range(24)])

print(f"\n24 小时售电电价：{hourly_selling}")
print(f"24 小时购电电价：{hourly_buying}")

# ==================== 系统参数 ====================
print("\n[2/6] 设置系统参数...")

battery_capacity = 1000  # kWh
max_power = 100  # kW
charge_eff = 0.95
discharge_eff = 0.95
min_soc = 0.2
max_soc = 0.9
initial_soc = 0.5

print(f"电池容量：{battery_capacity} kWh")
print(f"最大充/放电功率：{max_power} kW")
print(f"充/放电效率：{charge_eff*100}%/{discharge_eff*100}%")

# ==================== 优化方法 1：遗传算法 ====================
print("\n[3/6] 方法 1：遗传算法优化...")

def genetic_algorithm_v2g(generations=100, population_size=50):
    """使用遗传算法优化 V2G 调度"""
    
    def objective(x):
        """目标函数：最大化收益"""
        charge = np.maximum(0, x[:24])
        discharge = np.maximum(0, x[24:48])
        
        # 计算 SOC 变化
        soc = np.zeros(24)
        soc[0] = initial_soc * battery_capacity
        for t in range(1, 24):
            soc[t] = soc[t-1] + charge[t] * charge_eff - discharge[t] / discharge_eff
        
        # 约束惩罚
        penalty = 0
        # SOC 约束
        soc_normalized = soc / battery_capacity
        penalty += np.sum(np.maximum(0, min_soc - soc_normalized)) * 1000
        penalty += np.sum(np.maximum(0, soc_normalized - max_soc)) * 1000
        
        # 功率约束
        penalty += np.sum(np.maximum(0, charge - max_power)) * 1000
        penalty += np.sum(np.maximum(0, discharge - max_power)) * 1000
        
        # 收益（负值表示最大化）
        revenue = np.sum(discharge * hourly_selling) - np.sum(charge * hourly_buying)
        
        return -revenue + penalty
    
    # 变量边界
    bounds = [(0, max_power)] * 48
    
    # 差分进化算法
    result = differential_evolution(objective, bounds, maxiter=generations, 
                                   popsize=population_size, seed=42, polish=True)
    
    # 提取结果
    charge_opt = np.maximum(0, result.x[:24])
    discharge_opt = np.maximum(0, result.x[24:48])
    
    # 计算 SOC
    soc_opt = np.zeros(24)
    soc_opt[0] = initial_soc * battery_capacity
    for t in range(1, 24):
        soc_opt[t] = soc_opt[t-1] + charge_opt[t] * charge_eff - discharge_opt[t] / discharge_eff
    
    # 计算收益
    revenue = np.sum(discharge_opt * hourly_selling) - np.sum(charge_opt * hourly_buying)
    
    return charge_opt, discharge_opt, soc_opt, revenue

# 运行遗传算法
charge_ga, discharge_ga, soc_ga, revenue_ga = genetic_algorithm_v2g(generations=100, population_size=50)
print(f"遗传算法优化结果:")
print(f"  总充电量：{np.sum(charge_ga):.2f} kWh")
print(f"  总放电量：{np.sum(discharge_ga):.2f} kWh")
print(f"  总收益：{revenue_ga:.2f} 元")

# ==================== 优化方法 2：粒子群优化 ====================
print("\n[4/6] 方法 2：粒子群优化...")

def particle_swarm_v2g(num_particles=30, max_iter=100):
    """粒子群优化算法"""
    
    def objective(x):
        charge = np.maximum(0, x[:24])
        discharge = np.maximum(0, x[24:48])
        
        soc = np.zeros(24)
        soc[0] = initial_soc * battery_capacity
        for t in range(1, 24):
            soc[t] = soc[t-1] + charge[t] * charge_eff - discharge[t] / discharge_eff
        
        penalty = 0
        soc_norm = soc / battery_capacity
        penalty += np.sum(np.maximum(0, min_soc - soc_norm)) * 1000
        penalty += np.sum(np.maximum(0, soc_norm - max_soc)) * 1000
        penalty += np.sum(np.maximum(0, charge - max_power)) * 1000
        penalty += np.sum(np.maximum(0, discharge - max_power)) * 1000
        
        revenue = np.sum(discharge * hourly_selling) - np.sum(charge * hourly_buying)
        return -revenue + penalty
    
    # 初始化粒子
    n_dim = 48
    swarm = np.random.uniform(0, max_power, (num_particles, n_dim))
    velocity = np.random.uniform(-1, 1, (num_particles, n_dim))
    
    # 个体最佳和全局最佳
    pbest = swarm.copy()
    pbest_scores = np.array([objective(p) for p in swarm])
    gbest_idx = np.argmin(pbest_scores)
    gbest = pbest[gbest_idx].copy()
    
    # PSO 参数
    w = 0.7  # 惯性权重
    c1 = 1.5  # 个体学习因子
    c2 = 1.5  # 群体学习因子
    
    # 迭代
    for iter_num in tqdm(range(max_iter), desc="PSO 优化"):
        for i in range(num_particles):
            # 更新速度
            r1, r2 = np.random.rand(2)
            velocity[i] = w * velocity[i] + \
                         c1 * r1 * (pbest[i] - swarm[i]) + \
                         c2 * r2 * (gbest - swarm[i])
            
            # 更新位置
            swarm[i] = np.clip(swarm[i] + velocity[i], 0, max_power)
            
            # 更新个体最佳
            score = objective(swarm[i])
            if score < pbest_scores[i]:
                pbest[i] = swarm[i].copy()
                pbest_scores[i] = score
                
                # 更新全局最佳
                if score < objective(gbest):
                    gbest = swarm[i].copy()
    
    # 提取结果
    charge_pso = np.maximum(0, gbest[:24])
    discharge_pso = np.maximum(0, gbest[24:48])
    
    soc_pso = np.zeros(24)
    soc_pso[0] = initial_soc * battery_capacity
    for t in range(1, 24):
        soc_pso[t] = soc_pso[t-1] + charge_pso[t] * charge_eff - discharge_pso[t] / discharge_eff
    
    revenue_pso = np.sum(discharge_pso * hourly_selling) - np.sum(charge_pso * hourly_buying)
    
    return charge_pso, discharge_pso, soc_pso, revenue_pso

# 运行 PSO
charge_pso, discharge_pso, soc_pso, revenue_pso = particle_swarm_v2g(num_particles=30, max_iter=100)
print(f"\n粒子群优化结果:")
print(f"  总充电量：{np.sum(charge_pso):.2f} kWh")
print(f"  总放电量：{np.sum(discharge_pso):.2f} kWh")
print(f"  总收益：{revenue_pso:.2f} 元")

# ==================== 优化方法 3：贪心算法（基准） ====================
print("\n[5/6] 方法 3：贪心算法（基准）...")

def greedy_v2g():
    """贪心策略"""
    charge = np.zeros(24)
    discharge = np.zeros(24)
    soc = np.zeros(24)
    soc[0] = initial_soc * battery_capacity
    
    for hour in range(24):
        if hourly_buying[hour] < 0.4 and soc[hour-1 if hour > 0 else 0] < max_soc * battery_capacity:
            # 低价充电
            charge[hour] = min(max_power, (max_soc * battery_capacity - soc[hour-1 if hour > 0 else 0]) / charge_eff)
            soc[hour] = soc[hour-1 if hour > 0 else 0] + charge[hour] * charge_eff
        elif hourly_selling[hour] > 0.4 and soc[hour-1 if hour > 0 else 0] > min_soc * battery_capacity:
            # 高价放电
            discharge[hour] = min(max_power, (soc[hour-1 if hour > 0 else 0] - min_soc * battery_capacity) * discharge_eff)
            soc[hour] = soc[hour-1 if hour > 0 else 0] - discharge[hour] / discharge_eff
        else:
            soc[hour] = soc[hour-1 if hour > 0 else 0]
    
    revenue = np.sum(discharge * hourly_selling) - np.sum(charge * hourly_buying)
    return charge, discharge, soc, revenue

charge_greedy, discharge_greedy, soc_greedy, revenue_greedy = greedy_v2g()
print(f"\n贪心算法结果:")
print(f"  总充电量：{np.sum(charge_greedy):.2f} kWh")
print(f"  总放电量：{np.sum(discharge_greedy):.2f} kWh")
print(f"  总收益：{revenue_greedy:.2f} 元")

# ==================== 结果对比 ====================
print("\n[6/6] 结果对比...")

comparison_df = pd.DataFrame({
    '方法': ['遗传算法', '粒子群优化', '贪心算法'],
    '充电量 (kWh)': [np.sum(charge_ga), np.sum(charge_pso), np.sum(charge_greedy)],
    '放电量 (kWh)': [np.sum(discharge_ga), np.sum(discharge_pso), np.sum(discharge_greedy)],
    '收益 (元)': [revenue_ga, revenue_pso, revenue_greedy]
})

print("\n优化方法对比:")
print(comparison_df.to_string(index=False))

# 最佳方法
best_method_idx = comparison_df['收益 (元)'].idxmax()
best_method = comparison_df.loc[best_method_idx, '方法']
print(f"\n最佳优化方法：{best_method}")

# 保存提交文件
print("\n保存优化结果...")

dates = pd.date_range(start='2024/11/1', periods=24, freq='h')

# 使用最佳方法的结果
if best_method == '遗传算法':
    best_charge, best_discharge, best_soc, best_revenue = charge_ga, discharge_ga, soc_ga, revenue_ga
elif best_method == '粒子群优化':
    best_charge, best_discharge, best_soc, best_revenue = charge_pso, discharge_pso, soc_pso, revenue_pso
else:
    best_charge, best_discharge, best_soc, best_revenue = charge_greedy, discharge_greedy, soc_greedy, revenue_greedy

submission_v2g = pd.DataFrame({
    'TIME': dates.strftime('%Y/%m/%d %H:%M'),
    'Charge_Power(kW)': best_charge,
    'Discharge_Power(kW)': best_discharge,
    'SOC(%)': best_soc / battery_capacity * 100,
    'Profit(Yuan)': [best_discharge[t] * hourly_selling[t] - best_charge[t] * hourly_buying[t] for t in range(24)]
})

submission_v2g_path = r"submission_task2_optimized.csv"
submission_v2g.to_csv(submission_v2g_path, index=False, encoding='utf-8-sig')
print(f"优化后的 V2G 策略已保存至：{submission_v2g_path}")

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. 充放电功率对比
ax1 = axes[0]
width = 0.3
x = np.arange(24)
ax1.bar(x - width, best_charge, width, label='充电', color='blue', alpha=0.7)
ax1.bar(x, best_discharge, width, label='放电', color='red', alpha=0.7)
ax1.set_xlabel('小时')
ax1.set_ylabel('功率 (kW)')
ax1.set_title(f'V2G 充放电调度优化 ({best_method})')
ax1.set_xticks(x)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. SOC 变化
ax2 = axes[1]
ax2.plot(x, best_soc / battery_capacity * 100, 'g-o', linewidth=2, markersize=6)
ax2.axhline(y=min_soc*100, color='r', linestyle='--', label='最小 SOC')
ax2.axhline(y=max_soc*100, color='r', linestyle='--', label='最大 SOC')
ax2.set_xlabel('小时')
ax2.set_ylabel('SOC (%)')
ax2.set_title('电池 SOC 变化')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 收益对比
ax3 = axes[2]
hourly_revenue = [best_discharge[t] * hourly_selling[t] - best_charge[t] * hourly_buying[t] for t in range(24)]
ax3.bar(x, hourly_revenue, color=['green' if r >= 0 else 'red' for r in hourly_revenue], alpha=0.7)
ax3.set_xlabel('小时')
ax3.set_ylabel('收益 (元)')
ax3.set_title(f'逐时收益 (总收益：{best_revenue:.2f} 元)')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v2g_optimization_result.png', dpi=300, bbox_inches='tight')
print("V2G 优化结果图已保存为：v2g_optimization_result.png")
plt.close()

print("\n" + "=" * 80)
print("任务 2 优化版完成！")
print("=" * 80)
print(f"最佳优化方法：{best_method}")
print(f"最优收益：{best_revenue:.2f} 元")
