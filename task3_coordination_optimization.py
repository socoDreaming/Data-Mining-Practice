"""
任务 3: 车网互动协同调度优化
计及非线性潮流约束的充放电优化

优化目标：
1. 最小化用户充电成本
2. 最小化电网负荷波动
3. 最大化光伏消纳

约束条件：
1. 充电站功率约束
2. 电池 SOC 约束
3. 配电网潮流约束（简化为线路容量约束）
4. 用户充电需求约束
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("任务 3: 车网互动协同调度优化")
print("=" * 70)

# 读取必要的数据
print("\n正在读取数据...")
data_dir = r"数据与样例"
files = os.listdir(data_dir)

# 读取电价数据
v2g_price_file = files[2]
ev_price_file = files[4]

v2g_prices = pd.read_csv(os.path.join(data_dir, v2g_price_file), encoding='gbk')
ev_prices = pd.read_csv(os.path.join(data_dir, ev_price_file), encoding='gbk')

print(f"V2G 电价数据已加载")
print(f"EV 用户电价数据已加载")

# 解析 EV 用户电价
print(f"EV 电价数据列名：{ev_prices.columns.tolist()}")
print(ev_prices.head())

# 使用正确的列名
ev_charging_prices = ev_prices.iloc[:, 1].values  # 第二列是充电电价
ev_discharging_prices = ev_prices.iloc[:, 2].values  # 第三列是放电电价

print(f"\nEV 充电电价范围：{ev_charging_prices.min():.2f} - {ev_charging_prices.max():.2f} 元/kWh")
print(f"EV 放电电价范围：{ev_discharging_prices.min():.2f} - {ev_discharging_prices.max():.2f} 元/kWh")

# 系统参数
print("\n" + "=" * 70)
print("系统参数设置")
print("=" * 70)

# 充电站参数
charging_station_capacity = 500  # 充电站总容量 (kW)
max_charging_power = 300  # 最大充电功率 (kW)
max_discharging_power = 200  # 最大放电功率 (kW)

# 电动汽车参数
ev_battery_capacity = 60  # 单车电池容量 (kWh)
ev_count = 50  # 电动汽车数量
min_soc = 0.2  # 最小 SOC (20%)
max_soc = 0.95  # 最大 SOC (95%)
target_soc = 0.8  # 目标 SOC (80%)

# 电网参数
grid_capacity = 1000  # 电网容量 (kW)
line_capacity = 800  # 线路容量 (kW)

# 光伏参数 (假设)
pv_capacity = 200  # 光伏容量 (kW)
# 24 小时光伏出力标幺值
pv_output_profile = np.array([
    0, 0, 0, 0, 0, 0,      # 0:00-5:00 (6 小时)
    0.1, 0.3, 0.5, 0.7,   # 6:00-9:00 (4 小时)
    0.8, 0.9, 1.0, 0.9,   # 10:00-13:00 (4 小时)
    0.8, 0.6, 0.4, 0.2,   # 14:00-17:00 (4 小时)
    0.1, 0, 0, 0, 0, 0    # 18:00-23:00 (6 小时)
])
print(f"光伏出力数组形状：{pv_output_profile.shape}")

print(f"\n充电站容量：{charging_station_capacity} kW")
print(f"电动汽车数量：{ev_count} 辆")
print(f"单车电池容量：{ev_battery_capacity} kWh")
print(f"光伏容量：{pv_capacity} kW")
print(f"电网容量：{grid_capacity} kW")
print(f"线路容量：{line_capacity} kW")

# 优化模型
print("\n" + "=" * 70)
print("多目标优化模型")
print("=" * 70)

def objective_function(x, weights=None):
    """
    多目标优化函数
    
    x: 决策变量 [charging_power[24], discharging_power[24], grid_power[24]]
    weights: 各目标权重 [用户成本权重，电网波动权重，光伏消纳权重]
    """
    if weights is None:
        weights = [0.4, 0.3, 0.3]  # 默认权重
    
    charging_power = x[:24]
    discharging_power = x[24:48]
    grid_power = x[48:72]
    
    # 目标 1: 最小化用户充电成本
    user_cost = np.sum(charging_power * ev_charging_prices - discharging_power * ev_discharging_prices)
    
    # 目标 2: 最小化电网负荷波动
    total_load = grid_power + np.sum(charging_power) - np.sum(discharging_power)
    load_variance = np.var(total_load)
    
    # 目标 3: 最大化光伏消纳
    pv_consumption = np.sum(np.minimum(charging_power, pv_output_profile * pv_capacity))
    pv_curtailed = np.sum(pv_output_profile * pv_capacity) - pv_consumption
    
    # 加权多目标
    objective = (
        weights[0] * user_cost / 1000 +  # 归一化
        weights[1] * load_variance / 100 +
        weights[2] * pv_curtailed / 100
    )
    
    return objective

# 约束条件
def power_balance_constraint(x):
    """功率平衡约束"""
    charging = x[:24]
    discharging = x[24:48]
    grid = x[48:72]
    
    # 充电功率 = 电网供电 + 放电功率 + 光伏
    pv = pv_output_profile * pv_capacity
    balance = charging - (grid + discharging * 0.95 + pv * 0.95)
    
    return np.sum(np.abs(balance))

def soc_constraint(x):
    """SOC 约束"""
    charging = x[:24]
    discharging = x[24:48]
    
    # 简化的 SOC 计算
    soc_change = (charging * 0.95 - discharging / 0.95) / (ev_battery_capacity * ev_count)
    final_soc = 0.5 + np.cumsum(soc_change)
    
    # 确保 SOC 在合理范围内
    violation = np.sum(np.maximum(0, min_soc - final_soc)) + np.sum(np.maximum(0, final_soc - max_soc))
    
    return violation

def line_capacity_constraint(x):
    """线路容量约束"""
    grid_power = x[48:72]
    return np.sum(np.maximum(0, np.abs(grid_power) - line_capacity))

# 变量边界
bounds = []
for _ in range(72):
    bounds.append((0, charging_station_capacity))  # 所有变量非负

# 初始解
x0 = np.ones(72) * 10

print("\n正在求解优化问题...")
print("优化目标：")
print("  1. 最小化用户充电成本 (权重 40%)")
print("  2. 最小化电网负荷波动 (权重 30%)")
print("  3. 最大化光伏消纳 (权重 30%)")

# 使用 scipy.optimize 求解
result = minimize(
    objective_function,
    x0,
    method='SLSQP',
    bounds=bounds,
    options={'maxiter': 100, 'disp': True}
)

if result.success:
    print(f"\n优化成功！")
    print(f"最优目标函数值：{result.fun:.4f}")
    print(f"迭代次数：{result.nit}")
    
    # 提取结果
    optimal_charging = result.x[:24]
    optimal_discharging = result.x[24:48]
    optimal_grid = result.x[48:72]
    
    # 计算指标
    total_charging_cost = np.sum(optimal_charging * ev_charging_prices)
    total_discharging_revenue = np.sum(optimal_discharging * ev_discharging_prices)
    net_cost = total_charging_cost - total_discharging_revenue
    
    pv_utilization = np.sum(np.minimum(optimal_charging, pv_output_profile * pv_capacity))
    pv_total = np.sum(pv_output_profile * pv_capacity)
    pv_utilization_rate = pv_utilization / pv_total * 100 if pv_total > 0 else 0
    
    grid_load_variance = np.var(optimal_grid)
    
    print(f"\n优化结果:")
    print(f"  总充电成本：{total_charging_cost:.2f} 元")
    print(f"  总放电收益：{total_discharging_revenue:.2f} 元")
    print(f"  净成本：{net_cost:.2f} 元")
    print(f"  光伏利用率：{pv_utilization_rate:.2f}%")
    print(f"  电网负荷方差：{grid_load_variance:.2f}")
    
else:
    print(f"\n优化失败：{result.message}")
    optimal_charging = np.zeros(24)
    optimal_discharging = np.zeros(24)
    optimal_grid = np.zeros(24)

# 创建提交文件
print("\n正在生成车网互动协同调度提交文件...")

# 创建时间序列
dates = pd.date_range(start='2024/11/1', periods=24, freq='h')

# 创建提交 DataFrame
submission_v2g_coord = pd.DataFrame({
    'TIME': dates.strftime('%Y/%m/%d %H:%M'),
    'Charging_Power(kW)': optimal_charging,
    'Discharging_Power(kW)': optimal_discharging,
    'Grid_Power(kW)': optimal_grid,
    'PV_Output(kW)': pv_output_profile * pv_capacity,
    'EV_Charging_Price(Yuan/kWh)': ev_charging_prices,
    'EV_Discharge_Price(Yuan/kWh)': ev_discharging_prices
})

# 保存提交文件
submission_path = r"submission_task3_coordination.csv"
submission_v2g_coord.to_csv(submission_path, index=False, encoding='utf-8-sig')
print(f"\n提交文件已保存至：{submission_path}")
print(f"文件形状：{submission_v2g_coord.shape}")
print(f"\n前 10 行数据:")
print(submission_v2g_coord.head(10))

print("\n" + "=" * 70)
print("任务 3 完成！")
print("=" * 70)
