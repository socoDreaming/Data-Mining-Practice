"""
任务 2: V2G 站运营策略优化
基于动态购售电价的最优能量调度

优化目标：最大化 V2G 站的收益
收益 = 售电收入 - 购电成本

策略：
1. 在电价低的时段（如 0:00-6:00）从电网购电存储
2. 在电价高的时段（如 18:00-24:00）向电网售电
3. 考虑充放电效率约束
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("任务 2: V2G 站运营策略优化")
print("=" * 60)

# 读取 V2G 电价数据
print("\n正在读取电价数据...")
data_dir = r"数据与样例"
files = os.listdir(data_dir)

print(f"数据目录文件列表：{files}")

# 找到对应的文件 - 使用文件名匹配
v2g_price_file = files[2]  # 附件 1-V2G 站向电网售电及从电网购电电价.csv
ev_price_file = files[4]   # 附件 3 -EV 用户充放电电价.csv

v2g_price_path = os.path.join(data_dir, v2g_price_file)
ev_price_path = os.path.join(data_dir, ev_price_file)

print(f"V2G 电价文件：{v2g_price_path}")
print(f"EV 电价文件：{ev_price_path}")

v2g_prices = pd.read_csv(v2g_price_path, encoding='gbk')
ev_prices = pd.read_csv(ev_price_path, encoding='gbk')

print(f"V2G 电价数据:\n{v2g_prices}")
print(f"\nEV 用户电价数据前 5 行:\n{ev_prices.head()}")

# 解析 V2G 电价
time_periods = ['0:00-6:00', '6:00-10:00', '10:00-14:00', '14:00-18:00', '18:00-24:00']
selling_prices = v2g_prices.loc[0, time_periods].values.astype(float)  # 售电电价
buying_prices = v2g_prices.loc[1, time_periods].values.astype(float)   # 购电电价

print(f"\n售电电价 (元/kWh): {selling_prices}")
print(f"购电电价 (元/kWh): {buying_prices}")

# 创建小时到时段索引的映射
def get_period_index(hour):
    """根据小时获取时段索引"""
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
    else:
        return 0

# 创建 24 小时的电价表
hourly_selling_prices = np.array([selling_prices[get_period_index(h)] for h in range(24)])
hourly_buying_prices = np.array([buying_prices[get_period_index(h)] for h in range(24)])

print(f"\n24 小时售电电价：{hourly_selling_prices}")
print(f"24 小时购电电价：{hourly_buying_prices}")

# 计算电价差值（利润空间）
price_spread = hourly_selling_prices - hourly_buying_prices
print(f"\n电价差值 (售电 - 购电): {price_spread}")
print(f"正差值表示盈利时段，负差值表示亏损时段")

# V2G 运营策略优化
print("\n" + "=" * 60)
print("V2G 站运营策略优化")
print("=" * 60)

# 假设 V2G 站参数
battery_capacity = 1000  # 电池容量 (kWh)
max_charge_power = 100   # 最大充电功率 (kW)
max_discharge_power = 100  # 最大放电功率 (kW)
charge_efficiency = 0.95   # 充电效率
discharge_efficiency = 0.95  # 放电效率
initial_soc = 0.5  # 初始 SOC (50%)

print(f"\nV2G 站参数:")
print(f"电池容量：{battery_capacity} kWh")
print(f"最大充/放电功率：{max_charge_power}/{max_discharge_power} kW")
print(f"充/放电效率：{charge_efficiency*100}%/{discharge_efficiency*100}%")
print(f"初始 SOC: {initial_soc*100}%")

# 优化策略：基于电价差值的贪心算法
def optimize_v2g_operation(prices, battery_cap, max_p, efficiency, initial_soc, hours=24):
    """
    V2G 运营优化
    
    策略:
    1. 在购电电价低的时段充电
    2. 在售电电价高的时段放电
    3. 考虑电价差值，确保盈利
    """
    soc = initial_soc * battery_cap  # 当前电量 (kWh)
    charge_schedule = []  # 充电计划
    discharge_schedule = []  # 放电计划
    profit = []  # 每小时收益
    soc_history = []  # SOC 历史
    
    # 按购电电价排序，找出最便宜的时段充电
    buy_price_order = np.argsort(prices[:, 1])  # 购电电价从低到高
    # 按售电电价排序，找出最贵的时段放电
    sell_price_order = np.argsort(-prices[:, 0])  # 售电电价从高到低
    
    print(f"\n最优充电时段 (购电电价从低到高):")
    for idx in buy_price_order:
        print(f"  时段{idx}: {prices[idx, 1]:.2f} 元/kWh")
    
    print(f"\n最优放电时段 (售电电价从高到低):")
    for idx in sell_price_order:
        print(f"  时段{idx}: {prices[idx, 0]:.2f} 元/kWh")
    
    # 简化的优化：每个时段根据电价差值决定充放电
    for hour in range(hours):
        buy_price = prices[hour, 1]
        sell_price = prices[hour, 0]
        spread = sell_price - buy_price
        
        # 如果售电电价高于购电电价，考虑放电
        if spread > 0 and soc > battery_cap * 0.2:  # SOC 高于 20% 才放电
            # 放电量
            discharge_power = min(max_discharge_power, soc * discharge_efficiency, 
                                 (battery_cap - battery_cap * 0.2))
            energy_sold = discharge_power * discharge_efficiency
            revenue = energy_sold * sell_price
            
            soc -= discharge_power
            charge_schedule.append(0)
            discharge_schedule.append(discharge_power)
            profit.append(revenue)
            
        # 如果购电电价低，考虑充电
        elif buy_price < 0.5 and soc < battery_cap * 0.9:  # 购电电价低于 0.5 且 SOC 低于 90%
            # 充电量
            charge_power = min(max_charge_power, (battery_cap - soc) / charge_efficiency)
            energy_bought = charge_power / charge_efficiency
            cost = energy_bought * buy_price
            
            soc += charge_power * charge_efficiency
            charge_schedule.append(charge_power)
            discharge_schedule.append(0)
            profit.append(-cost)
        else:
            # 不充不放
            charge_schedule.append(0)
            discharge_schedule.append(0)
            profit.append(0)
        
        soc_history.append(soc)
    
    return charge_schedule, discharge_schedule, profit, soc_history

# 创建电价矩阵 (24 小时 x 2 列：售电/购电)
price_matrix = np.column_stack([hourly_selling_prices, hourly_buying_prices])

# 运行优化
charge_sched, discharge_sched, profit_sched, soc_hist = optimize_v2g_operation(
    price_matrix, battery_capacity, max_discharge_power, discharge_efficiency, initial_soc
)

# 计算总收益
total_profit = sum(profit_sched)
total_charged = sum(charge_sched)
total_discharged = sum(discharge_sched)

print(f"\n" + "=" * 60)
print("优化结果")
print("=" * 60)
print(f"总充电量：{total_charged:.2f} kWh")
print(f"总放电量：{total_discharged:.2f} kWh")
print(f"总收益：{total_profit:.2f} 元")

# 创建提交文件
print("\n正在生成 V2G 运营策略提交文件...")

# 创建时间序列 (假设从 2024/11/1 开始)
dates = pd.date_range(start='2024/11/1', periods=24, freq='h')

# 创建提交 DataFrame
submission_v2g = pd.DataFrame({
    'TIME': dates.strftime('%Y/%m/%d %H:%M'),
    'Charge_Power(kW)': charge_sched,
    'Discharge_Power(kW)': discharge_sched,
    'SOC(%)': [soc/battery_capacity*100 for soc in soc_hist],
    'Profit(Yuan)': profit_sched
})

# 保存提交文件
submission_v2g_path = r"submission_task2_v2g_strategy.csv"
submission_v2g.to_csv(submission_v2g_path, index=False, encoding='utf-8-sig')
print(f"\nV2G 策略文件已保存至：{submission_v2g_path}")
print(f"文件形状：{submission_v2g.shape}")
print(f"\n前 10 行数据:")
print(submission_v2g.head(10))

print("\n" + "=" * 60)
print("任务 2 完成！")
print("=" * 60)
