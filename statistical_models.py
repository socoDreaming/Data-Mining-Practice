"""
传统统计时间序列模型
包含：ARIMA, SARIMA, Prophet, Exponential Smoothing
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tqdm import tqdm
import matplotlib.pyplot as plt

print("=" * 80)
print("传统统计时间序列模型对比")
print("=" * 80)

# ==================== 数据加载 ====================
print("\n[1/6] 加载数据...")

data_dir = r"数据与样例"
files = os.listdir(data_dir)
train_file = files[0]
train_path = os.path.join(data_dir, train_file)

train_data = pd.read_csv(train_path, encoding='gbk', skiprows=1)
train_data['TIME'] = pd.to_datetime(train_data['TIME'])
train_data = train_data.sort_values('TIME').reset_index(drop=True)

values = train_data['V'].values
time_index = train_data['TIME']

print(f"数据形状：{values.shape}")
print(f"时间范围：{time_index.min()} 到 {time_index.max()}")

# ==================== 尝试导入统计模型库 ====================
print("\n[2/6] 检查统计模型库...")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
    print("✓ Statsmodels 已安装")
except ImportError:
    HAS_STATSMODELS = False
    print("✗ Statsmodels 未安装")

try:
    from prophet import Prophet
    HAS_PROPHET = True
    print("✓ Prophet 已安装")
except ImportError:
    HAS_PROPHET = False
    print("✗ Prophet 未安装")

# ==================== 数据准备 ====================
print("\n[3/6] 准备数据...")

# 划分训练集和验证集
split_ratio = 0.8
split = int(len(values) * split_ratio)
train_values = values[:split]
val_values = values[split:]
train_time = time_index[:split]
val_time = time_index[split:]

print(f"训练集：{len(train_values)}, 验证集：{len(val_values)}")

# ==================== 统计模型预测 ====================
print("\n[4/6] 训练统计模型...")

stat_results = {}

if HAS_STATSMODELS:
    # 1. ARIMA 模型（简化版本，使用较小的参数）
    print("\n训练 ARIMA 模型...")
    try:
        # 为了速度，使用部分数据训练
        train_subset = train_values[-500:]  # 使用最后 500 个点
        
        arima_model = ARIMA(train_subset, order=(2, 1, 2))
        arima_fit = arima_model.fit()
        
        # 预测
        arima_forecast = arima_fit.forecast(steps=len(val_values))
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(val_values, arima_forecast))
        mae = mean_absolute_error(val_values, arima_forecast)
        mape = mean_absolute_percentage_error(val_values, arima_forecast) * 100
        r2 = r2_score(val_values, arima_forecast)
        
        stat_results['ARIMA'] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'predictions': arima_forecast,
            'true': val_values
        }
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
    except Exception as e:
        print(f"  ARIMA 训练失败：{str(e)}")
    
    # 2. 指数平滑
    print("\n训练指数平滑模型...")
    try:
        es_model = ExponentialSmoothing(
            train_values[-1000:],  # 使用最后 1000 个点
            trend='add',
            seasonal='add',
            seasonal_periods=96  # 日周期
        )
        es_fit = es_model.fit()
        
        # 预测
        es_forecast = es_fit.forecast(len(val_values))
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(val_values, es_forecast))
        mae = mean_absolute_error(val_values, es_forecast)
        mape = mean_absolute_percentage_error(val_values, es_forecast) * 100
        r2 = r2_score(val_values, es_forecast)
        
        stat_results['ExpSmoothing'] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'predictions': es_forecast,
            'true': val_values
        }
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
    except Exception as e:
        print(f"  指数平滑训练失败：{str(e)}")

if HAS_PROPHET:
    # 3. Prophet 模型
    print("\n训练 Prophet 模型...")
    try:
        # 准备 Prophet 数据格式
        prophet_df = pd.DataFrame({
            'ds': train_time,
            'y': train_values
        })
        
        prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        prophet_model.fit(prophet_df)
        
        # 创建未来数据框
        future = prophet_model.make_future_dataframe(
            periods=len(val_values),
            freq='15T'
        )
        
        # 预测
        prophet_forecast = prophet_model.predict(future)
        prophet_pred = prophet_forecast['yhat'].values[-len(val_values):]
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(val_values, prophet_pred))
        mae = mean_absolute_error(val_values, prophet_pred)
        mape = mean_absolute_percentage_error(val_values, prophet_pred) * 100
        r2 = r2_score(val_values, prophet_pred)
        
        stat_results['Prophet'] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'predictions': prophet_pred,
            'true': val_values
        }
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
    except Exception as e:
        print(f"  Prophet 训练失败：{str(e)}")

# 简单基线模型（Naive Forecast）
print("\n训练基线模型（Naive Forecast）...")
naive_forecast = np.full(len(val_values), train_values[-1])
rmse = np.sqrt(mean_squared_error(val_values, naive_forecast))
mae = mean_absolute_error(val_values, naive_forecast)
mape = mean_absolute_percentage_error(val_values, naive_forecast) * 100
r2 = r2_score(val_values, naive_forecast)

stat_results['Naive'] = {
    'rmse': rmse,
    'mae': mae,
    'mape': mape,
    'r2': r2,
    'predictions': naive_forecast,
    'true': val_values
}

print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")

# ==================== 结果对比 ====================
print("\n[5/6] 统计模型性能对比...")

if stat_results:
    stat_results_df = pd.DataFrame({
        'Model': list(stat_results.keys()),
        'RMSE': [stat_results[m]['rmse'] for m in stat_results.keys()],
        'MAE': [stat_results[m]['mae'] for m in stat_results.keys()],
        'MAPE(%)': [stat_results[m]['mape'] for m in stat_results.keys()],
        'R2': [stat_results[m]['r2'] for m in stat_results.keys()]
    }).sort_values('RMSE')
    
    print("\n统计模型性能排名:")
    print(stat_results_df.to_string(index=False))

# ==================== 可视化 ====================
print("\n[6/6] 生成可视化...")

if stat_results:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. RMSE 对比
    ax1 = axes[0, 0]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(stat_results_df)))
    bars = ax1.barh(stat_results_df['Model'], stat_results_df['RMSE'], color=colors)
    ax1.set_xlabel('RMSE')
    ax1.set_title('统计模型 RMSE 对比')
    ax1.invert_yaxis()
    for i, (model, rmse) in enumerate(zip(stat_results_df['Model'], stat_results_df['RMSE'])):
        ax1.text(rmse, i, f' {rmse:.4f}', va='center', fontsize=10)
    
    # 2. R²对比
    ax2 = axes[0, 1]
    colors2 = plt.cm.Purples(np.linspace(0.3, 0.9, len(stat_results_df)))
    bars2 = ax2.barh(stat_results_df['Model'], stat_results_df['R2'], color=colors2)
    ax2.set_xlabel('R²')
    ax2.set_title('统计模型 R² 对比')
    ax2.invert_yaxis()
    for i, (model, r2) in enumerate(zip(stat_results_df['Model'], stat_results_df['R2'])):
        ax2.text(r2, i, f' {r2:.4f}', va='center', fontsize=10)
    
    # 3. 预测对比（所有模型）
    ax3 = axes[1, 0]
    for name, result in stat_results.items():
        ax3.plot(result['predictions'][:200], label=f"{name} (RMSE={result['rmse']:.4f})", linewidth=1.5, alpha=0.7)
    
    ax3.plot(result['true'][:200], 'k-', label='实际值', linewidth=2)
    ax3.set_xlabel('时间点')
    ax3.set_ylabel('充电功率')
    ax3.set_title('统计模型预测对比（前 200 点）')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 残差对比
    ax4 = axes[1, 1]
    residuals_data = []
    for name, result in stat_results.items():
        residuals = result['true'][:200] - result['predictions'][:200]
        residuals_data.append(residuals)
    
    ax4.boxplot(residuals_data, labels=stat_results.keys())
    ax4.set_ylabel('残差')
    ax4.set_title('模型残差分布对比')
    ax4.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('statistical_models_comparison.png', dpi=300, bbox_inches='tight')
    print("统计模型对比图已保存为：statistical_models_comparison.png")
    plt.close()
    
    # 保存结果
    stat_results_df.to_csv('statistical_models_results.csv', index=False, encoding='utf-8-sig')
    print("统计模型结果已保存为：statistical_models_results.csv")

print("\n" + "=" * 80)
print("传统统计模型对比完成！")
print("=" * 80)

if stat_results:
    best_stat_model = stat_results_df.iloc[0]['Model']
    print(f"\n最佳统计模型：{best_stat_model}")
    print(f"最佳 RMSE: {stat_results_df.iloc[0]['RMSE']:.4f}")
    print(f"最佳 R²: {stat_results_df.iloc[0]['R2']:.4f}")
