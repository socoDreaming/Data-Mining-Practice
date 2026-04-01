"""
高级时间序列预测模型对比框架
包含：传统统计模型、机器学习模型、深度学习模型
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("高级时间序列预测模型对比框架")
print("=" * 80)

# ==================== 数据加载与预处理 ====================
print("\n[1/8] 加载和预处理数据...")

data_dir = r"数据与样例"
files = os.listdir(data_dir)
train_file = files[0]
train_path = os.path.join(data_dir, train_file)

train_data = pd.read_csv(train_path, encoding='gbk', skiprows=1)
train_data['TIME'] = pd.to_datetime(train_data['TIME'])
train_data = train_data.sort_values('TIME').reset_index(drop=True)

print(f"数据形状：{train_data.shape}")
print(f"时间范围：{train_data['TIME'].min()} 到 {train_data['TIME'].max()}")

# ==================== 特征工程 ====================
print("\n[2/8] 特征工程...")

def create_advanced_features(df):
    """创建高级特征"""
    df = df.copy()
    
    # 基础时间特征
    df['hour'] = df['TIME'].dt.hour
    df['minute'] = df['TIME'].dt.minute
    df['day_of_week'] = df['TIME'].dt.dayofweek
    df['day_of_month'] = df['TIME'].dt.day
    df['month'] = df['TIME'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 周期特征（正弦/余弦编码）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 滞后特征（多尺度）
    for lag in [1, 2, 3, 4, 12, 24, 48, 96, 192, 288]:
        df[f'lag_{lag}'] = df['V'].shift(lag)
    
    # 滚动统计特征
    for window in [4, 12, 24, 96]:
        df[f'rolling_mean_{window}'] = df['V'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['V'].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['V'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['V'].rolling(window=window).max()
    
    # 差分特征
    df['diff_1'] = df['V'].diff(1)
    df['diff_4'] = df['V'].diff(4)
    df['diff_96'] = df['V'].diff(96)
    
    # 扩展特征
    df['ewm_mean_12'] = df['V'].ewm(span=12).mean()
    df['ewm_mean_24'] = df['V'].ewm(span=24).mean()
    
    return df

train_data = create_advanced_features(train_data)
train_data_clean = train_data.dropna().reset_index(drop=True)

print(f"清理后数据形状：{train_data_clean.shape}")

# ==================== 模型定义 ====================
print("\n[3/8] 定义模型...")

# 准备数据
feature_cols = [col for col in train_data_clean.columns if col not in ['TIME', 'V', 'NAME', 'SENID', 'MAXT', 'MINT']]
X = train_data_clean[feature_cols].values
y = train_data_clean['V'].values

# 数据归一化
scaler_x = StandardScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 划分训练集和验证集
split_ratio = 0.8
split = int(len(X_scaled) * split_ratio)
X_train, X_val = X_scaled[:split], X_scaled[split:]
y_train, y_val = y_scaled[:split], y_scaled[split:]

print(f"训练集：{X_train.shape}, 验证集：{X_val.shape}")

# 定义多个模型
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, n_jobs=-1, random_state=42),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.01, random_state=42, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1),
    'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)
}

# ==================== 模型训练与评估 ====================
print("\n[4/8] 训练和评估模型...")

results = {}

for name, model in tqdm(models.items(), desc="训练模型"):
    print(f"\n训练 {name}...")
    model.fit(X_train, y_train)
    
    # 预测
    y_pred_scaled = model.predict(X_val)
    
    # 反归一化
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    results[name] = {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'predictions': y_pred,
        'true': y_true
    }
    
    print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")

# ==================== 结果对比 ====================
print("\n[5/8] 模型性能对比...")

# 创建结果 DataFrame
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[m]['rmse'] for m in results.keys()],
    'MAE': [results[m]['mae'] for m in results.keys()],
    'MAPE(%)': [results[m]['mape'] for m in results.keys()],
    'R2': [results[m]['r2'] for m in results.keys()]
}).sort_values('RMSE')

print("\n模型性能排名（按 RMSE）:")
print(results_df.to_string(index=False))

# ==================== 可视化 ====================
print("\n[6/8] 生成可视化图表...")

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. RMSE 对比
ax1 = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
bars = ax1.barh(results_df['Model'], results_df['RMSE'], color=colors)
ax1.set_xlabel('RMSE')
ax1.set_title('RMSE 对比')
ax1.invert_yaxis()
for i, (model, rmse) in enumerate(zip(results_df['Model'], results_df['RMSE'])):
    ax1.text(rmse, i, f' {rmse:.4f}', va='center', fontsize=9)

# 2. R²对比
ax2 = axes[0, 1]
colors2 = plt.cm.plasma(np.linspace(0, 1, len(results_df)))
bars2 = ax2.barh(results_df['Model'], results_df['R2'], color=colors2)
ax2.set_xlabel('R²')
ax2.set_title('R² 对比')
ax2.invert_yaxis()
for i, (model, r2) in enumerate(zip(results_df['Model'], results_df['R2'])):
    ax2.text(r2, i, f' {r2:.4f}', va='center', fontsize=9)

# 3. MAPE 对比
ax3 = axes[0, 2]
colors3 = plt.cm.Greens(np.linspace(0, 1, len(results_df)))
bars3 = ax3.barh(results_df['Model'], results_df['MAPE(%)'], color=colors3)
ax3.set_xlabel('MAPE (%)')
ax3.set_title('MAPE 对比')
ax3.invert_yaxis()
for i, (model, mape) in enumerate(zip(results_df['Model'], results_df['MAPE(%)'])):
    ax3.text(mape, i, f' {mape:.2f}%', va='center', fontsize=9)

# 4. 预测 vs 实际（最佳模型）
ax4 = axes[1, 0]
best_model = results_df.iloc[0]['Model']
y_pred_best = results[best_model]['predictions'][:500]
y_true_best = results[best_model]['true'][:500]
ax4.plot(y_true_best, label='实际值', linewidth=2)
ax4.plot(y_pred_best, label=f'{best_model}预测', linestyle='--', linewidth=2)
ax4.set_xlabel('时间点')
ax4.set_ylabel('充电功率')
ax4.set_title(f'最佳模型 ({best_model}) 预测对比')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 散点图
ax5 = axes[1, 1]
ax5.scatter(y_true_best, y_pred_best, alpha=0.5, s=10)
ax5.plot([y_true_best.min(), y_true_best.max()], 
         [y_true_best.min(), y_true_best.max()], 'r--', linewidth=2)
ax5.set_xlabel('实际值')
ax5.set_ylabel('预测值')
ax5.set_title(f'{best_model} 预测散点图')
ax5.grid(True, alpha=0.3)

# 6. 残差分析
ax6 = axes[1, 2]
residuals = y_true_best - y_pred_best
ax6.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax6.set_xlabel('残差')
ax6.set_ylabel('频数')
ax6.set_title('残差分布')
ax6.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("可视化图表已保存为：model_comparison.png")
plt.close()

# ==================== 保存最佳模型预测结果 ====================
print("\n[7/8] 保存最佳模型预测结果...")

# 使用最佳模型生成完整预测
best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

# 生成完整数据集的预测
print(f"使用最佳模型 {best_model_name} 生成完整预测...")

full_predictions = []
full_times = []

start_idx = 288
total_iterations = len(train_data) - start_idx

with tqdm(total=total_iterations, desc="生成预测") as pbar:
    for i in range(start_idx, len(train_data)):
        current_features = train_data_clean.loc[i - start_idx, feature_cols]
        if np.any(np.isnan(current_features)):
            pbar.update(1)
            continue
        
        # 特征缩放
        current_features_scaled = scaler_x.transform(current_features.values.reshape(1, -1))
        pred_scaled = best_model.predict(current_features_scaled)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        
        full_predictions.append(pred)
        full_times.append(train_data.loc[i, 'TIME'])
        pbar.update(1)

# 创建提交文件
submission = pd.DataFrame({
    'TIME': full_times,
    'V': full_predictions
})

submission_path = r"submission_task1_advanced.csv"
submission.to_csv(submission_path, index=False, encoding='utf-8-sig')
print(f"\n提交文件已保存至：{submission_path}")
print(f"提交文件形状：{submission.shape}")

# ==================== 保存详细结果 ====================
print("\n[8/8] 保存详细结果...")

# 保存结果到 CSV
results_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8-sig')
print("模型对比结果已保存为：model_comparison_results.csv")

# 保存预测对比图
fig_pred, ax_pred = plt.subplots(1, 1, figsize=(15, 6))
for i, (name, result) in enumerate(results.items()):
    if i < 5:  # 只显示前 5 个模型
        ax_pred.plot(result['predictions'][:200], label=f"{name} (RMSE={result['rmse']:.4f})", linewidth=1.5)

ax_pred.plot(result['true'][:200], 'k-', label='实际值', linewidth=2)
ax_pred.set_xlabel('时间点')
ax_pred.set_ylabel('充电功率')
ax_pred.set_title('前 5 个模型预测对比')
ax_pred.legend()
ax_pred.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('top5_models_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("前 5 模型对比图已保存为：top5_models_comparison.png")

print("\n" + "=" * 80)
print("高级模型对比完成！")
print("=" * 80)
print(f"\n最佳模型：{best_model_name}")
print(f"最佳 RMSE: {results_df.iloc[0]['RMSE']:.4f}")
print(f"最佳 R²: {results_df.iloc[0]['R2']:.4f}")
print(f"\n生成的文件:")
print(f"  1. {submission_path} - 最佳模型预测结果")
print(f"  2. model_comparison_results.csv - 模型对比结果")
print(f"  3. model_comparison.png - 模型性能可视化")
print(f"  4. top5_models_comparison.png - 前 5 模型预测对比图")
