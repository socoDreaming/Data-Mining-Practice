"""
高级集成学习模型
包含：XGBoost, LightGBM, CatBoost, Stacking Ensemble
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import matplotlib.pyplot as plt

print("=" * 80)
print("高级集成学习模型对比")
print("=" * 80)

# ==================== 数据加载 ====================
print("\n[1/7] 加载数据...")

data_dir = r"数据与样例"
files = os.listdir(data_dir)
train_file = files[0]
train_path = os.path.join(data_dir, train_file)

train_data = pd.read_csv(train_path, encoding='gbk', skiprows=1)
train_data['TIME'] = pd.to_datetime(train_data['TIME'])
train_data = train_data.sort_values('TIME').reset_index(drop=True)

print(f"数据形状：{train_data.shape}")

# ==================== 高级特征工程 ====================
print("\n[2/7] 高级特征工程...")

def create_advanced_features_v2(df):
    """创建更高级的特征"""
    df = df.copy()
    
    # 基础时间特征
    df['hour'] = df['TIME'].dt.hour
    df['minute'] = df['TIME'].dt.minute
    df['day_of_week'] = df['TIME'].dt.dayofweek
    df['day_of_month'] = df['TIME'].dt.day
    df['month'] = df['TIME'].dt.month
    df['quarter'] = df['TIME'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['TIME'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['TIME'].dt.is_month_end.astype(int)
    
    # 周期特征
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 滞后特征
    for lag in [1, 2, 3, 4, 12, 24, 48, 96, 192, 288]:
        df[f'lag_{lag}'] = df['V'].shift(lag)
    
    # 滚动统计特征
    for window in [4, 12, 24, 96, 288]:
        df[f'rolling_mean_{window}'] = df['V'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['V'].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['V'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['V'].rolling(window=window).max()
        df[f'rolling_median_{window}'] = df['V'].rolling(window=window).median()
    
    # 扩展窗口统计
    df['expanding_mean'] = df['V'].expanding().mean()
    df['expanding_std'] = df['V'].expanding().std()
    
    # 差分特征
    for diff in [1, 4, 12, 96]:
        df[f'diff_{diff}'] = df['V'].diff(diff)
    
    # EWM 特征
    for span in [12, 24, 96]:
        df[f'ewm_mean_{span}'] = df['V'].ewm(span=span).mean()
        df[f'ewm_std_{span}'] = df['V'].ewm(span=span).std()
    
    # 分位数特征
    for q in [0.25, 0.5, 0.75]:
        df[f'rolling_quantile_{int(q*100)}_96'] = df['V'].rolling(window=96).quantile(q)
    
    return df

train_data = create_advanced_features_v2(train_data)
train_data_clean = train_data.dropna().reset_index(drop=True)

print(f"清理后数据形状：{train_data_clean.shape}")
print(f"特征数量：{len(train_data_clean.columns) - 5}")  # 减去 TIME, V, NAME, SENID, MAXT, MINT

# ==================== 准备数据 ====================
print("\n[3/7] 准备训练数据...")

feature_cols = [col for col in train_data_clean.columns if col not in ['TIME', 'V', 'NAME', 'SENID', 'MAXT', 'MINT']]
X = train_data_clean[feature_cols].values
y = train_data_clean['V'].values

# 数据标准化
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 时间序列划分
split_ratio = 0.8
split = int(len(X_scaled) * split_ratio)
X_train, X_val = X_scaled[:split], X_scaled[split:]
y_train, y_val = y_scaled[:split], y_scaled[split:]

print(f"训练集：{X_train.shape}, 验证集：{X_val.shape}")

# ==================== 定义集成学习模型 ====================
print("\n[4/7] 定义集成学习模型...")

# 尝试导入高级库
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("未安装 XGBoost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("未安装 LightGBM")

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("未安装 CatBoost")

ensemble_models = {}

# 基础模型
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

base_models = {
    'RF': RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=3, n_jobs=-1, random_state=42),
    'ET': ExtraTreesRegressor(n_estimators=200, max_depth=20, min_samples_split=3, n_jobs=-1, random_state=42),
    'GB': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
}

# 添加高级模型
if HAS_XGB:
    base_models['XGB'] = xgb.XGBRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1
    )

if HAS_LGB:
    base_models['LGB'] = lgb.LGBMRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1
    )

if HAS_CB:
    base_models['CatBoost'] = cb.CatBoostRegressor(
        iterations=200, depth=8, learning_rate=0.05,
        verbose=0, random_state=42
    )

ensemble_models.update(base_models)

# Stacking 集成
print("\n创建 Stacking 集成模型...")
stacking_model = StackingRegressor(
    estimators=[(name, model) for name, model in list(ensemble_models.items())[:4]],
    final_estimator=GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
    cv=5,
    n_jobs=-1
)
ensemble_models['Stacking'] = stacking_model

# Voting 集成
print("创建 Voting 集成模型...")
voting_model = VotingRegressor(
    estimators=[(name, model) for name, model in list(ensemble_models.items())[:4]],
    n_jobs=-1
)
ensemble_models['Voting'] = voting_model

print(f"共定义 {len(ensemble_models)} 个集成模型")

# ==================== 训练和评估 ====================
print("\n[5/7] 训练和评估模型...")

ensemble_results = {}

for name, model in tqdm(ensemble_models.items(), desc="训练集成模型"):
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
    
    ensemble_results[name] = {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'predictions': y_pred,
        'true': y_true
    }
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")

# ==================== 特征重要性分析 ====================
print("\n[6/7] 特征重要性分析...")

# 提取特征重要性（如果有）
feature_importance_df = None

if HAS_XGB and 'XGB' in ensemble_results:
    xgb_model = ensemble_results['XGB']['model']
    if hasattr(xgb_model, 'feature_importances_'):
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nXGBoost 前 20 重要特征:")
        print(feature_importance_df.head(20).to_string(index=False))

# ==================== 结果对比和可视化 ====================
print("\n[7/7] 结果对比和可视化...")

# 创建结果 DataFrame
ensemble_results_df = pd.DataFrame({
    'Model': list(ensemble_results.keys()),
    'RMSE': [ensemble_results[m]['rmse'] for m in ensemble_results.keys()],
    'MAE': [ensemble_results[m]['mae'] for m in ensemble_results.keys()],
    'MAPE(%)': [ensemble_results[m]['mape'] for m in ensemble_results.keys()],
    'R2': [ensemble_results[m]['r2'] for m in ensemble_results.keys()]
}).sort_values('RMSE')

print("\n集成学习模型性能排名:")
print(ensemble_results_df.to_string(index=False))

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# 1. RMSE 对比
ax1 = axes[0, 0]
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(ensemble_results_df)))
bars = ax1.barh(ensemble_results_df['Model'], ensemble_results_df['RMSE'], color=colors)
ax1.set_xlabel('RMSE')
ax1.set_title('集成模型 RMSE 对比')
ax1.invert_yaxis()
for i, (model, rmse) in enumerate(zip(ensemble_results_df['Model'], ensemble_results_df['RMSE'])):
    ax1.text(rmse, i, f' {rmse:.4f}', va='center', fontsize=9)

# 2. R²对比
ax2 = axes[0, 1]
colors2 = plt.cm.Blues(np.linspace(0.3, 0.9, len(ensemble_results_df)))
bars2 = ax2.barh(ensemble_results_df['Model'], ensemble_results_df['R2'], color=colors2)
ax2.set_xlabel('R²')
ax2.set_title('集成模型 R² 对比')
ax2.invert_yaxis()
for i, (model, r2) in enumerate(zip(ensemble_results_df['Model'], ensemble_results_df['R2'])):
    ax2.text(r2, i, f' {r2:.4f}', va='center', fontsize=9)

# 3. MAPE 对比
ax3 = axes[0, 2]
colors3 = plt.cm.Oranges(np.linspace(0.3, 0.9, len(ensemble_results_df)))
bars3 = ax3.barh(ensemble_results_df['Model'], ensemble_results_df['MAPE(%)'], color=colors3)
ax3.set_xlabel('MAPE (%)')
ax3.set_title('集成模型 MAPE 对比')
ax3.invert_yaxis()
for i, (model, mape) in enumerate(zip(ensemble_results_df['Model'], ensemble_results_df['MAPE(%)'])):
    ax3.text(mape, i, f' {mape:.2f}%', va='center', fontsize=9)

# 4. 最佳模型预测对比
ax4 = axes[1, 0]
best_model = ensemble_results_df.iloc[0]['Model']
y_pred_best = ensemble_results[best_model]['predictions'][:500]
y_true_best = ensemble_results[best_model]['true'][:500]
ax4.plot(y_true_best, label='实际值', linewidth=2, alpha=0.7)
ax4.plot(y_pred_best, label=f'{best_model}预测', linestyle='--', linewidth=2, alpha=0.7)
ax4.set_xlabel('时间点')
ax4.set_ylabel('充电功率')
ax4.set_title(f'最佳集成模型 ({best_model}) 预测对比')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 散点图
ax5 = axes[1, 1]
ax5.scatter(y_true_best, y_pred_best, alpha=0.5, s=15, c='blue')
ax5.plot([y_true_best.min(), y_true_best.max()], 
         [y_true_best.min(), y_true_best.max()], 'r--', linewidth=2)
ax5.set_xlabel('实际值')
ax5.set_ylabel('预测值')
ax5.set_title(f'{best_model} 预测散点图')
ax5.grid(True, alpha=0.3)

# 6. 特征重要性（如果有）
ax6 = axes[1, 2]
if feature_importance_df is not None:
    top_features = feature_importance_df.head(15)
    ax6.barh(range(len(top_features)), top_features['importance'].values)
    ax6.set_yticks(range(len(top_features)))
    ax6.set_yticklabels(top_features['feature'].values, fontsize=8)
    ax6.set_xlabel('重要性')
    ax6.set_title('XGBoost 前 15 重要特征')
    ax6.invert_yaxis()
else:
    ax6.text(0.5, 0.5, '无特征重要性数据', ha='center', va='center', fontsize=14)
    ax6.axis('off')

plt.tight_layout()
plt.savefig('ensemble_models_comparison.png', dpi=300, bbox_inches='tight')
print("\n集成模型对比图已保存为：ensemble_models_comparison.png")
plt.close()

# 保存结果
ensemble_results_df.to_csv('ensemble_models_results.csv', index=False, encoding='utf-8-sig')
print("集成模型结果已保存为：ensemble_models_results.csv")

print("\n" + "=" * 80)
print("集成学习模型对比完成！")
print("=" * 80)
print(f"\n最佳集成模型：{best_model}")
print(f"最佳 RMSE: {ensemble_results_df.iloc[0]['RMSE']:.4f}")
print(f"最佳 R²: {ensemble_results_df.iloc[0]['R2']:.4f}")
