import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 读取训练数据
print("正在读取训练数据...")
data_dir = r"数据与样例"
files = os.listdir(data_dir)
train_file = files[0]  # A 榜 - 充电站充电负荷训练数据.csv
train_path = os.path.join(data_dir, train_file)
print(f"使用文件：{train_path}")

train_data = pd.read_csv(train_path, encoding='gbk', skiprows=1)

print(f"训练数据形状：{train_data.shape}")
print(f"列名：{train_data.columns.tolist()}")

# 数据预处理 - TIME 列已经存在
train_data['TIME'] = pd.to_datetime(train_data['TIME'])
train_data = train_data.sort_values('TIME').reset_index(drop=True)

print(f"\n时间范围：{train_data['TIME'].min()} 到 {train_data['TIME'].max()}")
print(f"总数据点数：{len(train_data)}")

# 特征工程
def create_features(df):
    df = df.copy()
    df['hour'] = df['TIME'].dt.hour
    df['minute'] = df['TIME'].dt.minute
    df['day_of_week'] = df['TIME'].dt.dayofweek
    df['day_of_month'] = df['TIME'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df

train_data = create_features(train_data)

# 创建滞后特征
def create_lag_features(df, target_col, lags=[1, 2, 3, 4, 96, 192, 288]):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df

train_data = create_lag_features(train_data, 'V', lags=[1, 2, 3, 4, 96, 192, 288])

# 删除含有 NaN 的行
train_data_clean = train_data.dropna().reset_index(drop=True)
print(f"\n清理后数据形状：{train_data_clean.shape}")

# 准备特征和目标 - 排除字符串列和时间列
feature_cols = [col for col in train_data_clean.columns if col not in ['TIME', 'V', 'NAME', 'SENID', 'MAXT', 'MINT']]
X = train_data_clean[feature_cols]
y = train_data_clean['V']

print(f"特征数量：{len(feature_cols)}")

# 划分训练集和验证集
split_ratio = 0.8
split = int(len(X) * split_ratio)
X_train, X_val = X.iloc[:split], X.iloc[split:]
y_train, y_val = y.iloc[:split], y.iloc[split:]

print(f"训练集形状：X_train={X_train.shape}, y_train={y_train.shape}")
print(f"验证集形状：X_val={X_val.shape}, y_val={y_val.shape}")

# 训练随机森林模型
print("\n正在训练随机森林模型...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# 使用 tqdm 包装训练过程
from sklearn.utils import resample
with tqdm(total=1, desc="训练模型") as pbar:
    rf_model.fit(X_train, y_train)
    pbar.update(1)
print("随机森林模型训练完成!")

# 验证集预测
print("\n正在验证模型...")
val_predictions = rf_model.predict(X_val)

# 模型评估
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, val_predictions)
r2 = r2_score(y_val, val_predictions)

print(f"\n随机森林模型评估指标:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# 生成整个数据集的预测
print("\n正在生成完整预测...")

full_predictions = []
full_times = []

start_idx = 288
total_iterations = len(train_data) - start_idx

with tqdm(total=total_iterations, desc="生成预测") as pbar:
    for i in range(start_idx, len(train_data)):
        current_features = train_data.loc[i, feature_cols]
        if current_features.isna().any():
            pbar.update(1)
            continue
        pred = rf_model.predict(current_features.values.reshape(1, -1))[0]
        full_predictions.append(pred)
        full_times.append(train_data.loc[i, 'TIME'])
        pbar.update(1)

# 创建提交文件
submission = pd.DataFrame({
    'TIME': full_times,
    'V': full_predictions
})

submission_path = r"submission_task1.csv"
submission.to_csv(submission_path, index=False, encoding='utf-8-sig')
print(f"\n提交文件已保存至：{submission_path}")
print(f"提交文件形状：{submission.shape}")
print(f"提交文件前 5 行:\n{submission.head()}")
print(f"提交文件后 5 行:\n{submission.tail()}")

print("\n任务 1 完成！")
