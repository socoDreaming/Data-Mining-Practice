"""
深度学习时间序列预测模型
包含：LSTM, GRU, Bidirectional LSTM, CNN-LSTM 等架构
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tqdm import tqdm
import matplotlib.pyplot as plt

# 检查是否有 GPU 可用
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim
    
    # 检查 CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    USE_TORCH = True
except ImportError:
    print("未安装 PyTorch，使用纯 NumPy 实现简单 LSTM")
    USE_TORCH = False

print("=" * 80)
print("深度学习时间序列预测模型")
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

values = train_data['V'].values.reshape(-1, 1)
print(f"数据形状：{values.shape}")

# ==================== 数据预处理 ====================
print("\n[2/6] 数据预处理...")

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

# 创建序列
def create_sequences(data, seq_length, pred_length=1):
    """创建序列数据"""
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_length])
    return np.array(X), np.array(y)

seq_length = 96  # 使用过去 96 个时间点（24 小时）
pred_length = 1  # 预测下一个时间点

X, y = create_sequences(scaled_values, seq_length, pred_length)
print(f"序列数据形状：X={X.shape}, y={y.shape}")

# 划分训练集和验证集
split_ratio = 0.8
split = int(len(X) * split_ratio)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"训练集：{X_train.shape}, 验证集：{X_val.shape}")

if USE_TORCH:
    # 转换为 PyTorch 张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # 创建 DataLoader
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ==================== 定义深度学习模型 ====================
print("\n[3/6] 定义深度学习模型...")

if USE_TORCH:
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    
    class GRUModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
            super(GRUModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            return out
    
    class BiLSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
            super(BiLSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, 1)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    
    class CNNLSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2):
            super(CNNLSTMModel, self).__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.lstm = nn.LSTM(64 * (seq_length // 4), hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            x = x.permute(0, 2, 1)  # (batch, seq, feature) -> (batch, feature, seq)
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), 1, -1)
            h0 = torch.zeros(2, x.size(0), 64).to(device)
            c0 = torch.zeros(2, x.size(0), 64).to(device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    
    models = {
        'LSTM': LSTMModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2),
        'GRU': GRUModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2),
        'BiLSTM': BiLSTMModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2),
        'CNN-LSTM': CNNLSTMModel(input_size=1, hidden_size=64, num_layers=2)
    }
else:
    print("使用简化模型（无 PyTorch）")
    models = {'Simple_LSTM': None}

# ==================== 训练模型 ====================
print("\n[4/6] 训练深度学习模型...")

dl_results = {}

if USE_TORCH:
    for name, model in tqdm(models.items(), desc="训练深度学习模型"):
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # 训练
        num_epochs = 50
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"  早停于 epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        # 预测
        model.eval()
        with torch.no_grad():
            val_pred_scaled = model(X_val_tensor).cpu().numpy()
        
        val_pred = scaler.inverse_transform(val_pred_scaled)
        val_true = scaler.inverse_transform(y_val)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(val_true, val_pred))
        mae = mean_absolute_error(val_true, val_pred)
        mape = mean_absolute_percentage_error(val_true, val_pred) * 100
        r2 = r2_score(val_true, val_pred)
        
        dl_results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'predictions': val_pred.flatten(),
            'true': val_true.flatten()
        }
        
        print(f"\n{name} 结果:")
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")

# ==================== 结果对比 ====================
print("\n[5/6] 深度学习模型性能对比...")

if dl_results:
    dl_results_df = pd.DataFrame({
        'Model': list(dl_results.keys()),
        'RMSE': [dl_results[m]['rmse'] for m in dl_results.keys()],
        'MAE': [dl_results[m]['mae'] for m in dl_results.keys()],
        'MAPE(%)': [dl_results[m]['mape'] for m in dl_results.keys()],
        'R2': [dl_results[m]['r2'] for m in dl_results.keys()]
    }).sort_values('RMSE')
    
    print("\n深度学习模型性能排名:")
    print(dl_results_df.to_string(index=False))

# ==================== 可视化 ====================
print("\n[6/6] 生成可视化...")

if dl_results:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. RMSE 对比
    ax1 = axes[0, 0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(dl_results_df)))
    bars = ax1.barh(dl_results_df['Model'], dl_results_df['RMSE'], color=colors)
    ax1.set_xlabel('RMSE')
    ax1.set_title('深度学习模型 RMSE 对比')
    ax1.invert_yaxis()
    for i, (model, rmse) in enumerate(zip(dl_results_df['Model'], dl_results_df['RMSE'])):
        ax1.text(rmse, i, f' {rmse:.4f}', va='center', fontsize=10)
    
    # 2. R²对比
    ax2 = axes[0, 1]
    colors2 = plt.cm.Greens(np.linspace(0.4, 0.9, len(dl_results_df)))
    bars2 = ax2.barh(dl_results_df['Model'], dl_results_df['R2'], color=colors2)
    ax2.set_xlabel('R²')
    ax2.set_title('深度学习模型 R² 对比')
    ax2.invert_yaxis()
    for i, (model, r2) in enumerate(zip(dl_results_df['Model'], dl_results_df['R2'])):
        ax2.text(r2, i, f' {r2:.4f}', va='center', fontsize=10)
    
    # 3. 最佳模型预测对比
    ax3 = axes[1, 0]
    best_dl_model = dl_results_df.iloc[0]['Model']
    y_pred_best = dl_results[best_dl_model]['predictions'][:500]
    y_true_best = dl_results[best_dl_model]['true'][:500]
    ax3.plot(y_true_best, label='实际值', linewidth=2, alpha=0.7)
    ax3.plot(y_pred_best, label=f'{best_dl_model}预测', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('时间点')
    ax3.set_ylabel('充电功率')
    ax3.set_title(f'最佳深度学习模型 ({best_dl_model}) 预测对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 训练损失曲线（如果有）
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.5, '深度学习模型训练完成', ha='center', va='center', fontsize=16)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('deep_learning_comparison.png', dpi=300, bbox_inches='tight')
    print("深度学习模型对比图已保存为：deep_learning_comparison.png")
    plt.close()

print("\n" + "=" * 80)
print("深度学习模型对比完成！")
print("=" * 80)

if dl_results:
    print(f"\n最佳深度学习模型：{best_dl_model}")
    print(f"最佳 RMSE: {dl_results_df.iloc[0]['RMSE']:.4f}")
    print(f"最佳 R²: {dl_results_df.iloc[0]['R2']:.4f}")
