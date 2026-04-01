# 任务 1：充电负荷预测模型 - 运行日志

## 运行时间
2026-04-01

## 模型信息
- 模型类型：Random Forest Regressor
- 参数配置:
  - n_estimators: 100
  - max_depth: 15
  - min_samples_split: 5
  - min_samples_leaf: 2
  - random_state: 42
  - n_jobs: -1

## 训练数据
- 文件：A 榜 - 充电站充电负荷训练数据.csv
- 数据形状：(29280, 14)
- 时间范围：2024-01-01 00:00:00 到 2024-10-31 23:45:00
- 总数据点数：29280
- 清理后数据形状：(28992, 32)

## 特征工程
### 时间特征
- hour, minute, day_of_week, day_of_month, is_weekend
- hour_sin, hour_cos, minute_sin, minute_cos, day_sin, day_cos

### 滞后特征
- lag_1, lag_2, lag_3, lag_4, lag_96, lag_192, lag_288

### 其他特征
- AVGV, MAXV, MINV, S, AVGS, MAXS, MINS, SPAN

## 训练结果
- 训练集形状：X_train=(23193, 26), y_train=(23193,)
- 验证集形状：X_val=(5799, 26), y_val=(5799,)

## 模型评估指标
- RMSE: 0.1668
- MAE: 0.1324
- R²: 0.9933

## 提交结果
- 提交文件：submission_task1.csv
- 提交文件形状：(28992, 2)
- 时间范围：2024-01-04 00:00:00 到 2024-10-31 23:45:00
- 预测值范围：约 4.3 - 9.1

## 运行时长
- 总运行时间：约 10 分 39 秒
- 预测生成速度：约 45-60 it/s

## 结论
模型在验证集上表现良好，R²达到 0.9933，说明模型能够很好地捕捉充电负荷的时间序列模式。
