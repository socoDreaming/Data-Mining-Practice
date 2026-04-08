import os
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

DATA_DIR = '数据与样例'
TEMPLATE_PATH = os.path.join(DATA_DIR, 'submit_example.csv')
OUTPUT_PATH = 'submission_task1.csv'


@dataclass
class ModelSpec:
    name: str
    model: object


def find_train_csv(data_dir: str) -> str:
    """寻找包含 TIME,V 列的训练数据文件（排除 submit_example.csv）。"""
    candidates = []
    for fn in os.listdir(data_dir):
        if not fn.lower().endswith('.csv'):
            continue
        if fn == 'submit_example.csv':
            continue
        candidates.append(os.path.join(data_dir, fn))

    for path in sorted(candidates):
        try:
            df = pd.read_csv(path, nrows=5)
            if {'TIME', 'V'}.issubset(set(df.columns)):
                return path
        except Exception:
            continue

    raise FileNotFoundError(
        '未找到训练CSV。请把“A榜/B榜单-充电站充电负荷训练数据.csv”放到 数据与样例/ 目录。'
    )


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out['TIME']
    out['hour'] = dt.dt.hour
    out['minute'] = dt.dt.minute
    out['dow'] = dt.dt.dayofweek
    out['dom'] = dt.dt.day
    out['month'] = dt.dt.month
    out['is_weekend'] = (out['dow'] >= 5).astype(int)

    out['hour_sin'] = np.sin(2 * np.pi * out['hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * out['hour'] / 24)
    out['dow_sin'] = np.sin(2 * np.pi * out['dow'] / 7)
    out['dow_cos'] = np.cos(2 * np.pi * out['dow'] / 7)
    out['minute_sin'] = np.sin(2 * np.pi * out['minute'] / 60)
    out['minute_cos'] = np.cos(2 * np.pi * out['minute'] / 60)
    return out


def make_supervised(df: pd.DataFrame, lags=None, rolls=None) -> pd.DataFrame:
    if lags is None:
        lags = [1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 288, 672]
    if rolls is None:
        rolls = [4, 12, 24, 96, 288]

    out = add_time_features(df)
    for lag in lags:
        out[f'lag_{lag}'] = out['V'].shift(lag)

    for w in rolls:
        s = out['V'].shift(1)
        out[f'roll_mean_{w}'] = s.rolling(w).mean()
        out[f'roll_std_{w}'] = s.rolling(w).std()
        out[f'roll_min_{w}'] = s.rolling(w).min()
        out[f'roll_max_{w}'] = s.rolling(w).max()

    out['diff_1'] = out['V'].diff(1)
    out['diff_96'] = out['V'].diff(96)
    return out


def train_ensemble(train_df: pd.DataFrame):
    sup = make_supervised(train_df).dropna().reset_index(drop=True)
    feature_cols = [
        c for c in sup.columns
        if c not in ['TIME', 'V', 'NAME', 'SENID', 'MAXT', 'MINT']
    ]

    X = sup[feature_cols]
    y = sup['V']

    models = [
        ModelSpec('rf', RandomForestRegressor(
            n_estimators=500,
            max_depth=22,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )),
        ModelSpec('et', ExtraTreesRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )),
        ModelSpec('ridge', Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])),
    ]

    # 时间序列CV估计权重
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = {m.name: np.zeros(len(X)) for m in models}

    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr = y.iloc[tr_idx]
        for m in models:
            m.model.fit(Xtr, ytr)
            oof_preds[m.name][va_idx] = m.model.predict(Xva)

    rmses = {}
    for m in models:
        rmse = np.sqrt(mean_squared_error(y, oof_preds[m.name]))
        rmses[m.name] = rmse

    inv = np.array([1.0 / max(rmses[m.name], 1e-6) for m in models])
    weights = inv / inv.sum()
    wmap = {m.name: w for m, w in zip(models, weights)}

    for m in models:
        m.model.fit(X, y)

    return models, wmap, feature_cols


def step_features(history: pd.DataFrame, target_time: pd.Timestamp, feature_cols) -> pd.Series:
    temp = history[['TIME', 'V']].copy()
    temp = pd.concat([temp, pd.DataFrame([{'TIME': target_time, 'V': np.nan}])], ignore_index=True)
    feat_df = make_supervised(temp)
    row = feat_df.iloc[-1]
    return row[feature_cols]


def recursive_predict(train_df: pd.DataFrame, pred_times: pd.Series, models, wmap, feature_cols) -> np.ndarray:
    history = train_df[['TIME', 'V']].copy().sort_values('TIME').reset_index(drop=True)
    preds = []

    for t in pred_times:
        x = step_features(history, pd.to_datetime(t), feature_cols)
        if x.isna().any():
            # 兜底：用最近一周同一时刻平均值
            dt = pd.to_datetime(t)
            mask = (
                (history['TIME'].dt.hour == dt.hour)
                & (history['TIME'].dt.minute == dt.minute)
                & (history['TIME'].dt.dayofweek == dt.dayofweek)
            )
            if mask.any():
                y_hat = history.loc[mask, 'V'].tail(8).mean()
            else:
                y_hat = history['V'].tail(96).mean()
        else:
            x_arr = x.values.reshape(1, -1)
            pred = 0.0
            for m in models:
                pred += wmap[m.name] * float(m.model.predict(x_arr)[0])
            y_hat = pred

        preds.append(y_hat)
        history = pd.concat([
            history,
            pd.DataFrame([{'TIME': pd.to_datetime(t), 'V': y_hat}])
        ], ignore_index=True)

    return np.array(preds)


def main():
    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f'未找到模板文件: {TEMPLATE_PATH}')

    template = pd.read_csv(TEMPLATE_PATH)
    if list(template.columns) != ['TIME', 'V']:
        raise ValueError('submit_example.csv 列名应为 TIME,V')

    train_path = find_train_csv(DATA_DIR)
    # 部分训练文件首行是说明，自动适配
    try:
        train = pd.read_csv(train_path, encoding='gbk')
        if 'TIME' not in train.columns or 'V' not in train.columns:
            train = pd.read_csv(train_path, encoding='gbk', skiprows=1)
    except UnicodeDecodeError:
        train = pd.read_csv(train_path)
        if 'TIME' not in train.columns or 'V' not in train.columns:
            train = pd.read_csv(train_path, skiprows=1)

    train['TIME'] = pd.to_datetime(train['TIME'])
    train['V'] = pd.to_numeric(train['V'], errors='coerce')
    train = train.dropna(subset=['TIME', 'V']).sort_values('TIME').reset_index(drop=True)

    models, wmap, feature_cols = train_ensemble(train)
    preds = recursive_predict(train, template['TIME'], models, wmap, feature_cols)

    sub = pd.DataFrame({'TIME': template['TIME'], 'V': preds})
    sub.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print(f'训练文件: {train_path}')
    print('模型权重:', wmap)
    print(f'输出文件: {OUTPUT_PATH}')
    print(f'输出行数: {len(sub)}')
    print(f'输出时间范围: {sub.iloc[0,0]} -> {sub.iloc[-1,0]}')


if __name__ == '__main__':
    main()
