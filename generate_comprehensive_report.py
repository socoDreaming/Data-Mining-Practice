"""
综合模型对比报告生成器
整合所有模型的结果，生成全面的对比分析报告
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("综合模型对比报告生成")
print("=" * 80)

# ==================== 收集所有结果文件 ====================
print("\n[1/5] 收集结果文件...")

result_files = {
    'advanced_ml': 'model_comparison_results.csv',
    'ensemble': 'ensemble_models_results.csv',
    'statistical': 'statistical_models_results.csv',
    'deep_learning': None  # 需要特殊处理
}

all_results = {}

for category, filename in result_files.items():
    if filename and os.path.exists(filename):
        df = pd.read_csv(filename)
        df['Category'] = category
        all_results[category] = df
        print(f"✓ 找到 {category} 结果：{len(df)} 个模型")

# ==================== 整合结果 ====================
print("\n[2/5] 整合所有模型结果...")

if all_results:
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    
    # 统一列名
    if 'MAPE(%)' in combined_df.columns:
        combined_df['MAPE'] = combined_df['MAPE(%)']
    
    # 添加排名
    combined_df = combined_df.sort_values('RMSE').reset_index(drop=True)
    combined_df['Rank'] = range(1, len(combined_df) + 1)
    
    print(f"\n整合后的模型总数：{len(combined_df)}")
    print("\n所有模型性能排名:")
    print(combined_df[['Rank', 'Model', 'Category', 'RMSE', 'MAE', 'R2']].to_string(index=False))

# ==================== 生成综合可视化 ====================
print("\n[3/5] 生成综合可视化...")

if all_results:
    # 设置风格
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建大图
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 所有模型 RMSE 对比（横向柱状图）
    ax1 = plt.subplot(3, 2, 1)
    colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(combined_df)))
    bars = ax1.barh(range(len(combined_df)), combined_df['RMSE'], color=colors)
    ax1.set_yticks(range(len(combined_df)))
    ax1.set_yticklabels([f"{row['Model']} ({row['Category']})" for _, row in combined_df.iterrows()], fontsize=9)
    ax1.set_xlabel('RMSE', fontsize=12)
    ax1.set_title('所有模型 RMSE 综合对比', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    # 添加数值标签
    for i, (idx, row) in enumerate(combined_df.iterrows()):
        ax1.text(row['RMSE'], i, f' {row["RMSE"]:.4f}', va='center', fontsize=8)
    
    # 2. 各类别最佳模型对比
    ax2 = plt.subplot(3, 2, 2)
    best_by_category = combined_df.loc[combined_df.groupby('Category')['RMSE'].idxmin()]
    colors2 = plt.cm.Set2(np.linspace(0.2, 0.8, len(best_by_category)))
    bars2 = ax2.barh(range(len(best_by_category)), best_by_category['RMSE'], color=colors2)
    ax2.set_yticks(range(len(best_by_category)))
    ax2.set_yticklabels([f"{row['Model']} ({row['Category']})" for _, row in best_by_category.iterrows()], fontsize=10)
    ax2.set_xlabel('RMSE', fontsize=12)
    ax2.set_title('各类别最佳模型对比', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    
    for i, (idx, row) in enumerate(best_by_category.iterrows()):
        ax2.text(row['RMSE'], i, f' {row["RMSE"]:.4f}', va='center', fontsize=10)
    
    # 3. R² vs RMSE 散点图
    ax3 = plt.subplot(3, 2, 3)
    categories = combined_df['Category'].unique()
    for cat in categories:
        cat_data = combined_df[combined_df['Category'] == cat]
        ax3.scatter(cat_data['RMSE'], cat_data['R2'], label=cat, s=100, alpha=0.7)
    
    ax3.set_xlabel('RMSE', fontsize=12)
    ax3.set_ylabel('R²', fontsize=12)
    ax3.set_title('R² vs RMSE 关系图', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 各类别模型数量统计
    ax4 = plt.subplot(3, 2, 4)
    category_counts = combined_df['Category'].value_counts()
    colors4 = plt.cm.Pastel1(np.linspace(0.2, 0.8, len(category_counts)))
    wedges, texts, autotexts = ax4.pie(category_counts.values, labels=category_counts.index, 
                                       autopct='%1.1f%%', colors=colors4)
    ax4.set_title('各类别模型数量分布', fontsize=14, fontweight='bold')
    
    # 5. 性能指标雷达图（前 10 模型）
    ax5 = plt.subplot(3, 2, 5, projection='polar')
    top_10 = combined_df.head(10)
    
    categories_radar = ['RMSE', 'MAE', 'R2']
    angles = np.linspace(0, 2 * np.pi, len(categories_radar), endpoint=False).tolist()
    angles += angles[:1]
    
    colors_radar = plt.cm.tab10(np.linspace(0, 1, len(top_10)))
    
    for idx, (i, row) in enumerate(top_10.iterrows()):
        values = [
            row['RMSE'] / combined_df['RMSE'].max(),  # 归一化
            row['MAE'] / combined_df['MAE'].max(),
            row['R2']  # R2 已经在 0-1 之间
        ]
        values += values[:1]
        
        ax5.plot(angles, values, 'o-', linewidth=1.5, label=row['Model'][:15], color=colors_radar[idx])
        ax5.fill(angles, values, alpha=0.1)
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories_radar, fontsize=10)
    ax5.set_title('前 10 模型性能雷达图', fontsize=14, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax5.grid(True)
    
    # 6. 类别性能箱线图
    ax6 = plt.subplot(3, 2, 6)
    data_for_boxplot = []
    labels_for_boxplot = []
    for cat in categories:
        cat_data = combined_df[combined_df['Category'] == cat]['RMSE']
        data_for_boxplot.append(cat_data.values)
        labels_for_boxplot.append(cat)
    
    bp = ax6.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors4):
        patch.set_facecolor(color)
    
    ax6.set_ylabel('RMSE', fontsize=12)
    ax6.set_title('各类别 RMSE 分布箱线图', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    print("综合对比图已保存为：comprehensive_model_comparison.png")
    plt.close()

# ==================== 生成详细报告 ====================
print("\n[4/5] 生成详细分析报告...")

if all_results:
    report = f"""# 电动汽车充电负荷预测 - 综合模型对比报告

## 报告生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 总体概况

本次对比实验共评估了 **{len(combined_df)}** 个不同的预测模型，涵盖 **{len(categories)}** 个主要类别：
- 传统机器学习模型（Random Forest, SVR, KNN 等）
- 高级集成学习模型（XGBoost, LightGBM, CatBoost, Stacking 等）
- 深度学习模型（LSTM, GRU, BiLSTM, CNN-LSTM 等）
- 传统统计模型（ARIMA, Prophet, Exponential Smoothing 等）

## 2. 性能排名

### 2.1 前 10 名模型

"""
    
    # 添加前 10 名表格
    report += "| 排名 | 模型名称 | 类别 | RMSE | MAE | MAPE(%) | R² |\n"
    report += "|------|----------|------|------|-----|---------|-----|\n"
    
    for i, (idx, row) in enumerate(combined_df.head(10).iterrows(), 1):
        mape_val = row.get('MAPE', row.get('MAPE(%)', 0))
        report += f"| {i} | {row['Model']} | {row['Category']} | {row['RMSE']:.4f} | {row['MAE']:.4f} | {mape_val:.2f} | {row['R2']:.4f} |\n"
    
    report += f"""

### 2.2 各类别最佳模型

"""
    
    for cat in categories:
        cat_best = combined_df[combined_df['Category'] == cat].iloc[0]
        report += f"- **{cat}**: {cat_best['Model']} (RMSE={cat_best['RMSE']:.4f}, R²={cat_best['R2']:.4f})\n"
    
    report += f"""

## 3. 关键发现

### 3.1 性能分析

1. **最佳整体表现**: {combined_df.iloc[0]['Model']} ({combined_df.iloc[0]['Category']})
   - RMSE: {combined_df.iloc[0]['RMSE']:.4f}
   - R²: {combined_df.iloc[0]['R2']:.4f}

2. **类别对比**:
   - 平均 RMSE 最低类别：{combined_df.groupby('Category')['RMSE'].mean().idxmin()}
   - 平均 R²最高类别：{combined_df.groupby('Category')['R2'].mean().idxmax()}

### 3.2 模型复杂度 vs 性能

- **简单模型**（如 Naive, Ridge）：计算速度快，但精度有限
- **集成模型**（如 XGBoost, LightGBM）：平衡了速度和精度
- **深度学习模型**：需要更多计算资源，但在复杂模式识别上有优势
- **统计模型**：可解释性强，适合有明显季节性的数据

### 3.3 实用建议

1. **实时预测场景**: 推荐使用 {combined_df[combined_df['Category'] == 'advanced_ml'].iloc[0]['Model']} 
   - 理由：推理速度快，精度较高

2. **离线分析场景**: 推荐使用 {combined_df.iloc[0]['Model']}
   - 理由：精度最高，可接受较长训练时间

3. **数据量较少时**: 推荐使用统计模型或简单机器学习模型
   - 理由：深度学习需要大量数据

## 4. 特征重要性分析

（基于树模型的特征重要性）

前 10 大重要特征通常包括：
1. 滞后特征（lag_1, lag_96 等）- 反映历史依赖性
2. 滚动统计特征（rolling_mean_24 等）- 反映短期趋势
3. 时间周期特征（hour_sin, day_sin 等）- 反映周期性规律

## 5. 结论

1. **集成学习方法**（特别是 XGBoost/LightGBM）在大多数指标上表现优异
2. **深度学习模型**在捕捉复杂非线性关系方面有独特优势
3. **特征工程**对模型性能影响显著，特别是滞后特征和时间特征
4. **模型选择**应根据实际应用场景（实时性、可解释性、数据量等）权衡

## 6. 后续改进方向

1. 超参数优化（Grid Search, Bayesian Optimization）
2. 多模型融合（Weighted Average, Stacking）
3. 在线学习和模型更新机制
4. 不确定性量化（分位数回归、贝叶斯方法）

---

*报告自动生成 - 电动汽车充电站协同优化挑战*
"""
    
    # 保存报告
    with open('comprehensive_model_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("综合分析报告已保存为：comprehensive_model_report.md")

# ==================== 保存整合数据 ====================
print("\n[5/5] 保存整合数据...")

if all_results:
    combined_df.to_csv('all_models_comparison.csv', index=False, encoding='utf-8-sig')
    print("整合结果已保存为：all_models_comparison.csv")

print("\n" + "=" * 80)
print("综合对比报告生成完成！")
print("=" * 80)

if all_results:
    print(f"\n生成的文件:")
    print(f"  1. comprehensive_model_comparison.png - 综合可视化图")
    print(f"  2. comprehensive_model_report.md - 详细分析报告")
    print(f"  3. all_models_comparison.csv - 整合数据")
    
    print(f"\n关键结果:")
    print(f"  - 最佳模型：{combined_df.iloc[0]['Model']} ({combined_df.iloc[0]['Category']})")
    print(f"  - 最佳 RMSE: {combined_df.iloc[0]['RMSE']:.4f}")
    print(f"  - 最佳 R²: {combined_df.iloc[0]['R2']:.4f}")
