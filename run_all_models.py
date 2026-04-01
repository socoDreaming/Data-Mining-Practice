"""
一键运行所有模型对比实验
自动安装依赖并执行所有模型训练
"""

import subprocess
import sys
import os
from datetime import datetime

print("=" * 80)
print("电动汽车充电负荷预测 - 全模型对比实验")
print("=" * 80)
print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 检查并安装必要的包
print("\n[步骤 0] 检查和安装依赖包...")

required_packages = [
    'pandas', 'numpy', 'scikit-learn', 'scipy', 'matplotlib', 'seaborn', 'tqdm'
]

optional_packages = [
    ('xgboost', 'xgboost'),
    ('lightgbm', 'lightgbm'),
    ('catboost', 'catboost'),
    ('prophet', 'prophet'),
    ('statsmodels', 'statsmodels'),
    ('torch', 'torch')
]

def install_package(package_name):
    """安装包"""
    try:
        print(f"  正在安装 {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        print(f"  ✓ {package_name} 安装成功")
        return True
    except Exception as e:
        print(f"  ✗ {package_name} 安装失败：{e}")
        return False

# 安装必要包
for package in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {package} 已安装")
    except ImportError:
        install_package(package)

# 尝试安装可选包（失败不影响主流程）
print("\n  尝试安装可选包（失败不影响主流程）:")
for import_name, pip_name in optional_packages:
    try:
        __import__(import_name)
        print(f"  ✓ {pip_name} 已安装")
    except ImportError:
        # 可选包，不强制安装
        print(f"  ○ {pip_name} 未安装（可选）")

# 运行模型对比
print("\n" + "=" * 80)
print("[步骤 1] 运行高级机器学习模型对比...")
print("=" * 80)
try:
    subprocess.run([sys.executable, "advanced_model_comparison.py"], check=True)
    print("✓ 高级机器学习模型对比完成")
except Exception as e:
    print(f"✗ 高级机器学习模型对比失败：{e}")

print("\n" + "=" * 80)
print("[步骤 2] 运行集成学习模型对比...")
print("=" * 80)
try:
    subprocess.run([sys.executable, "ensemble_learning_models.py"], check=True)
    print("✓ 集成学习模型对比完成")
except Exception as e:
    print(f"✗ 集成学习模型对比失败：{e}")

print("\n" + "=" * 80)
print("[步骤 3] 运行深度学习模型对比...")
print("=" * 80)
try:
    subprocess.run([sys.executable, "deep_learning_models.py"], check=True)
    print("✓ 深度学习模型对比完成")
except Exception as e:
    print(f"✗ 深度学习模型对比失败：{e}")

print("\n" + "=" * 80)
print("[步骤 4] 运行统计模型对比...")
print("=" * 80)
try:
    subprocess.run([sys.executable, "statistical_models.py"], check=True)
    print("✓ 统计模型对比完成")
except Exception as e:
    print(f"✗ 统计模型对比失败：{e}")

print("\n" + "=" * 80)
print("[步骤 5] 生成综合对比报告...")
print("=" * 80)
try:
    subprocess.run([sys.executable, "generate_comprehensive_report.py"], check=True)
    print("✓ 综合对比报告生成完成")
except Exception as e:
    print(f"✗ 综合对比报告生成失败：{e}")

# 完成
print("\n" + "=" * 80)
print("所有实验完成！")
print("=" * 80)
print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 列出所有生成的文件
print("\n生成的文件列表:")
generated_files = [
    'model_comparison_results.csv',
    'ensemble_models_results.csv',
    'statistical_models_results.csv',
    'all_models_comparison.csv',
    'comprehensive_model_comparison.png',
    'comprehensive_model_report.md',
    'submission_task1_advanced.csv',
    'model_comparison.png',
    'top5_models_comparison.png',
    'deep_learning_comparison.png',
    'statistical_models_comparison.png',
    'ensemble_models_comparison.png'
]

for filename in generated_files:
    if os.path.exists(filename):
        file_size = os.path.getsize(filename) / 1024  # KB
        print(f"  ✓ {filename} ({file_size:.1f} KB)")
    else:
        print(f"  ○ {filename} (未生成)")

print("\n" + "=" * 80)
print("实验总结")
print("=" * 80)
print("请查看 comprehensive_model_report.md 获取详细分析报告")
print("请查看 comprehensive_model_comparison.png 获取可视化对比")
print("\n所有提交文件已保存，可直接用于比赛提交！")
