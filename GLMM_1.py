# Confidence_2
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm  # 用于计算正态分布的临界值

# Load the datasets (assuming all participants have similar structure)
file_paths = [
    "../data/2024103001/experiment_log_pre.csv",
    "../data/2024110701/experiment_log_pre.csv",
    "../data/2024111801/experiment_log_pre.csv",
    "../data/2024111803/experiment_log_pre.csv",
    "../data/2024111901/experiment_log_pre.csv",
    "../data/2024111902/experiment_log_pre.csv",
    "../data/2024111903/experiment_log_pre.csv",
    "../data/2024112101/experiment_log_pre.csv",
    "../data/2024112102/experiment_log_pre.csv",
    "../data/2024112201/experiment_log_pre.csv",
    "../data/2024112202/experiment_log_pre.csv",
    "../data/2024112301/experiment_log_pre.csv",
    "../data/2024112302/experiment_log_pre.csv",
    "../data/2024112801/experiment_log_pre.csv",
    "../data/2024112901/experiment_log_pre.csv",
    "../data/2024120101/experiment_log_pre.csv",
    "../data/2024120102/experiment_log_pre.csv",
    "../data/2024120201/experiment_log_pre.csv",
    "../data/2024120202/experiment_log_pre.csv",
    "../data/2024120203/experiment_log_pre.csv",
    "../data/2024120301/experiment_log_pre.csv",
    "../data/2024120303/experiment_log_pre.csv",
    "../data/2024120401/experiment_log_pre.csv",
    "../data/2024120403/experiment_log_pre.csv",
    "../data/2024120501/experiment_log_pre.csv",
    "../data/2024120801/experiment_log_pre.csv",
    "../data/2024120901/experiment_log_pre.csv",
    "../data/2025010301/experiment_log_pre.csv",
    "../data/2025010401/experiment_log_pre.csv",
    "../data/2025010402/experiment_log_pre.csv",
    "../data/2025010403/experiment_log_pre.csv",
    "../data/2025010405/experiment_log_pre.csv",
    "../data/2025010601/experiment_log_pre.csv",
]

# 读取数据并添加参与者编号
dataframes = []
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    df['Participant_ID'] = i + 1  # 为每个文件的所有行添加唯一的参与者编号
    dataframes.append(df)

# 合并所有参与者的数据
combined_data = pd.concat(dataframes, ignore_index=True)

# 确保保留相关列
combined_data = combined_data[['Participant_ID', 'Experiment_Number', 'Initial_Distance', 'TTA', 'Confidence_2']]

# 标准化自变量
combined_data['Initial_Distance'] = (combined_data['Initial_Distance'] - combined_data['Initial_Distance'].mean()) / combined_data['Initial_Distance'].std()
combined_data['TTA'] = (combined_data['TTA'] - combined_data['TTA'].mean()) / combined_data['TTA'].std()

# 将 Confidence_2 转换为分类变量并编码
combined_data['Confidence_2'] = pd.Categorical(combined_data['Confidence_2'], categories=[1, 2, 3, 4, 5], ordered=True)
combined_data['Confidence_2'] = combined_data['Confidence_2'].cat.codes + 1

# 定义模型公式
formula = "Confidence_2 ~ Initial_Distance + TTA"

# 使用基于参与者编号 (Participant_ID) 的随机效应
model = smf.mixedlm(formula, combined_data, groups=combined_data["Participant_ID"])

# 优化算法
result = model.fit(method='powell', maxiter=1000)


print(result.summary())

# -------------------- 生成可视化图表 --------------------
# 重新加载数据是因为之前的数据被标准化了，结果图显示的应为原始数据，所以重新加载一下
file_paths = [
    "../data/2024103001/experiment_log_pre.csv",
    "../data/2024110701/experiment_log_pre.csv",
    "../data/2024111801/experiment_log_pre.csv",
    "../data/2024111803/experiment_log_pre.csv",
    "../data/2024111901/experiment_log_pre.csv",
    "../data/2024111902/experiment_log_pre.csv",
    "../data/2024111903/experiment_log_pre.csv",
    "../data/2024112101/experiment_log_pre.csv",
    "../data/2024112102/experiment_log_pre.csv",
    "../data/2024112201/experiment_log_pre.csv",
    "../data/2024112202/experiment_log_pre.csv",
    "../data/2024112301/experiment_log_pre.csv",
    "../data/2024112302/experiment_log_pre.csv",
    "../data/2024112801/experiment_log_pre.csv",
    "../data/2024112901/experiment_log_pre.csv",
    "../data/2024120101/experiment_log_pre.csv",
    "../data/2024120102/experiment_log_pre.csv",
    "../data/2024120201/experiment_log_pre.csv",
    "../data/2024120202/experiment_log_pre.csv",
    "../data/2024120203/experiment_log_pre.csv",
    "../data/2024120301/experiment_log_pre.csv",
    "../data/2024120303/experiment_log_pre.csv",
    "../data/2024120401/experiment_log_pre.csv",
    "../data/2024120403/experiment_log_pre.csv",
    "../data/2024120501/experiment_log_pre.csv",
    "../data/2024120801/experiment_log_pre.csv",
    "../data/2024120901/experiment_log_pre.csv",
    "../data/2025010301/experiment_log_pre.csv",
    "../data/2025010401/experiment_log_pre.csv",
    "../data/2025010402/experiment_log_pre.csv",
    "../data/2025010403/experiment_log_pre.csv",
    "../data/2025010405/experiment_log_pre.csv",
    "../data/2025010601/experiment_log_pre.csv",
]
# 读取数据并添加参与者编号
dataframes = []
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    df['Participant_ID'] = i + 1  # 为每个文件的所有行添加唯一的参与者编号
    dataframes.append(df)

# 合并所有参与者的数据
combined_data = pd.concat(dataframes, ignore_index=True)

# 确保保留相关列
combined_data = combined_data[['Participant_ID', 'Experiment_Number', 'Initial_Distance', 'TTA', 'Confidence_2']]

# 确保 Initial_Distance 和 TTA 是分类变量
combined_data['Initial_Distance'] = combined_data['Initial_Distance'].astype(str)
combined_data['TTA'] = combined_data['TTA'].astype(str)

# 创建画布和子图
fig, axes = plt.subplots(1, len(combined_data['Initial_Distance'].unique()), figsize=(6, 6), sharey=True)

# 遍历每个初始距离的组
for i, initial_distance in enumerate(sorted(combined_data['Initial_Distance'].unique(), key=float)):
    # 过滤数据为特定 Initial_Distance
    subset = combined_data[combined_data['Initial_Distance'] == initial_distance]

    # 按 TTA 分组计算个体趋势
    individual_trends = subset.groupby(['Participant_ID', 'TTA'])['Confidence_2'].mean().reset_index()

    # 按 TTA 分组计算总体平均值及标准误差
    average_trend = subset.groupby('TTA')['Confidence_2'].agg(['mean', 'sem']).reset_index()
    # 计算 95% 置信区间的误差范围
    z_value = norm.ppf(0.975)  # 95% 置信水平的 z 值（双尾），约为 1.96
    average_trend['ci95'] = average_trend['sem'] * z_value  # CI 95% = SEM * 1.96

    # 绘制个体趋势虚线（去掉 label）
    for participant_id in individual_trends['Participant_ID'].unique():
        participant_data = individual_trends[individual_trends['Participant_ID'] == participant_id]
        axes[i].plot(participant_data['TTA'], participant_data['Confidence_2'], linestyle='--', color='black', alpha=0.2)  # 移除 label
        axes[i].scatter(participant_data['TTA'], participant_data['Confidence_2'], color='black', alpha=0.2, s=10)

    # 绘制总体趋势的实线和误差棒（使用 CI 95%）
    axes[i].errorbar(average_trend['TTA'], average_trend['mean'], yerr=average_trend['ci95'], fmt='o-',
                     color='black',
                     capsize=8, linewidth=2, markeredgewidth=1)

    # 设置子图标题和标签
    axes[i].set_title(f"{initial_distance}m", fontsize=22)
    if i == 1:  # 只在第二个子图中设置 x 轴标签
        axes[i].set_xlabel("TTA", fontsize=16)
    if i == 0:  # 只在第一个子图中设置 y 轴标签
        axes[i].set_ylabel("ΔConfidence", fontsize=16)

    axes[i].tick_params(axis='y', labelsize=22)
    axes[i].tick_params(axis='x', labelsize=22)

# 调整子图间距
plt.subplots_adjust(wspace=0.02)

# 优化布局
plt.tight_layout(pad=0.5, w_pad=0.1)

# 设置保存路径
save_dir = "../output/"
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
plt.savefig(os.path.join(save_dir, "linear_2.svg"), format="svg", bbox_inches='tight')


plt.show()

