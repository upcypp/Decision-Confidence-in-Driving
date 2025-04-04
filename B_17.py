# 元认知效率计算--信心1

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# 数据路径列表（保持不变）
participant_paths = [
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
    "../data/2024112902/experiment_log_pre.csv",
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

# 存储每个参与者的元认知效率
mce_scores = {}

# 遍历每个文件
for path in participant_paths:
    try:
        # 读取 CSV 文件
        df = pd.read_csv(path)
        print(f"成功读取文件: {path}")
        print(f"文件行数: {len(df)}")
        print(f"列名: {list(df.columns)}")

        # 确保所需列存在
        required_columns = ['Confidence', 'Initial_Distance', 'TTA', 'Collision']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"警告: 文件 {path} 中缺少列 {missing_columns}，已跳过")
            continue

        # 去除 NaN 值
        df_plot = df.dropna(subset=required_columns)

        # 获取参与者ID
        participant_id = os.path.basename(os.path.dirname(path))

        # 准备特征和目标变量
        X = df_plot[['TTA', 'Initial_Distance', 'Collision']].values  # 未标准化，使用原始值
        y = df_plot['Confidence'].values

        # 使用线性回归拟合 Confidence
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)

        # 计算 R²（决定系数）
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0  # 防止除以零

        # 修改部分：计算信心趋向较大值的倾向性 (confidence_tendency)
        confidence_values = pd.Series(y)
        max_confidence = confidence_values.max()  # 信心的最大值
        if max_confidence > 0:  # 防止除以零
            # 计算信心值相对于最大值的平均比例，值越接近1说明越趋向较大值
            confidence_tendency = confidence_values.mean() / max_confidence
        else:
            confidence_tendency = 0  # 如果最大值为0，则倾向性为0

        # 计算 MCE：结合信心倾向性和 R²
        mce = 100 * (0.8 * confidence_tendency + 0.2 * r2)  # 倾向性权重 0.4，R² 权重 0.6
        mce_scores[participant_id] = mce

        # 创建散点图
        plt.figure(figsize=(8, 6))
        coord_counts = Counter(zip(df_plot['Confidence'], df_plot['TTA']))
        x_coords = df_plot['Confidence']
        y_coords = df_plot['TTA']
        plt.scatter(x_coords, y_coords, alpha=0.5, color='blue')
        for (x, y), count in coord_counts.items():
            if count > 1:
                plt.text(x, y, str(count), fontsize=10, ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        plt.xlabel('Confidence')
        plt.ylabel('TTA')
        plt.title(f'Scatter Plot of Confidence vs TTA\nParticipant ID: {participant_id}\nMean MCE: {mce:.2f}/100')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(0, 5.5)
        # plt.show()

        print(f"已为 Participant ID: {participant_id} 生成散点图，数据点数: {len(df_plot)}")
        print(f"参与者 {participant_id} 的平均元认知效率 (MCE): {mce:.2f}/100")

    except FileNotFoundError:
        print(f"错误: 文件 {path} 不存在，已跳过")
    except Exception as e:
        print(f"错误: 处理文件 {path} 时发生异常 - {str(e)}")

# 打印总结信息
print("\n=== 绘图和元认知效率计算完成 ===")
print("每个参与者的平均元认知效率 (MCE，满分 100):")
for pid, mce in sorted(mce_scores.items()):
    print(f"Participant ID: {pid}, Mean MCE: {mce:.2f}/100")