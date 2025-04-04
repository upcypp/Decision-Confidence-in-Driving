#方向盘运动学图
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 参与者数据路径列表
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

# 读取所有实验数据并为每个文件添加参与者编号
dataframes = []
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    df['Participant_ID'] = i + 1
    dataframes.append(df)

# 合并所有参与者的数据
combined_df = pd.concat(dataframes, ignore_index=True)

# 方向盘数据路径列表
steering_paths = [
    "../data/2024103001/steering_wheel_log_pre.csv",
    "../data/2024110701/steering_wheel_log_pre.csv",
    "../data/2024111801/steering_wheel_log_pre.csv",
    "../data/2024111803/steering_wheel_log_pre.csv",
    "../data/2024111901/steering_wheel_log_pre.csv",
    "../data/2024111902/steering_wheel_log_pre.csv",
    "../data/2024111903/steering_wheel_log_pre.csv",
    "../data/2024112101/steering_wheel_log_pre.csv",
    "../data/2024112102/steering_wheel_log_pre.csv",
    "../data/2024112201/steering_wheel_log_pre.csv",
    "../data/2024112202/steering_wheel_log_pre.csv",
    "../data/2024112301/steering_wheel_log_pre.csv",
    "../data/2024112302/steering_wheel_log_pre.csv",
    "../data/2024112801/steering_wheel_log_pre.csv",
    "../data/2024112901/steering_wheel_log_pre.csv",
    "../data/2024120101/steering_wheel_log_pre.csv",
    "../data/2024120102/steering_wheel_log_pre.csv",
    "../data/2024120201/steering_wheel_log_pre.csv",
    "../data/2024120202/steering_wheel_log_pre.csv",
    "../data/2024120203/steering_wheel_log_pre.csv",
    "../data/2024120301/steering_wheel_log_pre.csv",
    "../data/2024120303/steering_wheel_log_pre.csv",
    "../data/2024120401/steering_wheel_log_pre.csv",
    "../data/2024120403/steering_wheel_log_pre.csv",
    "../data/2024120501/steering_wheel_log_pre.csv",
    "../data/2024120801/steering_wheel_log_pre.csv",
    "../data/2024120901/steering_wheel_log_pre.csv",
    "../data/2025010301/steering_wheel_log_pre.csv",
    "../data/2025010401/steering_wheel_log_pre.csv",
    "../data/2025010402/steering_wheel_log_pre.csv",
    "../data/2025010403/steering_wheel_log_pre.csv",
    "../data/2025010405/steering_wheel_log_pre.csv",
    "../data/2025010601/steering_wheel_log_pre.csv",
]

# 读取方向盘数据
steering_dataframes = []
for i, file_path in enumerate(steering_paths):
    steering_df = pd.read_csv(file_path)
    steering_df['Participant_ID'] = i + 1
    steering_dataframes.append(steering_df)

# 合并所有方向盘数据
combined_steering_df = pd.concat(steering_dataframes, ignore_index=True)

# 转换时间格式
combined_df['Throttle_Time'] = pd.to_datetime(combined_df['Throttle_Time'], unit='ms')
combined_steering_df['Timestamp'] = pd.to_datetime(combined_steering_df['Timestamp'], unit='ms')

# 按时间戳排序
combined_df = combined_df.sort_values('Throttle_Time')
combined_steering_df = combined_steering_df.sort_values('Timestamp')

# 通过时间戳对齐数据
merged_df = pd.merge_asof(
    combined_steering_df, combined_df,
    left_on='Timestamp', right_on='Throttle_Time',
    by='Participant_ID', direction='backward'
)

# 筛选Throttle_Time到Throttle_Time + 5000ms的数据
merged_df['End_Throttle_Time'] = merged_df['Throttle_Time'] + pd.Timedelta(milliseconds=5000)
filtered_df = merged_df[
    (merged_df['Timestamp'] >= merged_df['Throttle_Time']) &
    (merged_df['Timestamp'] <= merged_df['End_Throttle_Time'])
].copy()

# 创建时间点从0秒到5秒
time_points = np.linspace(0, 5, 50)  # 从0到5秒，使用50个点

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))

# 定义颜色
colors = ['green', 'orange', 'blue']  # Low: green, Intermediate: orange, High: blue

# 计算标准误差函数，处理缺失值
def calculate_se(data):
    valid_data = data.dropna()  # 移除 NaN
    if len(valid_data) > 1:  # 确保至少有2个有效值计算标准差
        return valid_data.std() / np.sqrt(len(valid_data))
    return np.nan  # 如果样本量不足，返回 NaN

# 1和2合并为Low Confidence
low_Confidence_df = filtered_df[filtered_df['Confidence'].isin([1, 2])]
participant_lines_low = []
for participant_id, group in low_Confidence_df.groupby('Participant_ID'):
    steering_values = []
    for t in time_points:
        target_time = group['Throttle_Time'].iloc[0] + pd.Timedelta(seconds=t)
        closest_row = group.iloc[(group['Timestamp'] - target_time).abs().argsort()[:1]]
        steering_values.append(closest_row['Steering'].values[0] if not closest_row.empty else np.nan)
    participant_lines_low.append(steering_values)
if participant_lines_low:
    steering_df_low = pd.DataFrame(participant_lines_low, columns=time_points)
    mean_steering_low = steering_df_low.mean() * 90
    se_low = steering_df_low.apply(calculate_se) * 90
    ax.plot(time_points, mean_steering_low, linestyle='-', linewidth=4, color=colors[0], label="Low Confidence")
    ax.fill_between(time_points, mean_steering_low - se_low, mean_steering_low + se_low, color=colors[0], alpha=0.2)

# 3为Intermediate Confidence
intermediate_Confidence_df = filtered_df[filtered_df['Confidence'] == 3]
participant_lines_intermediate = []
for participant_id, group in intermediate_Confidence_df.groupby('Participant_ID'):
    steering_values = []
    for t in time_points:
        target_time = group['Throttle_Time'].iloc[0] + pd.Timedelta(seconds=t)
        closest_row = group.iloc[(group['Timestamp'] - target_time).abs().argsort()[:1]]
        steering_values.append(closest_row['Steering'].values[0] if not closest_row.empty else np.nan)
    participant_lines_intermediate.append(steering_values)
if participant_lines_intermediate:
    steering_df_intermediate = pd.DataFrame(participant_lines_intermediate, columns=time_points)
    mean_steering_intermediate = steering_df_intermediate.mean() * 90
    se_intermediate = steering_df_intermediate.apply(calculate_se) * 90
    ax.plot(time_points, mean_steering_intermediate, linestyle='-', linewidth=4, color=colors[1], label="Intermediate Confidence")
    ax.fill_between(time_points, mean_steering_intermediate - se_intermediate, mean_steering_intermediate + se_intermediate, color=colors[1], alpha=0.2)

# 4和5合并为High Confidence
high_Confidence_df = filtered_df[filtered_df['Confidence'].isin([4, 5])]
participant_lines_high = []
for participant_id, group in high_Confidence_df.groupby('Participant_ID'):
    steering_values = []
    for t in time_points:
        target_time = group['Throttle_Time'].iloc[0] + pd.Timedelta(seconds=t)
        closest_row = group.iloc[(group['Timestamp'] - target_time).abs().argsort()[:1]]
        steering_values.append(closest_row['Steering'].values[0] if not closest_row.empty else np.nan)
    participant_lines_high.append(steering_values)
if participant_lines_high:
    steering_df_high = pd.DataFrame(participant_lines_high, columns=time_points)
    mean_steering_high = steering_df_high.mean() * 90
    se_high = steering_df_high.apply(calculate_se) * 90
    ax.plot(time_points, mean_steering_high, linestyle='-', linewidth=4, color=colors[2], label="High Confidence")
    ax.fill_between(time_points, mean_steering_high - se_high, mean_steering_high + se_high, color=colors[2], alpha=0.2)

# 设置图表标题和标签
ax.tick_params(axis='x', labelsize=28)  # 调整 X 轴刻度字体大小
ax.tick_params(axis='y', labelsize=28)  # 调整 Y 轴刻度字体大小
ax.set_xlabel("Time(s)", fontsize=34)
ax.set_ylabel("Steering Angle(°)", fontsize=34)
# ax.legend(fontsize=16, loc='lower right')


# 设置 Y 轴刻度仅显示 0.0, 0.1, 0.2, 0.3
ax.set_yticks([-30, -20, -10, 0])
# 保存为SVG格式到指定位置
plt.tight_layout()
# 设置保存路径
save_dir = "../output/"
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
plt.savefig(os.path.join(save_dir, "Steering_2.svg"), format="svg", bbox_inches='tight')

plt.show()