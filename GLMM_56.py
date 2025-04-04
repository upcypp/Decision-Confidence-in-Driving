# 速度运动学图
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# [之前的读取数据部分保持不变，直到创建时间点]
# 读取实验数据
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

dataframes = []
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    df['Participant_ID'] = i + 1
    dataframes.append(df)
combined_df = pd.concat(dataframes, ignore_index=True)

# 读取车辆数据
vehicle_paths = [
    "../data/2024103001/vehicle_log_pre.csv",
    "../data/2024110701/vehicle_log_pre.csv",
    "../data/2024111801/vehicle_log_pre.csv",
    "../data/2024111803/vehicle_log_pre.csv",
    "../data/2024111901/vehicle_log_pre.csv",
    "../data/2024111902/vehicle_log_pre.csv",
    "../data/2024111903/vehicle_log_pre.csv",
    "../data/2024112101/vehicle_log_pre.csv",
    "../data/2024112102/vehicle_log_pre.csv",
    "../data/2024112201/vehicle_log_pre.csv",
    "../data/2024112202/vehicle_log_pre.csv",
    "../data/2024112301/vehicle_log_pre.csv",
    "../data/2024112302/vehicle_log_pre.csv",
    "../data/2024112801/vehicle_log_pre.csv",
    "../data/2024112901/vehicle_log_pre.csv",
    "../data/2024120101/vehicle_log_pre.csv",
    "../data/2024120102/vehicle_log_pre.csv",
    "../data/2024120201/vehicle_log_pre.csv",
    "../data/2024120202/vehicle_log_pre.csv",
    "../data/2024120203/vehicle_log_pre.csv",
    "../data/2024120301/vehicle_log_pre.csv",
    "../data/2024120303/vehicle_log_pre.csv",
    "../data/2024120401/vehicle_log_pre.csv",
    "../data/2024120403/vehicle_log_pre.csv",
    "../data/2024120501/vehicle_log_pre.csv",
    "../data/2024120801/vehicle_log_pre.csv",
    "../data/2024120901/vehicle_log_pre.csv",
    "../data/2025010301/vehicle_log_pre.csv",
    "../data/2025010401/vehicle_log_pre.csv",
    "../data/2025010402/vehicle_log_pre.csv",
    "../data/2025010403/vehicle_log_pre.csv",
    "../data/2025010405/vehicle_log_pre.csv",
    "../data/2025010601/vehicle_log_pre.csv",
]

vehicle_dataframes = []
for i, file_path in enumerate(vehicle_paths):
    vehicle_df = pd.read_csv(file_path)
    vehicle_df['Participant_ID'] = i + 1
    vehicle_dataframes.append(vehicle_df)
combined_vehicle_df = pd.concat(vehicle_dataframes, ignore_index=True)

# 转换时间格式
combined_df['Throttle_Time'] = pd.to_datetime(combined_df['Throttle_Time'], unit='ms')
combined_vehicle_df['Timestamp'] = pd.to_datetime(combined_vehicle_df['Timestamp'], unit='ms')

# 按时间戳排序
combined_df = combined_df.sort_values('Throttle_Time')
combined_vehicle_df = combined_vehicle_df.sort_values('Timestamp')

# 通过时间戳对齐数据
merged_df = pd.merge_asof(
    combined_vehicle_df, combined_df,
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
time_points = np.linspace(0, 5, 50)  # 细化时间粒度，生成50个时间点

fig, ax = plt.subplots(figsize=(10, 6))

# 定义颜色
colors = ['blue', 'orange', 'green']

# 计算标准误差函数
def calculate_se(data):
    return data.std() / np.sqrt(len(data))

# High Confidence_2 (信心4和信心5的平均值)
high_Confidence_2_df = filtered_df[filtered_df['Confidence_2'].isin([4, 5])]
participant_lines_high = []
for participant_id, group in high_Confidence_2_df.groupby('Participant_ID'):
    speed_values = []
    for t in time_points:
        target_time = group['Throttle_Time'].iloc[0] + pd.Timedelta(seconds=t)
        closest_row = group.iloc[(group['Timestamp'] - target_time).abs().argsort()[:1]]
        speed_values.append(closest_row['Hero_Speed_kmh'].values[0] if not closest_row.empty else np.nan)
    participant_lines_high.append(speed_values)
if participant_lines_high:
    speed_df_high = pd.DataFrame(participant_lines_high, columns=time_points)
    mean_speed_high = speed_df_high.mean()
    se_high = speed_df_high.apply(calculate_se)
    ax.plot(time_points, mean_speed_high, linestyle='-', linewidth=4, color=colors[0], label='High Confidence_2')
    ax.fill_between(time_points, mean_speed_high - se_high, mean_speed_high + se_high, color=colors[0], alpha=0.2)

# Intermediate Confidence_2 (信心3)
intermediate_Confidence_2_df = filtered_df[filtered_df['Confidence_2'] == 3]
participant_lines_intermediate = []
for participant_id, group in intermediate_Confidence_2_df.groupby('Participant_ID'):
    speed_values = []
    for t in time_points:
        target_time = group['Throttle_Time'].iloc[0] + pd.Timedelta(seconds=t)
        closest_row = group.iloc[(group['Timestamp'] - target_time).abs().argsort()[:1]]
        speed_values.append(closest_row['Hero_Speed_kmh'].values[0] if not closest_row.empty else np.nan)
    participant_lines_intermediate.append(speed_values)
if participant_lines_intermediate:
    speed_df_intermediate = pd.DataFrame(participant_lines_intermediate, columns=time_points)
    mean_speed_intermediate = speed_df_intermediate.mean()
    se_intermediate = speed_df_intermediate.apply(calculate_se)
    ax.plot(time_points, mean_speed_intermediate, linestyle='-', linewidth=4, color=colors[1], label='Intermediate Confidence_2')
    ax.fill_between(time_points, mean_speed_intermediate - se_intermediate, mean_speed_intermediate + se_intermediate, color=colors[1], alpha=0.2)

# Low Confidence_2 (信心1和信心2的平均值)
low_Confidence_2_df = filtered_df[filtered_df['Confidence_2'].isin([1, 2])]
participant_lines_low = []
for participant_id, group in low_Confidence_2_df.groupby('Participant_ID'):
    speed_values = []
    for t in time_points:
        target_time = group['Throttle_Time'].iloc[0] + pd.Timedelta(seconds=t)
        closest_row = group.iloc[(group['Timestamp'] - target_time).abs().argsort()[:1]]
        speed_values.append(closest_row['Hero_Speed_kmh'].values[0] if not closest_row.empty else np.nan)
    participant_lines_low.append(speed_values)
if participant_lines_low:
    speed_df_low = pd.DataFrame(participant_lines_low, columns=time_points)
    mean_speed_low = speed_df_low.mean()
    se_low = speed_df_low.apply(calculate_se)
    ax.plot(time_points, mean_speed_low, linestyle='-', linewidth=4, color=colors[2], label='Low Confidence_2')
    ax.fill_between(time_points, mean_speed_low - se_low, mean_speed_low + se_low, color=colors[2], alpha=0.2)

# 设置图表标题和标签
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlabel("Time(s)", fontsize=34)
ax.set_ylabel("Speed(Km/h)", fontsize=34)
# ax.legend(fontsize=14, loc='upper left')

# 保存为SVG格式到指定位置
plt.tight_layout()
# 设置保存路径
save_dir = "../output/"
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
plt.savefig(os.path.join(save_dir, "speed_2.svg"), format="svg", bbox_inches='tight')


plt.show()