import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator  # 导入MaxNLocator

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

# 车辆数据路径列表
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

# 读取车辆数据
vehicle_dataframes = []
for i, file_path in enumerate(vehicle_paths):
    vehicle_df = pd.read_csv(file_path)
    vehicle_df['Participant_ID'] = i + 1
    vehicle_dataframes.append(vehicle_df)

# 合并所有车辆数据
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
time_points = np.linspace(0, 5, 50)  # 从0到5秒，使用50个点

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))

# 定义颜色
colors = ['green', 'orange', 'blue']

# 定义信心等级分组
confidence_groups = {
    'Low': [1, 2],
    'Medium': [3],
    'High': [4, 5]
}

# 画 低、中、高信心的平均位置坐标曲线
for idx, (group_name, confidence_range) in enumerate(confidence_groups.items()):
    confidence_df = filtered_df[filtered_df['Confidence'].isin(confidence_range)]

    participant_lines_x = []
    participant_lines_y = []

    for participant_id, group in confidence_df.groupby('Participant_ID'):
        location_x_values = []
        location_y_values = []
        for t in time_points:
            target_time = group['Throttle_Time'].iloc[0] + pd.Timedelta(seconds=t)
            closest_row = group.iloc[(group['Timestamp'] - target_time).abs().argsort()[:1]]
            location_x_values.append(closest_row['Hero_Location_X'].values[0] if not closest_row.empty else np.nan)
            location_y_values.append(closest_row['Hero_Location_Y'].values[0] if not closest_row.empty else np.nan)

        # 存储每个参与者的轨迹
        participant_lines_x.append(location_x_values)
        participant_lines_y.append(location_y_values)

    # 计算当前信心组的平均位置坐标
    if participant_lines_x and participant_lines_y:  # 确保有数据
        location_df_x = pd.DataFrame(participant_lines_x, columns=time_points)
        location_df_y = pd.DataFrame(participant_lines_y, columns=time_points)

        mean_location_x = location_df_x.mean()
        mean_location_y = location_df_y.mean()

        # 计算标准误差
        std_location_x = location_df_x.std() / np.sqrt(location_df_x.shape[0])  # 标准误差
        std_location_y = location_df_y.std() / np.sqrt(location_df_y.shape[0])  # 标准误差

        # 绘制标准误差的阴影区域（置信区间）
        ax.fill_between(-mean_location_y, -mean_location_x - std_location_x, -mean_location_x + std_location_x,
                        color=colors[idx], alpha=0.3)  # 阴影区域

        # 绘制该信心组的平均轨迹（彩色实线）
        ax.plot(-mean_location_y, -mean_location_x, linestyle='-', linewidth=4, color=colors[idx], label=group_name)

# 设置图表标题和标签
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlabel("Location X (m)", fontsize=34)
ax.set_ylabel("Location Y (m)", fontsize=34)

# 设置坐标轴标签为整数
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 设置X轴为整数
ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # 设置Y轴为整数

# 保存为SVG格式到指定位置
plt.tight_layout()
# 设置保存路径
save_dir = "../output/"
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
plt.savefig(os.path.join(save_dir, "location_1.svg"), format="svg", bbox_inches='tight')
plt.show()
