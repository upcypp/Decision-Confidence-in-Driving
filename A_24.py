#信心1-12个小提琴

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
from scipy.stats import differential_entropy
from scipy.signal import butter, filtfilt
import random
import torch
from torch.backends import cudnn
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# 固定随机种子函数
def setup_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        cudnn.enabled = False

rs = 42
setup_seed(rs)

# 参与者路径（保持不变）
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

# 参数设置（保持不变）
WINDOW_SIZE = 1000
FIXED_WINDOWS = 5
FS = 500
FEATURE_SIZES = {
    'driving': 35 + 7,
    'eye': 5 + 8,
    'eeg': 24
}
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._split')

# 窗口处理函数和特征提取函数（保持不变）
def process_window(data, window_size=WINDOW_SIZE, columns=None):
    timestamps = data['Timestamp'].values
    if columns is None:
        columns = [col for col in data.columns if col != 'Timestamp']
    windowed_data = []
    for col in columns:
        values = data[col].values if col in data.columns else np.zeros(len(timestamps))
        window_starts = np.arange(timestamps[0], timestamps[-1], window_size)
        window_values = []
        for ws in window_starts:
            we = ws + window_size
            mask = (timestamps >= ws) & (timestamps < we)
            if np.any(mask):
                idx = np.where(mask)[0][0]
                window_values.append(values[idx])
            else:
                window_values.append(0)
        windowed_data.append(window_values)
    return np.array(windowed_data).T

def extract_driving_features_raw(steering_data, vehicle_data):
    steering_angles = steering_data['Steering'].values
    location_x = vehicle_data['Hero_Location_X'].values
    location_y = vehicle_data['Hero_Location_Y'].values
    speed = vehicle_data['Hero_Speed_kmh'].values
    steering_std = np.std(steering_angles) if len(steering_angles) > 1 else 0
    location_x_std = np.std(location_x) if len(location_x) > 1 else 0
    location_y_std = np.std(location_y) if len(location_y) > 1 else 0
    speed_max = np.max(speed) if len(speed) > 0 else 0
    steering_max = np.max(steering_angles) if len(steering_angles) > 0 else 0
    location_x_max = np.max(location_x) if len(location_x) > 0 else 0
    location_y_max = np.max(location_y) if len(location_y) > 0 else 0
    return [steering_std, location_x_std, location_y_std, speed_max, steering_max, location_x_max, location_y_max]

def extract_eye_features_raw(pupil_data, blink_data, start_time, end_time):
    pupil_diameters = pupil_data['diameter'].values
    relevant_blinks = blink_data[
        (blink_data['start_timestamp'] >= start_time) &
        (blink_data['start_timestamp'] <= end_time)
    ]
    pupil_mean = np.mean(pupil_diameters) if len(pupil_diameters) > 0 else 0
    pupil_std = np.std(pupil_diameters) if len(pupil_diameters) > 1 else 0
    pupil_max = np.max(pupil_diameters) if len(pupil_diameters) > 0 else 0
    pupil_min = np.min(pupil_diameters) if len(pupil_diameters) > 0 else 0
    blink_durations = relevant_blinks['duration'].values
    blink_count = len(relevant_blinks)
    total_duration = (end_time - start_time) / 1000
    blink_frequency = blink_count / total_duration if total_duration > 0 else 0
    blink_mean_duration = np.mean(blink_durations) if len(blink_durations) > 0 else 0
    blink_std_duration = np.std(blink_durations) if len(blink_durations) > 1 else 0
    return [pupil_mean, pupil_std, pupil_max, pupil_min,
            blink_count, blink_frequency, blink_mean_duration, blink_std_duration]

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

def extract_eeg_features(eeg_raw, fs=FS):
    eeg_features = []
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14)}
    for channel in range(8):
        channel_data = eeg_raw[channel]
        if np.std(channel_data) == 0:
            de_delta = de_theta = de_alpha = 0
        else:
            delta_data = apply_bandpass_filter(channel_data, bands['delta'][0], bands['delta'][1], fs)
            theta_data = apply_bandpass_filter(channel_data, bands['theta'][0], bands['theta'][1], fs)
            alpha_data = apply_bandpass_filter(channel_data, bands['alpha'][0], bands['alpha'][1], fs)
            de_delta = differential_entropy(delta_data) if np.std(delta_data) > 0 else 0
            de_theta = differential_entropy(theta_data) if np.std(theta_data) > 0 else 0
            de_alpha = differential_entropy(alpha_data) if np.std(alpha_data) > 0 else 0
        eeg_features.extend([de_delta, de_theta, de_alpha])
    return eeg_features

# 定义模型（保持不变）
models = {
    'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    'NB': GaussianNB(),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'DT': DecisionTreeClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0)
}

feature_combinations = ['驾驶+眼动+EEG']
results = {model: {'驾驶+眼动+EEG': {'accuracies': [], 'f1_scores': [], 'avg_accuracies': [], 'avg_f1_scores': []}} for model in models}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 数据处理和模型训练
for path_idx, path in enumerate(participant_paths):
    base_dir = os.path.dirname(path)
    participant_id = os.path.basename(base_dir)
    steering_path = os.path.join(base_dir, 'steering_wheel_log_pre.csv')
    vehicle_path = os.path.join(base_dir, 'vehicle_log_pre.csv')
    eeg_path = os.path.join(base_dir, 'eeg.easy')
    pupil_path = os.path.join(base_dir, 'pupil_positions_pre.csv')
    blink_path = os.path.join(base_dir, 'blinks_pre.csv')

    experiment_log = pd.read_csv(path)
    steering_log = pd.read_csv(steering_path)
    vehicle_log = pd.read_csv(vehicle_path)
    eeg_columns = [f'Electrode_{i + 1}' for i in range(8)] + ['Extra_1', 'Extra_2', 'Extra_3', 'Extra_4', 'Timestamp']
    eeg_log = pd.read_csv(eeg_path, header=None, names=eeg_columns, delimiter='\t')
    pupil_log = pd.read_csv(pupil_path)
    blink_log = pd.read_csv(blink_path)

    X_driving, X_eye, X_eeg, y = [], [], [], []

    for index, row in experiment_log.iterrows():
        start_time = row['Towards_Spawn_Time']
        end_time = row['Key_Press_Time']
        confidence = row['Confidence']

        steering_data = steering_log[
            (steering_log['Timestamp'] >= start_time) &
            (steering_log['Timestamp'] <= end_time)
        ][['Timestamp', 'Steering', 'Throttle', 'Brake']]

        vehicle_data = vehicle_log[
            (vehicle_log['Timestamp'] >= start_time) &
            (vehicle_log['Timestamp'] <= end_time)
        ][['Timestamp', 'Hero_Location_X', 'Hero_Location_Y', 'Hero_Speed_kmh', 'Hero_Acceleration_kmh2']]

        eeg_data = eeg_log[
            (eeg_log['Timestamp'] >= start_time) &
            (eeg_log['Timestamp'] <= end_time)
        ][[f'Electrode_{i + 1}' for i in range(8)] + ['Timestamp']]

        pupil_data = pupil_log[
            (pupil_log['pupil_timestamp'] >= start_time) &
            (pupil_log['pupil_timestamp'] <= end_time)
        ][['pupil_timestamp', 'diameter']]

        blink_data = blink_log[
            (blink_log['start_timestamp'] <= end_time) &
            (blink_log['end_timestamp'] >= start_time)
        ]

        if not steering_data.empty and not vehicle_data.empty and not eeg_data.empty and not pupil_data.empty:
            steering_windowed = process_window(steering_data, columns=['Steering', 'Throttle', 'Brake'])
            vehicle_windowed = process_window(vehicle_data,
                                              columns=['Hero_Location_X', 'Hero_Location_Y', 'Hero_Speed_kmh',
                                                       'Hero_Acceleration_kmh2'])
            pupil_windowed = process_window(pupil_data.rename(columns={'pupil_timestamp': 'Timestamp'}),
                                            columns=['diameter'])

            min_rows = min(len(steering_windowed), len(vehicle_windowed), len(pupil_windowed))
            steering_windowed = steering_windowed[:min_rows]
            vehicle_windowed = vehicle_windowed[:min_rows]
            pupil_windowed = pupil_windowed[:min_rows]

            for data, dim, name in [(steering_windowed, 3, 'Steering'),
                                    (vehicle_windowed, 4, 'Vehicle'),
                                    (pupil_windowed, 1, 'Pupil')]:
                if len(data) < FIXED_WINDOWS:
                    padding = np.zeros((FIXED_WINDOWS - len(data), dim))
                    data = np.vstack((data, padding))
                else:
                    data = data[:FIXED_WINDOWS]
                if name == 'Steering':
                    steering_windowed = data
                elif name == 'Vehicle':
                    vehicle_windowed = data
                else:
                    pupil_windowed = data

            windowed_data = np.hstack((steering_windowed, vehicle_windowed))
            driving_raw_features = extract_driving_features_raw(steering_data, vehicle_data)
            driving_features = np.concatenate([windowed_data.flatten(), driving_raw_features])

            eye_raw_features = extract_eye_features_raw(pupil_data, blink_data, start_time, end_time)
            pupil_windowed_features = pupil_windowed.flatten()
            eye_features = np.concatenate([pupil_windowed_features, eye_raw_features])

            eeg_features = extract_eeg_features(eeg_data[[f'Electrode_{i + 1}' for i in range(8)]].values.T, fs=FS)

            if (len(driving_features) != FEATURE_SIZES['driving'] or
                    len(eye_features) != FEATURE_SIZES['eye'] or
                    len(eeg_features) != FEATURE_SIZES['eeg']):
                print(f"特征长度错误: 驾驶 {len(driving_features)}, 眼动 {len(eye_features)}, EEG {len(eeg_features)}")
                continue

            X_driving.append(driving_features)
            X_eye.append(eye_features)
            X_eeg.append(eeg_features)
            y.append(confidence)

    if len(y) == 0:
        print(f"参与者 {participant_id} 没有有效数据！")
        continue

    X_driving = np.array(X_driving)
    X_eye = np.array(X_eye)
    X_eeg = np.array(X_eeg)
    y = np.array(y)

    X_combo = np.hstack((X_driving, X_eye, X_eeg))

    # 训练和评估
    for model_name, model in models.items():
        print(f"\n=== 正在评估模型: {model_name} (参与者: {participant_id}) ===")
        fold_accuracies = []
        fold_f1_scores = []

        for train_idx, test_idx in skf.split(X_combo, y):
            X_train, X_test = X_combo[train_idx], X_combo[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            fold_accuracies.append(accuracy * 100)  # 转换为百分比
            fold_f1_scores.append(f1 * 100)         # 转换为百分比

        avg_accuracy = np.mean(fold_accuracies)
        avg_f1 = np.mean(fold_f1_scores)
        std_accuracy = np.std(fold_accuracies)

        print(f"  特征组合: 驾驶+眼动+EEG")
        print(f"    平均准确率: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
        print(f"    平均F1值: {avg_f1:.2f}%")

        # 存储所有折的结果用于小提琴图
        results[model_name]['驾驶+眼动+EEG']['accuracies'].extend(fold_accuracies)
        results[model_name]['驾驶+眼动+EEG']['f1_scores'].extend(fold_f1_scores)
        # 存储每个参与者的平均结果用于散点图
        results[model_name]['驾驶+眼动+EEG']['avg_accuracies'].append(avg_accuracy)
        results[model_name]['驾驶+眼动+EEG']['avg_f1_scores'].append(avg_f1)

# 绘图部分
if any(results[model]['驾驶+眼动+EEG']['avg_accuracies'] for model in models):
    # 准备绘图数据（使用每个参与者的平均结果）
    data = []
    for model_name in models:
        avg_accuracies = results[model_name]['驾驶+眼动+EEG']['avg_accuracies']
        avg_f1_scores = results[model_name]['驾驶+眼动+EEG']['avg_f1_scores']
        data.extend([
            {'Model': model_name, 'Metric': 'Accuracy', 'Value': acc} for acc in avg_accuracies
        ])
        data.extend([
            {'Model': model_name, 'Metric': 'F1-score', 'Value': f1} for f1 in avg_f1_scores
        ])
    data_df = pd.DataFrame(data)

    # 创建画布
    plt.figure(figsize=(20, 6))
    ax = plt.gca()

    # 自定义颜色
    custom_colors = ['#1f77b4', '#ff7f0e']  # Accuracy: 蓝色, F1-score: 橙色

    # 使用 seaborn 的分组功能绘制小提琴图
    parts = sns.violinplot(
        x='Model', y='Value', hue='Metric', data=data_df,
        palette=custom_colors, inner=None, linewidth=1.5, alpha=0.7, zorder=1, ax=ax,
        split=False  # 如果你想让小提琴图分开显示，而不是对半切分
    )
    # 手动设置边框颜色为黑色，无透明度
    for violin in parts.collections:
        violin.set_edgecolor('black')  # 设置边框颜色为黑色
        violin.set_alpha(0.7)  # 设置填充透明度为 0.5（如果你仍想要填充有透明度）
        violin.set_linewidth(1.5)  # 保持边框粗细


    # 叠加箱线图
    for i, model_name in enumerate(models):
        for j, metric in enumerate(['Accuracy', 'F1-score']):
            metric_data = data_df[(data_df['Model'] == model_name) & (data_df['Metric'] == metric)]['Value']
            if len(metric_data) > 0:
                # 计算箱线图的位置，基于模型索引和指标偏移
                position = i + (j - 0.5) * 0.4  # 0.4 是两个指标之间的间距
                ax.boxplot(
                    metric_data,
                    positions=[position], widths=0.15,
                    patch_artist=True,
                    boxprops=dict(facecolor=custom_colors[j], color="white", alpha=0.95),
                    capprops=dict(color="white"),
                    whiskerprops=dict(color="white"),
                    medianprops=dict(color="white"),
                    showfliers=False,
                    zorder=2
                )

    # 添加散点图（使用每个参与者的平均值）
    for i, model_name in enumerate(models):
        avg_accuracies = results[model_name]['驾驶+眼动+EEG']['avg_accuracies']
        avg_f1_scores = results[model_name]['驾驶+眼动+EEG']['avg_f1_scores']
        for acc in avg_accuracies:
            jitter = np.random.normal(0, 0.03, size=1)  # 添加少量抖动以避免重叠
            plt.scatter(i - 0.2 + jitter, acc, color='#1E90FF', s=30, alpha=0.9, zorder=3,
                        edgecolors='white', linewidth=1)
        for f1 in avg_f1_scores:
            jitter = np.random.normal(0, 0.03, size=1)
            plt.scatter(i + 0.2 + jitter, f1, color='darkorange', s=30, alpha=0.9, zorder=3,
                        edgecolors='white', linewidth=1)

    # 添加20%基准线
    plt.axhline(y=20, color='black', linestyle='--', linewidth=1)

    # 设置图例
    legend_patches = [
        mpatches.Patch(color=custom_colors[0], label='Accuracy', alpha=0.7),
        mpatches.Patch(color=custom_colors[1], label='F1-score', alpha=0.7)
    ]
    plt.legend(handles=legend_patches, loc='lower right', fontsize=18)

    # 设置标签和样式
    plt.ylabel('Performance (%)', fontsize=24)
    plt.xticks(range(len(models)), models.keys(), fontsize=22)
    plt.yticks(fontsize=18)
    ax.set_xlabel('')
    plt.ylim(0, 82)


    # 保存为SVG格式到指定位置
    plt.tight_layout()
    # 设置保存路径
    save_dir = "../output/"
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    plt.savefig(os.path.join(save_dir, "violin_1.svg"), format="svg", bbox_inches='tight')
    plt.show()

    # 输出总体结果（基于平均值）
    print("\n=== 模型性能总体结果（基于每个参与者的平均值） ===")
    for model in models:
        print(f"\n模型: {model}")
        avg_accuracy = np.mean(results[model]['驾驶+眼动+EEG']['avg_accuracies'])
        avg_f1 = np.mean(results[model]['驾驶+眼动+EEG']['avg_f1_scores'])
        std_accuracy = np.std(results[model]['驾驶+眼动+EEG']['avg_accuracies'])
        print(f"  特征组合: 驾驶+眼动+EEG")
        print(f"    平均准确率: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
        print(f"    平均F1值: {avg_f1:.2f}%")
else:
    print("没有有效的参与者数据！")