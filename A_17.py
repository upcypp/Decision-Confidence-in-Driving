# Button_Press_Time-1000ms到Button_Press_Time、Button_Press_Time到Button_Press_Time+1000ms、
# Button_Press_Time+1000ms到Button_Press_Time+2000ms、Button_Press_Time+2000ms到Button_Press_Time+3000ms、
# Button_Press_Time+3000ms到Button_Press_Time+4000ms这5种时间区间对应的rf结果对比，信心1，已去掉碰撞结果，加入T检验

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import differential_entropy, ttest_ind
from scipy.signal import butter, filtfilt
import random
import torch
from torch.backends import cudnn

# 固定随机种子函数
def setup_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        cudnn.enabled = False

rs = 42
setup_seed(rs)

# 参与者路径
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

# 获取 Confidence 标签
all_confidences = []
for path in participant_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_confidences.extend(df['Confidence'].values)
all_labels = np.unique(all_confidences)

# 参数设置
TOTAL_DURATION = 5000  # 总时长 5000 毫秒（仅用于参考，不直接使用）
NUM_WINDOWS = 5  # 分为 5 个窗口
WINDOW_SIZE = 1000  # 每个窗口 1000 毫秒
SUB_WINDOW_SIZE = 200  # 每 0.2 秒取一个数据
FIXED_POINTS = 5  # 每个窗口取 5 个数据点
FS = 500  # EEG 采样率
FEATURE_SIZES = {
    'driving': 35 + 7,  # 窗口特征（35） + 新特征（7）
    'eye': 5 + 8,  # 窗口特征（5） + 新特征（8，包括眨眼次数和频率）
    'eeg': 24  # EEG 特征
}
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._split')

# 窗口处理函数（每 0.2 秒取一个数据，共 5 个数据点）
def process_window(data, start_time, end_time, columns=None):
    timestamps = data['Timestamp'].values
    if columns is None:
        columns = [col for col in data.columns if col != 'Timestamp']
    windowed_data = []
    for col in columns:
        values = data[col].values if col in data.columns else np.zeros(len(timestamps))
        window_starts = np.arange(start_time, end_time, SUB_WINDOW_SIZE)
        window_values = []
        for ws in window_starts[:FIXED_POINTS]:  # 只取前 5 个点
            we = ws + SUB_WINDOW_SIZE
            mask = (timestamps >= ws) & (timestamps < we)
            if np.any(mask):
                idx = np.where(mask)[0][0]
                window_values.append(values[idx])
            else:
                window_values.append(0)
        # 如果不足 5 个点，用 0 填充
        while len(window_values) < FIXED_POINTS:
            window_values.append(0)
        windowed_data.append(window_values)
    return np.array(windowed_data).T

# 提取驾驶新特征（基于对应窗口的原始数据）
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

# 提取瞳孔和眨眼特征（基于对应窗口的原始数据）
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
    total_duration = (end_time - start_time) / 1000  # 秒
    blink_frequency = blink_count / total_duration if total_duration > 0 else 0
    blink_mean_duration = np.mean(blink_durations) if len(blink_durations) > 0 else 0
    blink_std_duration = np.std(blink_durations) if len(blink_durations) > 1 else 0

    return [pupil_mean, pupil_std, pupil_max, pupil_min,
            blink_count, blink_frequency, blink_mean_duration, blink_std_duration]

# 滤波和 EEG 特征提取函数
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

# 定义模型和特征组合
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

feature_combinations = [
    '驾驶+眼动+EEG'
]

# 存储每个窗口的结果
window_results = {i: {'accuracies': [], 'f1_scores': []} for i in range(NUM_WINDOWS)}

label_encoder = LabelEncoder()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 数据处理和模型训练
for path in participant_paths:
    base_dir = os.path.dirname(path)
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

    # 为每个时间窗口准备数据
    for window_idx in range(NUM_WINDOWS):
        X_driving, X_eye, X_eeg, y = [], [], [], []

        for index, row in experiment_log.iterrows():
            button_press_time = row['Button_Press_Time']
            # 定义每个窗口的起始和结束时间
            if window_idx == 0:
                start_time = button_press_time - 1000
                end_time = button_press_time
            elif window_idx == 1:
                start_time = button_press_time
                end_time = button_press_time + 1000
            elif window_idx == 2:
                start_time = button_press_time + 1000
                end_time = button_press_time + 2000
            elif window_idx == 3:
                start_time = button_press_time + 2000
                end_time = button_press_time + 3000
            else:  # window_idx == 4
                start_time = button_press_time + 3000
                end_time = button_press_time + 4000

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
                # 窗口化数据（每 0.2 秒取一个数据）
                steering_windowed = process_window(steering_data, start_time, end_time, columns=['Steering', 'Throttle', 'Brake'])
                vehicle_windowed = process_window(vehicle_data, start_time, end_time,
                                                  columns=['Hero_Location_X', 'Hero_Location_Y', 'Hero_Speed_kmh',
                                                           'Hero_Acceleration_kmh2'])
                pupil_windowed = process_window(pupil_data.rename(columns={'pupil_timestamp': 'Timestamp'}),
                                                start_time, end_time, columns=['diameter'])

                # 提取驾驶特征
                windowed_data = np.hstack((steering_windowed, vehicle_windowed))
                driving_raw_features = extract_driving_features_raw(steering_data, vehicle_data)
                driving_features = np.concatenate([windowed_data.flatten(), driving_raw_features])

                # 提取眼动特征
                eye_raw_features = extract_eye_features_raw(pupil_data, blink_data, start_time, end_time)
                pupil_windowed_features = pupil_windowed.flatten()
                eye_features = np.concatenate([pupil_windowed_features, eye_raw_features])

                # 提取 EEG 特征
                eeg_features = extract_eeg_features(eeg_data[[f'Electrode_{i + 1}' for i in range(8)]].values.T, fs=FS)

                # 检查特征长度
                if (len(driving_features) != FEATURE_SIZES['driving'] or
                        len(eye_features) != FEATURE_SIZES['eye'] or
                        len(eeg_features) != FEATURE_SIZES['eeg']):
                    print(f"特征长度错误 (窗口 {window_idx}): 驾驶 {len(driving_features)}, 眼动 {len(eye_features)}, EEG {len(eeg_features)}")
                    continue

                X_driving.append(driving_features)
                X_eye.append(eye_features)
                X_eeg.append(eeg_features)
                y.append(confidence)

        if len(y) == 0:
            print(f"参与者 {os.path.basename(base_dir)} 在窗口 {window_idx} 没有有效数据！")
            continue

        X_driving = np.array(X_driving)
        X_eye = np.array(X_eye)
        X_eeg = np.array(X_eeg)
        y = np.array(y)

        # 特征组合
        feature_sets = {
            '驾驶+眼动+EEG': np.hstack((X_driving, X_eye, X_eeg))
        }

        # 训练和评估
        for model_name, model in models.items():
            print(f"\n=== 正在评估模型: {model_name} (窗口 {window_idx}) ===")
            for combo_name, X_combo in feature_sets.items():
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

                    fold_accuracies.append(accuracy)
                    fold_f1_scores.append(f1)

                # 计算结果
                avg_accuracy = np.mean(fold_accuracies)
                avg_f1 = np.mean(fold_f1_scores)
                std_accuracy = np.std(fold_accuracies)

                print(f"  特征组合: {combo_name}")
                print(f"    平均准确率: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
                print(f"    平均F1值: {avg_f1:.4f}")

                # 更新窗口结果
                window_results[window_idx]['accuracies'].extend(fold_accuracies)
                window_results[window_idx]['f1_scores'].extend(fold_f1_scores)

# 计算每个窗口的平均准确率和标准差
avg_accuracies = [np.mean(window_results[i]['accuracies']) * 100 for i in range(NUM_WINDOWS)]
std_accuracies = [np.std(window_results[i]['accuracies']) * 100 for i in range(NUM_WINDOWS)]

# 打印每个窗口的平均准确率和标准差
print("\n=== 每个窗口的平均准确率和标准差 ===")
for i in range(NUM_WINDOWS):
    window_name = f"Window {i + 1} ({-1000 + i * 1000}-{i * 1000 + 1000}ms)"
    print(f"{window_name}: 平均准确率 = {avg_accuracies[i]:.4f}, 标准差 = {std_accuracies[i]:.4f}")

# 进行窗口1与其他窗口的T检验
p_values = []
for i in range(1, NUM_WINDOWS):
    t_stat, p_val = ttest_ind(window_results[0]['accuracies'], window_results[i]['accuracies'])
    p_values.append(p_val)
    print(f"窗口1与窗口{i+1}的T检验 p值: {p_val:.4f}")

# 确定显著性标记
significance = []
for p_val in p_values:
    if p_val < 0.001:
        significance.append('***')
    elif p_val < 0.01:
        significance.append('**')
    elif p_val < 0.05:
        significance.append('*')
    else:
        significance.append('ns')

# 绘制柱状图
plt.figure(figsize=(6, 8))
windows = [
    '1 ',
    '2 ',
    '3 ',
    '4 ',
    '5 '
]
bars = plt.bar(windows, avg_accuracies, yerr=std_accuracies, capsize=5, color='#4682B4', alpha=0.7)

# 设置横纵坐标数值字体大小
plt.tick_params(axis='both', labelsize=20)
# 添加基线准确率 (0.2)
plt.axhline(y=20, color='black', linestyle='--', label='Baseline')

# 设置横纵坐标标签字体大小
plt.xlabel('Time Window', fontsize=24)
plt.ylabel('Accuracy(%)', fontsize=24)

# 限制纵坐标范围为 0-1
plt.ylim(0, 100)
'''
# 在柱状图上标注数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=16)
'''
# 在柱状图上方添加盖子形状和显著性标记
y_max = max(avg_accuracies) + 20  # 计算盖子起始高度
for i, sig in enumerate(significance):
    if sig != 'ns':  # 仅为显著性结果添加线
        x1, x2 = 0, i + 1  # 第一个柱到其他柱的连线
        y = y_max + i * 7  # 盖子高度逐步上升
        plt.plot([x1, x1, x2, x2], [y, y + 2, y + 2, y], color='black', linewidth=1.5)
        plt.text((x1 + x2) / 2, y + 2.5, sig, ha='center', va='bottom', fontsize=20, color='black')

# 将图例放在右侧中间
# plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, fontsize=16)

# 调整布局
plt.tight_layout()

# 设置保存路径
save_dir = "../output/"
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
plt.savefig(os.path.join(save_dir, "time_window_1.svg"), format="svg", bbox_inches='tight')
plt.show()