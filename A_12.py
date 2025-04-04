# 80个特征版，7种特征组合，信心1,RF的“驾驶+眼动+EEG”混淆矩阵

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import differential_entropy
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
n_classes = len(all_labels)  # 动态获取最大标签数

# 参数设置
WINDOW_SIZE = 1000  # 窗口大小
FIXED_WINDOWS = 5  # 固定窗口数
FS = 500  # EEG 采样率
FEATURE_SIZES = {
    'driving': 35 + 7,  # 窗口特征（35） + 新特征（7），去掉了Collision
    'eye': 5 + 8,  # 窗口特征（5） + 新特征（8，包括眨眼次数和频率）
    'eeg': 24  # EEG 特征
}
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._split')

# 窗口处理函数
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

# 提取驾驶新特征（基于原始数据）
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

# 提取瞳孔和眨眼特征（基于原始数据，包含眨眼次数和频率）
def extract_eye_features_raw(pupil_data, blink_data, start_time, end_time):
    pupil_diameters = pupil_data['diameter'].values
    relevant_blinks = blink_data[
        (blink_data['start_timestamp'] >= start_time) &
        (blink_data['start_timestamp'] <= end_time)
        ]

    # 瞳孔特征
    pupil_mean = np.mean(pupil_diameters) if len(pupil_diameters) > 0 else 0
    pupil_std = np.std(pupil_diameters) if len(pupil_diameters) > 1 else 0
    pupil_max = np.max(pupil_diameters) if len(pupil_diameters) > 0 else 0
    pupil_min = np.min(pupil_diameters) if len(pupil_diameters) > 0 else 0

    # 眨眼特征
    blink_durations = relevant_blinks['duration'].values
    blink_count = len(relevant_blinks)  # 眨眼次数
    total_duration = (end_time - start_time) / 1000  # 时间段长度（秒）
    blink_frequency = blink_count / total_duration if total_duration > 0 else 0  # 眨眼频率
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
    for channel in range(8):  # 8 个通道
        channel_data = eeg_raw[channel]  # 每个通道的完整时间序列
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
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
}

feature_combinations = [
    '驾驶单独', '眼动单独', 'EEG单独', '驾驶+眼动', '驾驶+EEG', '眼动+EEG', '驾驶+眼动+EEG'
]

results = {model: {combo: {'accuracies': [], 'f1_scores': [], 'conf_matrices': []} for combo in feature_combinations} for model in models}

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
            # 窗口化驾驶和眼动数据
            steering_windowed = process_window(steering_data, columns=['Steering', 'Throttle', 'Brake'])
            vehicle_windowed = process_window(vehicle_data,
                                              columns=['Hero_Location_X', 'Hero_Location_Y', 'Hero_Speed_kmh',
                                                       'Hero_Acceleration_kmh2'])
            pupil_windowed = process_window(pupil_data.rename(columns={'pupil_timestamp': 'Timestamp'}),
                                            columns=['diameter'])

            # 调整驾驶和眼动数据的窗口数量
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

            # 提取驾驶特征（去掉了collision）
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
                print(f"特征长度错误: 驾驶 {len(driving_features)}, 眼动 {len(eye_features)}, EEG {len(eeg_features)}")
                continue

            X_driving.append(driving_features)
            X_eye.append(eye_features)
            X_eeg.append(eeg_features)
            y.append(confidence)

    if len(y) == 0:
        print(f"参与者 {os.path.basename(base_dir)} 没有有效数据！")
        continue

    X_driving = np.array(X_driving)
    X_eye = np.array(X_eye)
    X_eeg = np.array(X_eeg)
    y = np.array(y)

    # 特征组合
    feature_sets = {
        '驾驶单独': X_driving,
        '眼动单独': X_eye,
        'EEG单独': X_eeg,
        '驾驶+眼动': np.hstack((X_driving, X_eye)),
        '驾驶+EEG': np.hstack((X_driving, X_eeg)),
        '眼动+EEG': np.hstack((X_eye, X_eeg)),
        '驾驶+眼动+EEG': np.hstack((X_driving, X_eye, X_eeg))
    }

    # 训练和评估
    for model_name, model in models.items():
        print(f"\n=== 正在评估模型: {model_name} ===")
        for combo_name, X_combo in feature_sets.items():
            fold_accuracies = []
            fold_f1_scores = []
            fold_conf_matrices = []

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
                conf_matrix = confusion_matrix(y_test, y_pred)

                fold_accuracies.append(accuracy)
                fold_f1_scores.append(f1)
                fold_conf_matrices.append(conf_matrix)

            # 立即计算并打印结果
            avg_accuracy = np.mean(fold_accuracies)
            avg_f1 = np.mean(fold_f1_scores)
            std_accuracy = np.std(fold_accuracies)

            print(f"  特征组合: {combo_name}")
            print(f"    平均准确率: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"    平均F1值: {avg_f1:.4f}")

            # 更新results字典
            results[model_name][combo_name]['accuracies'].extend(fold_accuracies)
            results[model_name][combo_name]['f1_scores'].extend(fold_f1_scores)
            results[model_name][combo_name]['conf_matrices'].extend(fold_conf_matrices)

# 输出结果和绘制混淆矩阵（仅绘制'驾驶+眼动+EEG'）
if any(results['RandomForest'][combo]['accuracies'] for combo in feature_combinations):
    print("\n=== 模型性能结果（5折交叉验证） ===")
    for combo in feature_combinations:
        avg_accuracy = np.mean(results['RandomForest'][combo]['accuracies'])
        avg_f1 = np.mean(results['RandomForest'][combo]['f1_scores'])
        std_accuracy = np.std(results['RandomForest'][combo]['accuracies'])
        conf_matrices = results['RandomForest'][combo]['conf_matrices']

        # 统一混淆矩阵形状为最大标签数 (n_classes x n_classes)
        max_classes = n_classes
        padded_conf_matrices = []
        for cm in conf_matrices:
            padded_cm = np.zeros((max_classes, max_classes))
            rows, cols = cm.shape
            padded_cm[:rows, :cols] = cm
            padded_conf_matrices.append(padded_cm)

        # 计算平均混淆矩阵
        avg_conf_matrix = np.mean(padded_conf_matrices, axis=0)
        # 归一化混淆矩阵（转换为准确率形式）
        row_sums = avg_conf_matrix.sum(axis=1, keepdims=True)
        avg_conf_matrix_normalized = np.divide(avg_conf_matrix, row_sums, out=np.zeros_like(avg_conf_matrix), where=row_sums != 0)

        print(f"\n特征组合: {combo}")
        print(f"  平均准确率: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"  平均F1值: {avg_f1:.4f}")

        # 仅绘制'驾驶+眼动+EEG'的混淆矩阵
        if combo == '驾驶+眼动+EEG':
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(avg_conf_matrix_normalized, annot=True, fmt='.2f', cmap='YlGnBu',
                            xticklabels=range(1, max_classes + 1), yticklabels=range(1, max_classes + 1))
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()
else:
    print("没有有效的参与者数据！")