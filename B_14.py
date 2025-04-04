# 元认知效率对准确率的解释--信心2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, t
from sklearn.linear_model import LinearRegression
import os

# 数据准备
participants = [
    "2024103001", "2024110701", "2024111801", "2024111803", "2024111901", "2024111902",
    "2024111903", "2024112101", "2024112102", "2024112201", "2024112202", "2024112301",
    "2024112302", "2024112801", "2024112901", "2024120101", "2024120102", "2024120201",
    "2024120202", "2024120203", "2024120301", "2024120303", "2024120401", "2024120403",
    "2024120501", "2024120801", "2024120901", "2025010301", "2025010401", "2025010402",
    "2025010403", "2025010405", "2025010601"
]

# MCE 数据
mce_data = {
    "2024103001": 63.26, "2024110701": 71.42, "2024111801": 70.33, "2024111803": 51.86,
    "2024111901": 63.56, "2024111902": 69.32, "2024111903": 67.31, "2024112101": 65.21,
    "2024112102": 68.20, "2024112201": 60.14, "2024112202": 62.69, "2024112301": 58.90,
    "2024112302": 59.99, "2024112801": 54.82, "2024112901": 56.65, "2024120101": 63.52,
    "2024120102": 60.31, "2024120201": 74.79, "2024120202": 52.56, "2024120203": 66.54,
    "2024120301": 67.38, "2024120303": 49.74, "2024120401": 66.35, "2024120403": 72.10,
    "2024120501": 68.78, "2024120801": 49.12, "2024120901": 48.35, "2025010301": 57.18,
    "2025010401": 49.94, "2025010402": 66.28, "2025010403": 49.46, "2025010405": 59.85,
    "2025010601": 51.43
}


# 模型准确率数据
accuracy_data = {
    "2024103001": 57.50, "2024110701": 71.04, "2024111801": 53.09, "2024111803": 37.87,
    "2024111901": 50.93, "2024111902": 61.96, "2024111903": 55.92, "2024112101": 45.37,
    "2024112102": 72.65, "2024112201": 42.33, "2024112202": 57.31, "2024112301": 48.77,
    "2024112302": 50.48, "2024112801": 39.49, "2024112901": 49.98, "2024120101": 48.43,
    "2024120102": 42.08, "2024120201": 62.35, "2024120202": 41.65, "2024120203": 49.97,
    "2024120301": 60.06, "2024120303": 32.97, "2024120401": 44.56, "2024120403": 66.83,
    "2024120501": 49.97, "2024120801": 43.33, "2024120901": 48.70, "2025010301": 35.07,
    "2025010401": 44.11, "2025010402": 38.38, "2025010403": 39.39, "2025010405": 44.09,
    "2025010601": 43.66
}

# 转换为列表
mce_values = [mce_data[pid] for pid in participants]
accuracy_values = [accuracy_data[pid] for pid in participants]

# 1. 计算 Pearson 相关系数
corr, p_value = pearsonr(mce_values, accuracy_values)
print(f"Pearson 相关系数: {corr:.3f}")
print(f"P 值: {p_value:.5f}")
if p_value < 0.05:
    print("相关性在 95% 置信水平下显著")
else:
    print("相关性不显著")

# 2. 线性回归分析
X = np.array(mce_values).reshape(-1, 1)
y = np.array(accuracy_values)
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
r2 = reg.score(X, y)
slope = reg.coef_[0]
intercept = reg.intercept_

print(f"\n线性回归结果:")
print(f"斜率: {slope:.3f}")
print(f"截距: {intercept:.3f}")
print(f"R² (解释方差比例): {r2:.3f}")

# 计算置信区间（95% 预测区间）
n = len(mce_values)
x_mean = np.mean(mce_values)
s_x = np.sum((mce_values - x_mean) ** 2)
s_y = np.std(accuracy_values, ddof=1)  # 样本标准差
t_value = t.ppf(0.975, df=n-2)  # t 分布临界值，95% 置信水平
se = s_y * np.sqrt(1/n + (np.array(mce_values) - x_mean)**2 / s_x)  # 标准误差
ci_upper = y_pred + t_value * se
ci_lower = y_pred - t_value * se

# 3. 绘制散点图并添加置信区间阴影
plt.figure(figsize=(10, 6))
plt.scatter(mce_values, accuracy_values, color='blue', alpha=0.6, s=100, label='data point')  # 增大散点大小
# 按 mce_values 排序以绘制平滑曲线和阴影
sort_idx = np.argsort(mce_values)
mce_sorted = np.array(mce_values)[sort_idx]
y_pred_sorted = y_pred[sort_idx]
ci_upper_sorted = ci_upper[sort_idx]
ci_lower_sorted = ci_lower[sort_idx]

plt.plot(mce_sorted, y_pred_sorted, color='red', linestyle='-', linewidth=4, label=f'regression curve (R² = {r2:.3f})')  # 增大曲线宽度
plt.fill_between(mce_sorted, ci_lower_sorted, ci_upper_sorted, color='gray', alpha=0.2, label='95% confidence interval')
# plt.xlabel('MCE (%)', fontsize=16)  # 增大轴标签字体
# plt.ylabel('Accuracy (%)', fontsize=16)
# plt.title('MCE vs Accuracy with 95% Confidence Interval', fontsize=20)  # 增大标题字体
# plt.legend(fontsize=14)  # 增大图例字体
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', labelsize=26)  # 增大刻度标签字体

# 保存为SVG格式到指定位置
plt.tight_layout()
save_dir = "../output/"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "medc_2.svg"), format="svg", bbox_inches='tight')
plt.show()

# 4. 输出每个参与者的 MCE 和准确率对比
df = pd.DataFrame({
    'Participant ID': participants,
    'MCE': mce_values,
    'Accuracy': accuracy_values
})
print("\n每个参与者的 MCE 和准确率对比:")
print(df.sort_values(by='MCE', ascending=False).to_string(index=False))