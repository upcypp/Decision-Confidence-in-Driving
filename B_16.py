# 元认知效率对准确率的解释--信心1
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, t
from sklearn.linear_model import LinearRegression



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
    "2024103001": 60.83, "2024110701": 79.53, "2024111801": 71.32, "2024111803": 66.39,
    "2024111901": 67.77, "2024111902": 72.91, "2024111903": 66.50, "2024112101": 60.51,
    "2024112102": 71.16, "2024112201": 64.85, "2024112202": 61.19, "2024112301": 67.21,
    "2024112302": 62.33, "2024112801": 67.06, "2024112901": 61.14, "2024120101": 69.57,
    "2024120102": 65.68, "2024120201": 72.06, "2024120202": 55.94, "2024120203": 78.17,
    "2024120301": 69.27, "2024120303": 66.62, "2024120401": 69.98, "2024120403": 72.27,
    "2024120501": 67.59, "2024120801": 72.60, "2024120901": 71.50, "2025010301": 54.44,
    "2025010401": 51.74, "2025010402": 59.67, "2025010403": 63.27, "2025010405": 63.65,
    "2025010601": 64.08
}

# 模型准确率数据
accuracy_data = {
    "2024103001": 45.83, "2024110701": 68.03, "2024111801": 52.65, "2024111803": 45.53,
    "2024111901": 50.94, "2024111902": 62.38, "2024111903": 39.40, "2024112101": 41.12,
    "2024112102": 61.20, "2024112201": 40.93, "2024112202": 40.08, "2024112301": 43.12,
    "2024112302": 27.40, "2024112801": 46.67, "2024112901": 47.24, "2024120101": 46.59,
    "2024120102": 40.19, "2024120201": 65.36, "2024120202": 30.19, "2024120203": 67.52,
    "2024120301": 42.21, "2024120303": 48.48, "2024120401": 37.57, "2024120403": 62.65,
    "2024120501": 47.21, "2024120801": 51.61, "2024120901": 64.02, "2025010301": 43.15,
    "2025010401": 25.49, "2025010402": 47.40, "2025010403": 43.54, "2025010405": 40.00,
    "2025010601": 44.10
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
plt.figure(figsize=(9, 6))
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
plt.savefig(os.path.join(save_dir, "medc_1.svg"), format="svg", bbox_inches='tight')
plt.show()

# 4. 输出每个参与者的 MCE 和准确率对比
df = pd.DataFrame({
    'Participant ID': participants,
    'MCE': mce_values,
    'Accuracy': accuracy_values
})
print("\n每个参与者的 MCE 和准确率对比:")
print(df.sort_values(by='MCE', ascending=False).to_string(index=False))