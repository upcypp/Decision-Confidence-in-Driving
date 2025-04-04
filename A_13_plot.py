# 两个混淆矩阵合并
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 手动输入的混淆矩阵数据(数据来自A_\12\13.py)
conf_matrix_1 = np.array([
    [36, 28, 9, 12, 15],
    [14, 36, 21, 16, 13],
    [5, 13, 31, 28, 23],
    [2, 6, 14, 50, 28],
    [2, 2, 7, 20, 68]
])

conf_matrix_2 = np.array([
    [66, 18, 5, 6, 5],
    [25, 41, 16, 11, 7],
    [11, 19, 24, 26, 21],
    [4, 8, 15, 37, 37],
    [2, 3, 6, 19, 70]
])

# 设置图表布局，调整 figsize 以控制整体大小
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1.2]})

# 第一个混淆矩阵 (Confidence)
sns.heatmap(conf_matrix_1, annot=True, cmap='YlGnBu', vmin=0, vmax=70, cbar=False, square=True, fmt='d',
            xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5], ax=ax1,
            annot_kws={'size': 16, 'weight': 'bold'},)  # 增大混淆矩阵值的字体大小
ax1.set_title('Initial Decision Confidence', fontsize=18)

# 第二个混淆矩阵 (Confidence_2)
sns.heatmap(conf_matrix_2, annot=True, cmap='YlGnBu', vmin=0, vmax=70, cbar=True, square=True, fmt='d',
            xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5], ax=ax2,
            annot_kws={'size': 16, 'weight': 'bold'})  # 增大混淆矩阵值的字体大小
ax2.set_title('Confidence During Decision Execution', fontsize=18)

# 设置颜色条标签字体大小
cbar = ax2.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)  # 增大颜色条刻度标签的字体大小
cbar.ax.set_ylabel('Accuracy(%)', fontsize=22, rotation=90, labelpad=20)  # 添加纵坐标标题


# 设置纵坐标标题，仅在左边图显示
ax1.set_ylabel('True Label', fontsize=22)
ax2.set_ylabel('')  # 移除第二个子图的纵坐标标题

# 移除各个子图的横坐标标题，使用全局标题
ax1.set_xlabel('')
ax2.set_xlabel('')

# 添加全局横坐标标题，调整位置以避免与下方重叠
plt.suptitle('Predicted Label', fontsize=22, x=0.5, y=0.97)  # 将标题移到顶部更靠近位置

# 调整刻度标签字体大小
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=16)

# 调整布局，增加 pad 增加整体边距，优化子图间距
plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)  # 增加 pad 和子图间距


# 设置保存路径
save_dir = "../output/"
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
plt.savefig(os.path.join(save_dir, "matrix.svg"), format="svg", bbox_inches='tight')

# 显示图形
plt.show()
