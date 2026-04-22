"""
生成包含三个模型（Naive Bayes、Bi-LSTM、BERT）的对比图。
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

# 绘图风格配置
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)

# 模型数据
models = ["Naive Bayes", "Bi-LSTM (Ours)", "BERT-Base Chinese"]
accuracies = [88.23, 92.10, 96.72]  # 准确率

# 模型体积（MB）
sizes = [0.1, 23.7, 390.2]  # 体积

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# 图 1：准确率对比
colors = ["#f39c12", "#2ecc71", "#95a5a6"]
bars = ax1.bar(models, accuracies, color=colors)
ax1.set_ylim(min(accuracies) - 3, 100)
ax1.set_title("Accuracy Comparison", fontsize=13, fontweight="bold", pad=8)
ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.tick_params(axis="x", labelsize=10)
for bar in bars:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.5,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
    )

# 图 2：模型体积对比
bars2 = ax2.bar(models, sizes, color=["#f39c12", "#2ecc71", "#e74c3c"])
ax2.set_title("Model Size Comparison", fontsize=13, fontweight="bold", pad=8)
ax2.set_ylabel("Size (MB)", fontsize=11)
ax2.tick_params(axis="x", labelsize=10)
for bar in bars2:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + max(1, height * 0.02),
        f"{height:.1f} MB",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()

# 保存图片
plot_path = "dataset/figures/model_comparison.png"
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"信息：模型对比图已保存至：{plot_path}。")
