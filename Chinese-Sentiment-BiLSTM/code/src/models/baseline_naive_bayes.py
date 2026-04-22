"""
src/models/baseline_naive_bayes.py

传统机器学习基线模型：TF-IDF + Multinomial Naive Bayes。

本模块作为情感分类任务的"下限"基线，与 Bi-LSTM (Ours) 和 BERT-Base
形成完整的性能-效率对比。

设计要点：
    - 使用与深度模型完全一致的 train.csv / test.csv 划分，保证对比公平性。
    - TF-IDF 维度上限设为 30,000，与 Bi-LSTM 词表规模对应。
    - 使用 jieba 进行中文分词，ngram_range=(1, 2) 捕获简单短语特征。

输出文件：
    - dataset/baseline_report.txt            Naive Bayes 测试集分类报告 + 训练耗时
    - dataset/baseline_confusion_matrix.png  Naive Bayes 测试集混淆矩阵图

主要函数：
    - load_and_cut(): 读取 CSV 并进行分词。
    - plot_confusion_matrix(): 绘制混淆矩阵热力图。
    - run_baseline(): 运行完整的 Naive Bayes 基线实验。
"""

import os
import time

import jieba
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm

# ============================
# 配置
# ============================

# 路径调整：数据位于 dataset/processed/，报告位于 dataset/reports/，图像位于 dataset/figures/
TRAIN_PATH = "dataset/processed/train.csv"
TEST_PATH = "dataset/processed/test.csv"
REPORT_PATH = "dataset/reports/baseline_report.txt"
CM_PATH = "dataset/figures/baseline_confusion_matrix.png"

# 绘图字体设置（与其他脚本保持一致）
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="white", palette="muted")


# ============================
# 数据加载与分词
# ============================

def load_and_cut(filepath: str, n_jobs: int = 8) -> tuple[list[str], list[int]]:
    """
    读取 CSV 文件并进行分词。

    参数：
        filepath: CSV 文件路径，需包含 'text' 与 'label' 列。
        n_jobs:   jieba 并行分词的线程数（>=1 时启用并行）。

    返回：
        texts:  分词后的文本列表（以空格分隔的 token 串）。
        labels: 标签列表（0/1）。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据集未找到：{filepath}")

    print(f"信息：正在读取数据集：{filepath}")
    df = pd.read_csv(filepath)

    print(f"信息：正在对文本进行分词，共 {len(df):,} 条样本。")

    texts: list[str] = []
    labels: list[int] = df["label"].astype(int).tolist()

    # 可选：启用 jieba 并行分词（在多核 CPU 上可以显著加速）
    if n_jobs and n_jobs > 1:
        try:
            jieba.enable_parallel(n_jobs)
            parallel_enabled = True
            print(f"信息：已启用 jieba 并行分词（线程数：{n_jobs}）。")
        except Exception:
            # 某些环境下（如 Windows 或部分 Docker 配置）并行模式会失败，此时退回单线程
            parallel_enabled = False
            print("信息：jieba 并行模式初始化失败，将使用单线程分词。")
    else:
        parallel_enabled = False

    for text in tqdm(df["text"], desc="Jieba Cutting", unit="doc"):
        # 精确模式分词
        words = jieba.lcut(str(text), cut_all=False)
        texts.append(" ".join(words))

    if parallel_enabled:
        jieba.disable_parallel()

    return texts, labels


# ============================
# 混淆矩阵绘制
# ============================

def plot_confusion_matrix(y_true, y_pred) -> None:
    """
    绘制并保存 Naive Bayes 的测试集混淆矩阵。
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        xticklabels=["Pred Neg", "Pred Pos"],
        yticklabels=["True Neg", "True Pos"],
        cbar=False,
    )

    ax.set_title(
        "Confusion Matrix (Naive Bayes Baseline)",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=300, bbox_inches="tight")
    print(f"信息：混淆矩阵图已保存至：{CM_PATH}。")


# ============================
# 主流程：基线实验
# ============================

def run_baseline() -> None:
    """
    运行 Naive Bayes 基线实验：

        1. 读取并分词 train.csv / test.csv。
        2. 使用 TF-IDF 提取文本特征（max_features=30000）。
        3. 训练 Multinomial Naive Bayes 模型。
        4. 在测试集上评估性能，输出分类报告与混淆矩阵。
    """
    print("信息：启动传统机器学习基线（TF-IDF + Multinomial Naive Bayes）。")

    # 1. 准备数据（全量加载）
    train_texts, train_labels = load_and_cut(TRAIN_PATH, n_jobs=8)
    test_texts, test_labels = load_and_cut(TEST_PATH, n_jobs=8)

    # 2. 特征提取（TF-IDF）
    print("信息：正在计算 TF-IDF 特征矩阵。")
    # max_features=30000：与 Bi-LSTM 词表大小一致，以便于公平对比
    # ngram_range=(1, 2)：使用 1-2gram，捕获简单短语特征
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=5,      # 至少在 5 个文档中出现
        max_df=0.8,    # 出现在 >80% 文档中的视为停用词
    )

    # 拟合训练集并转换
    X_train = vectorizer.fit_transform(train_texts)
    # 只对测试集做 transform，不能再次 fit
    X_test = vectorizer.transform(test_texts)

    print(f"信息：训练特征矩阵维度：{X_train.shape}。")

    # 3. 训练 Multinomial Naive Bayes
    print("信息：开始训练 Multinomial Naive Bayes 模型。")
    start_time = time.time()
    model = MultinomialNB()
    model.fit(X_train, train_labels)
    train_time = time.time() - start_time
    print(f"信息：Naive Bayes 训练耗时：{train_time:.2f} 秒。")

    # 4. 测试集评估
    print("信息：开始在测试集上评估 Naive Bayes。")
    preds = model.predict(X_test)
    acc = accuracy_score(test_labels, preds) * 100.0

    print("\n" + "=" * 60)
    print("Naive Bayes 测试集评估结果（Baseline）")
    print("=" * 60)
    print(f"测试集 Accuracy: {acc:.2f}%")

    target_names = ["Negative (负向)", "Positive (正向)"]
    report = classification_report(
        test_labels,
        preds,
        target_names=target_names,
        digits=4,
    )
    print(report)

    # 保存详细指标到文件：包含训练时间 + Accuracy + 详细报告
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Naive Bayes 训练耗时: {train_time:.2f} 秒\n")
        f.write(f"测试集 Accuracy: {acc:.2f}%\n\n")
        f.write(report)
    print(f"信息：Naive Bayes 测试集分类报告已保存至：{REPORT_PATH}。")

    # 5. 绘制混淆矩阵
    plot_confusion_matrix(test_labels, preds)


if __name__ == "__main__":
    if not os.path.exists(TRAIN_PATH):
        print("错误：未找到训练数据集，请先运行 init_data.py。")
    else:
        run_baseline()