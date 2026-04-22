"""
src/utils/plot_bert_confusion.py

BERT 模型测试集混淆矩阵绘制模块。

本模块加载已微调的 BERT 模型，在测试集上进行前向推理，
并根据预测结果绘制混淆矩阵热力图。

注意事项：
    - 本模块不重新训练模型，只加载 baseline_bert.py 已经微调好的权重。
    - 仅在 test.csv 上前向计算一次，得到预测标签和真实标签。
    - 根据预测结果绘制混淆矩阵 bert_confusion_matrix.png。

输出文件：
    - dataset/bert_confusion_matrix.png    BERT 测试集混淆矩阵图

主要类与函数：
    - BertConfConfig: 混淆矩阵绘制配置类（dataclass）。
    - BertTestDataset: BERT 测试集 Dataset 封装。
    - plot_confusion_matrix(): 绘制并保存混淆矩阵热力图。
    - main(): 主流程入口。

依赖：
    - 需要先运行 baseline_bert.py 生成 bert_finetuned.pth 权重文件。
"""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm


@dataclass
class BertConfConfig:
    """
    专用于绘制 BERT 测试集混淆矩阵的配置。

    注意：
        - 不重新训练模型，只加载 baseline_bert_base.py 已经微调好的权重；
        - 仅在 test.csv 上前向计算一次，得到预测标签和真实标签；
        - 根据预测结果绘制混淆矩阵 bert_confusion_matrix.png。
    """
    # 测试集路径（必须与 baseline_bert.py 中保持一致）
    # 路径调整：数据位于 dataset/processed/
    test_path: str = "dataset/processed/test.csv"

    # 预训练模型名称（与训练时保持一致）
    bert_model: str = "bert-base-chinese"

    # 微调后模型权重的路径（baseline_bert.py 已经保存好的文件）
    # 路径调整：模型位于 checkpoints/
    ckpt_path: str = "checkpoints/bert_finetuned.pth"

    # 输出混淆矩阵图片路径
    # 路径调整：图像位于 dataset/figures/
    cm_path: str = "dataset/figures/bert_confusion_matrix.png"

    # 测试时的 batch 大小与最大序列长度
    batch_size: int = 16
    max_len: int = 128

    # 推理设备：优先使用 GPU
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 全局配置实例
CONFIG = BertConfConfig()

# 绘图字体和风格设置（与其他脚本保持一致，避免论文插图风格不统一）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="white", palette="muted")


class BertTestDataset(Dataset):
    """
    仅用于在测试集上评估 BERT 的 Dataset 封装。

    与 baseline_bert_base.py 中的 Dataset 类似，
    但这里只需要测试集（不涉及训练），因此逻辑稍微简化。
    """

    def __init__(self, csv_path: str, tokenizer: BertTokenizer, max_len: int = 128):
        # 基本存在性检查，防止路径错误
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据集未找到：{csv_path}")

        print(f"信息：正在读取测试集：{csv_path}")
        df = pd.read_csv(csv_path)

        print(f"信息：测试集样本数量：{len(df):,}")
        # 提前提取文本和标签，避免在 __getitem__ 中重复索引 DataFrame
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        返回测试集样本数量。
        """
        return len(self.texts)

    def __getitem__(self, idx: int):
        """
        返回单条样本的 BERT 输入格式：
            - input_ids: token id 序列
            - attention_mask: 注意力 mask
            - labels: 真实标签（0/1）
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # 使用 tokenizer 将原始文本编码为 BERT 所需的张量格式
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def plot_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    """
    绘制并保存 BERT 的测试集混淆矩阵。

    参数：
        y_true: 测试集真实标签列表。
        y_pred: BERT 对测试集的预测标签列表。
        save_path: 输出图片保存路径。
    """
    # 计算 2x2 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 创建画布
    plt.figure(figsize=(6, 5))

    # 使用 seaborn 绘制带数字标注的热力图
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

    # 图形标题与坐标轴标签
    ax.set_title(
        "Confusion Matrix (BERT-Base Chinese)",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)

    # 布局优化并保存到文件
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"信息：BERT 混淆矩阵图已保存至：{save_path}。")


def main():
    """
    主流程：

        1. 检查并加载微调后的 BERT 权重；
        2. 构建测试集 DataLoader；
        3. 在测试集上做一次前向推理，收集预测标签；
        4. 计算并绘制混淆矩阵。
    """
    # 1. 检查微调模型权重是否存在
    if not os.path.exists(CONFIG.ckpt_path):
        raise FileNotFoundError(
            f"未找到微调后的 BERT 权重：{CONFIG.ckpt_path}。\n"
            "请先运行 baseline_bert_base.py 完成 BERT 微调。"
        )

    # 2. 加载分词器与模型结构（与训练时保持一致）
    print("信息：加载 BERT 分词器与模型结构。")
    tokenizer = BertTokenizer.from_pretrained(CONFIG.bert_model)
    model = BertForSequenceClassification.from_pretrained(
        CONFIG.bert_model,
        num_labels=2,   # 二分类任务
    )

    # 3. 加载微调后的权重
    print(f"信息：从 {CONFIG.ckpt_path} 加载微调后的权重。")
    state_dict = torch.load(CONFIG.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(CONFIG.device)
    model.eval()  # 设为评估模式，关闭 dropout 等

    # 4. 构建测试集 DataLoader
    test_dataset = BertTestDataset(CONFIG.test_path, tokenizer, max_len=CONFIG.max_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    preds: list[int] = []
    true_labels: list[int] = []

    print("信息：开始在测试集上生成预测结果（用于绘制混淆矩阵）。")
    # 5. 仅做前向推理，不计算梯度
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="BERT Confusion Matrix Eval"):
            input_ids = batch["input_ids"].to(CONFIG.device)
            attention_mask = batch["attention_mask"].to(CONFIG.device)
            labels = batch["labels"].to(CONFIG.device)

            # 前向传播得到 logits，再取 argmax 作为预测类别
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            # 将 GPU 张量移动到 CPU 并转换为 Python 列表
            preds.extend(predictions.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    # 6. 根据真实标签与预测标签绘制混淆矩阵
    plot_confusion_matrix(true_labels, preds, CONFIG.cm_path)


if __name__ == "__main__":
    main()