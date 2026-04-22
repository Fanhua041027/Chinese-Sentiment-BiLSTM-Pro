""" src/models/baseline_bert.py BERT-Base-Chinese 微调基线模型。
本模块实现 BERT 预训练模型在中文情感分析任务上的微调，
作为 Bi-LSTM + Attention 模型的对比上界（Upper Bound Baseline）。

设计要点：
- 使用 HuggingFace transformers 库加载 bert-base-chinese 预训练模型。
- 与 Bi-LSTM 使用完全相同的数据划分（train.csv / test.csv），保证对比公平。
- 训练完成后输出准确率与模型体积对比图，用于论文中的效率分析。

输出文件：
- dataset/bert_report.txt BERT 测试集分类报告
- dataset/bert_finetuned.pth 微调后的模型权重
- dataset/model_comparison.png Bi-LSTM 与 BERT 的准确率/模型体积对比图

主要类与函数：
- BertConfig: BERT 微调配置类（dataclass）。
- BertDataset: BERT 输入格式的 Dataset 封装。
- run_bert_experiment(): 运行完整的 BERT 微调与评估流程。
- plot_comparison(): 绘制模型对比图。

依赖：
- transformers: HuggingFace 的 Transformers 库。
- torch: PyTorch 深度学习框架。
"""
import os
import time
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class BertConfig:
    """ BERT 微调与对比实验配置。
    该配置与 Bi-LSTM 模型使用相同的数据划分（train.csv / test.csv），
    用于在相同条件下对比两种模型的性能与资源开销。
    """
    # 数据路径：与 Bi-LSTM 使用的切分保持一致
    # 路径调整：数据位于 dataset/processed/
    train_path: str = "dataset/processed/train.csv"
    test_path: str = "dataset/processed/test.csv"

    # 预训练模型名称：此处使用 HuggingFace 的 bert-base-chinese
    bert_model: str = "bert-base-chinese"

    # 训练超参数
    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 3

    # 设备：优先使用 GPU
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Bi-LSTM (Ours) 的最终测试集准确率（来自 predict.py 的 test_report）
    # accuracy = 0.9210 → 92.10%
    lstm_acc: float = 92.10

    # 结果与图像输出路径
    # 路径调整：报告位于 dataset/reports/，模型位于 checkpoints/，图像位于 dataset/figures/
    report_path: str = "dataset/reports/bert_report.txt"
    ckpt_path: str = "checkpoints/bert_finetuned.pth"
    plot_path: str = "dataset/figures/model_comparison.png"

    # BiLSTM 模型路径（用于读取实际模型大小）
    lstm_ckpt_path: str = "checkpoints/best_model.pth"

CONFIG = BertConfig()

# 绘图风格配置：与其他脚本保持一致（白底网格 + DejaVu / 文泉驿 字体）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)


class BertDataset(Dataset):
    """ 用于 BERT 微调的 Dataset 封装。
    直接从 train.csv / test.csv 中读取 text 与 label 列，
    使用 HuggingFace Tokenizer 将文本编码为 input_ids 与 attention_mask。
    """
    def __init__(self, csv_path: str, tokenizer: BertTokenizer, max_len: int = 128):
        # 基础存在检查
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据集未找到：{csv_path}")
        print(f"信息：正在读取数据集：{csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"信息：样本数量：{len(self.df):,}")
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 提前取出 text 与 label 列，避免在 __getitem__ 中重复索引
        self.texts = self.df["text"].astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()

    def __len__(self) -> int:
        """ 返回数据集中样本数量。 """
        return len(self.texts)

    def __getitem__(self, idx: int):
        """ 返回单条样本的 BERT 输入格式：
        - input_ids: [seq_len]
        - attention_mask: [seq_len]
        - labels: 标量类别（0/1）
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # === 修改点：使用 tokenizer() 直接调用代替已废弃的 encode_plus ===
        # 新版 Transformers 库中，encode_plus 已被弃用，直接调用 tokenizer 即可。
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            # 注意：在 Dataset 中，通常不需要 return_tensors='pt'，
            # 因为我们会在 __getitem__ 返回字典，由 DataLoader 统一 collate。
            # 如果保留 return_tensors，会得到嵌套字典，容易导致维度错误。
        )

        # 手动转换为 Tensor (因为上面没用 return_tensors)
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)
        labels = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def count_parameters(model: torch.nn.Module) -> int:
    """ 统计模型中可训练参数数量。 """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_bert_experiment() -> tuple[float, float, float]:
    """ 在完整训练集与测试集上微调 BERT-Base-Chinese，并返回：
    - test_acc: 测试集准确率（百分数，0~100）
    - train_time_min: 训练总时长（分钟）
    - model_size_mb: 微调后模型参数文件大小（MB）
    """
    print("信息：启动 BERT-Base-Chinese 微调实验。")
    print(f"信息：使用设备：{CONFIG.device}。")

    # 1. 加载分词器与数据集
    # 请改回这行原始代码
    tokenizer = BertTokenizer.from_pretrained(CONFIG.bert_model)
    print("信息：构建训练集与测试集。")
    train_dataset = BertDataset(CONFIG.train_path, tokenizer, max_len=128)
    test_dataset = BertDataset(CONFIG.test_path, tokenizer, max_len=128)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. 加载预训练 BERT 模型
    print("信息：加载预训练模型：bert-base-chinese。")
    model = BertForSequenceClassification.from_pretrained(
        CONFIG.bert_model,
        num_labels=2,
    )
    model.to(CONFIG.device)
    num_params = count_parameters(model)
    print(f"信息：BERT 可训练参数总数：{num_params:,}。")

    # 优化器与学习率调度器（线性预热 + 线性衰减）
    optimizer = AdamW(model.parameters(), lr=CONFIG.lr)
    total_steps = len(train_loader) * CONFIG.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 3. 训练循环
    print(
        f"信息：开始微调 BERT，训练集样本数：{len(train_dataset):,}，"
        f"batch_size={CONFIG.batch_size}，epoch={CONFIG.epochs}。"
    )
    model.train()
    start_train = time.time()
    for epoch in range(CONFIG.epochs):
        epoch_loss = 0.0
        start_epoch = time.time()
        for batch in tqdm(
            train_loader, desc=f"BERT Training Epoch {epoch + 1}/{CONFIG.epochs}"
        ):
            input_ids = batch["input_ids"].to(CONFIG.device)
            attention_mask = batch["attention_mask"].to(CONFIG.device)
            labels = batch["labels"].to(CONFIG.device)

            # 前向传播，BertForSequenceClassification 内部会计算 CrossEntropyLoss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # 反向传播与参数更新
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        epoch_time = time.time() - start_epoch
        avg_loss = epoch_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{CONFIG.epochs} | "
            f"Train Loss: {avg_loss:.4f} | Time: {epoch_time / 60:.2f} min"
        )

    train_time = time.time() - start_train
    train_time_min = train_time / 60.0
    print(f"信息：BERT 训练总耗时：{train_time_min:.2f} 分钟。")

    # 4. 保存微调后的模型参数并统计体积
    torch.save(model.state_dict(), CONFIG.ckpt_path)
    model_size_mb = os.path.getsize(CONFIG.ckpt_path) / (1024 * 1024)
    print(f"信息：微调后模型参数文件大小约为：{model_size_mb:.1f} MB。")

    # 5. 测试集评估
    print(f"信息：开始在测试集（{len(test_dataset):,} 条样本）上评估 BERT。")
    model.eval()
    preds: list[int] = []
    true_labels: list[int] = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="BERT Evaluation"):
            input_ids = batch["input_ids"].to(CONFIG.device)
            attention_mask = batch["attention_mask"].to(CONFIG.device)
            labels = batch["labels"].to(CONFIG.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            preds.extend(predictions.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    # 计算测试集 Accuracy
    test_acc = accuracy_score(true_labels, preds) * 100.0
    print("\n" + "=" * 60)
    print("BERT 测试集评估结果")
    print("=" * 60)
    print(f"测试集 Accuracy: {test_acc:.2f}%")

    # 分类报告（便于与 Bi-LSTM 对比 F1 与精细指标）
    report = classification_report(
        true_labels,
        preds,
        target_names=["Negative (负向)", "Positive (正向)"],
        digits=4,
    )
    print(report)
    with open(CONFIG.report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"信息：BERT 测试集分类报告已保存至：{CONFIG.report_path}。")

    return test_acc, train_time_min, model_size_mb


def plot_comparison(bert_acc: float, bert_model_size_mb: float) -> None:
    """ 绘制 Naive Bayes、Bi-LSTM 与 BERT 在准确率与模型体积上的对比图。
    左图：三种模型的测试集 Accuracy；
    右图：三种模型的参数文件体积（MB）。
    """
    if bert_acc <= 0:
        return

    # 添加 Naive Bayes 数据
    models = ["Naive Bayes", "Bi-LSTM (Ours)", "BERT-Base Chinese"]
    # Naive Bayes 准确率来自 baseline_naive_bayes.py 的测试结果
    naive_bayes_acc = 88.23
    accuracies = [naive_bayes_acc, CONFIG.lstm_acc, bert_acc]

    # 模型体积（MB）：从实际保存的模型文件读取大小
    # Naive Bayes 模型很小，这里设为 0.1 MB
    naive_bayes_size_mb = 0.1
    lstm_size_mb = os.path.getsize(CONFIG.lstm_ckpt_path) / (1024 * 1024)
    sizes = [naive_bayes_size_mb, lstm_size_mb, bert_model_size_mb]

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
    os.makedirs(os.path.dirname(CONFIG.plot_path), exist_ok=True)
    plt.savefig(CONFIG.plot_path, dpi=300, bbox_inches="tight")
    print(f"信息：模型对比图已保存至：{CONFIG.plot_path}。")


if __name__ == "__main__":
    # 运行 BERT 微调实验，获取测试集准确率、训练时间与模型大小
    bert_acc, train_time_min, model_size_mb = run_bert_experiment()
    print("\n" + "=" * 60)
    print("Bi-LSTM 与 BERT 对比摘要（可用于论文）")
    print("=" * 60)
    print(f"Bi-LSTM (Ours) Accuracy: {CONFIG.lstm_acc:.2f}%")
    print(f"BERT-Base Chinese Accuracy: {bert_acc:.2f}%")
    print(f"BERT 训练总耗时约：{train_time_min:.2f} 分钟")
    print(f"BERT 模型文件大小约：{model_size_mb:.1f} MB")
    print("=" * 60)

    # 绘制对比图
    plot_comparison(bert_acc, model_size_mb)