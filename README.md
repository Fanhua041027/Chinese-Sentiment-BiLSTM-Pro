# 中文影评情感分析

> 基于深度学习的中文影评情感分析系统——Naive Bayes / Bi-LSTM+Attention / BERT 三模型对比

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

在同一数据集上对比了 **三种不同层次** 的模型方案，为情感分析任务提供从传统方法到预训练大模型的完整实验参考。

## 项目亮点

- **三模型对比**：Naive Bayes（传统机器学习）→ Bi-LSTM + Attention（轻量深度学习）→ BERT（预训练语言模型），覆盖从经典到前沿的方法谱系
- **大规模中文数据**：基于 148 万条豆瓣影评，训练集 104 万条，测试集 29.7 万条
- **注意力可视化**：Bi-LSTM + Attention 模型支持案例级别的注意力热力图分析
- **完整可复现**：统一命令行入口 `main.py`，一键完成数据处理 → 训练 → 评估 → 可视化全流程
- **Docker 支持**：BERT 基线提供 Dockerfile，隔离 CUDA 兼容性问题

## 实验结果

| 模型                      | 准确率        | 加权 F1      | 训练时间     | 参数量    | 模型大小     |
| ----------------------- | ---------- | ---------- | -------- | ------ | -------- |
| TF-IDF + Naive Bayes    | 88.23%     | 87.41%     | 0.1 秒    | -      | -        |
| **Bi-LSTM + Attention** | **92.10%** | **92.15%** | \~51 分钟  | 620 万  | \~24 MB  |
| BERT-Base Chinese       | 94.08%     | 94.03%     | \~708 分钟 | 1.02 亿 | \~390 MB |

> Bi-LSTM + Attention 在仅使用 **1/16 参数量** 和 **1/14 训练时间** 的条件下，准确率相比 Naive Bayes 提升 **+3.87%**，与 BERT 仅差 **1.98%**，是资源受限场景下的高性价比方案。

### 训练曲线

<p align="center">
  <img src="code/dataset/figures/training_curve.png" width="700"/>
</p>

### 混淆矩阵

<p align="center">
  <img src="code/dataset/figures/baseline_confusion_matrix.png" width="270"/>
  <img src="code/dataset/figures/confusion_matrix.png" width="270"/>
  <img src="code/dataset/figures/bert_confusion_matrix.png" width="270"/>
</p>
<p align="center">
  <em>左：Naive Bayes &nbsp;|&nbsp; 中：Bi-LSTM + Attention &nbsp;|&nbsp; 右：BERT</em>
</p>

### 模型对比

<p align="center">
  <img src="code/dataset/figures/model_comparison.png" width="600"/>
</p>

### 注意力可视化（案例分析）

<p align="center">
  <img src="code/dataset/figures/case_study_1.png" width="270"/>
  <img src="code/dataset/figures/case_study_2.png" width="270"/>
  <img src="code/dataset/figures/case_study_3.png" width="270"/>
</p>
<p align="center">
  <em>左：负向评论 &nbsp;|&nbsp; 中：正向评论 &nbsp;|&nbsp; 右：含转折的混合情感</em>
</p>

## 数据集

数据来源于 [SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)（豆瓣影评数据），经清洗、去重、过滤后得到 **148.6 万条** 有效样本：

| 数据集 | 样本数       | 正向占比 | 负向占比 |
| --- | --------- | ---- | ---- |
| 训练集 | 1,040,362 | 78%  | 22%  |
| 验证集 | 148,624   | 78%  | 22%  |
| 测试集 | 297,247   | 78%  | 22%  |

- 词表大小：30,000
- 最大序列长度：128（P95 分位数）
- 文本平均长度：40 字

<p align="center">
  <img src="code/dataset/figures/length_distribution.png" width="600"/>
</p>

> **注意**：原始数据集较大（\~270 MB），未包含在仓库中。请从上述开源项目获取 `ratings.csv` 放入 `code/dataset/raw/`，然后运行 `python main.py init_data` 自动完成清洗与划分。

## 项目结构

```
code/
├── main.py                         # 统一命令行入口
├── pyproject.toml                  # Poetry 依赖配置
├── poetry.lock                     # 锁定的依赖版本
├── Dockerfile                      # BERT 基线 Docker 镜像
├── docker_requirements.txt         # Docker 容器内依赖
├── README.md                       # 详细使用说明
│
├── src/
│   ├── data/
│   │   ├── init_data.py            # 数据清洗与划分
│   │   └── preprocess.py           # 分词与词表构建
│   ├── models/
│   │   ├── bilstm_attention.py     # Bi-LSTM + Attention 模型
│   │   ├── baseline_naive_bayes.py # TF-IDF + Naive Bayes 基线
│   │   └── baseline_bert.py        # BERT 微调基线
│   ├── utils/
│   │   ├── dataset.py              # PyTorch Dataset 封装
│   │   ├── plot_curve.py           # 训练曲线绘制
│   │   └── plot_bert_confusion.py  # BERT 混淆矩阵绘制
│   ├── train.py                    # Bi-LSTM 训练逻辑
│   └── predict.py                  # 评估与可视化
│
├── dataset/
│   ├── figures/                    # 生成的图表（9 张）
│   ├── reports/                    # 实验报告与日志
│   └── processed/
│       └── meta.json               # 数据集统计信息
│
└── 运行结果/                       # 实验运行日志
```

## 快速开始

### 环境要求

- Python 3.11+
- [Poetry](https://python-poetry.org/)（推荐）或 pip
- 支持 CUDA 的 GPU（可选，加速训练）
- Docker + NVIDIA Container Toolkit（仅 BERT 基线需要）

### 1. 克隆仓库并安装依赖

```bash
git clone https://github.com/Candy-A-Mine/Chinese-Sentiment-BiLSTM.git
cd Chinese-Sentiment-BiLSTM/code
poetry install
```

### 2. 准备数据集

从 [SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus) 下载 `ratings.csv` 放入 `code/dataset/raw/`，然后：

```bash
poetry run python main.py init_data      # 数据清洗与划分
poetry run python main.py preprocess     # 构建词表
```

### 3. 训练与评估

```bash
# Bi-LSTM + Attention（本地运行）
poetry run python main.py train          # 训练模型
poetry run python main.py predict        # 评估与可视化

# Naive Bayes 基线（本地运行）
poetry run python main.py baseline_nb

# BERT 基线（Docker 运行）
docker build -t sentiment-bert .
docker run --gpus all --ipc=host -it --rm -v "$(pwd)":/app sentiment-bert \
    python main.py baseline_bert

# 绘制训练曲线
poetry run python main.py plot_curve
```

详细的运行说明和参数配置请参阅 [`code/README.md`](code/README.md)。

## 致谢

- 数据集：[SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
- BERT 基线使用了 [PyTorch 官方 Docker 镜像](https://hub.docker.com/r/pytorch/pytorch)（`pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`）

## 许可

MIT License — 详见 [LICENSE](LICENSE) 文件。
