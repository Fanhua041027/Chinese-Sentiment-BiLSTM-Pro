# Chinese Movie Review Sentiment Analysis

本项目实现了一个面向中文影评的情感分析系统，并在同一数据集上构建了三种层次的模型进行对比：

- 传统方法下限基线：**TF‑IDF + Naive Bayes**
- 本文提出的模型：**Bi‑LSTM + Attention**
- 预训练模型上限基线：**BERT‑Base Chinese**

---

## 1. 数据集说明

本研究所使用的影评情感分析数据集来源于开源项目 [SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)。该项目由社区维护，收集并整理了多种面向中文自然语言处理任务的公开数据集，包括情感分析、文本分类、文本匹配等多个方向，是当前常用的中文 NLP 基准数据来源之一。

### 1.1 数据来源与许可

- 仓库地址：<https://github.com/SophonPlus/ChineseNlpCorpus>  
- 仓库所有者：`SophonPlus`  
- 仓库 ID：`126583907`  
- 数据类型：中文文本语料，包含电商评论、豆瓣影评、新闻数据等多个子数据集  
- 使用场景：可用于中文情感分析、文本分类、情感倾向判别等任务

使用本项目中的数据集时，应遵循原仓库给出的使用条款与许可证要求。在撰写论文或报告时，建议在参考文献或数据集说明部分对该仓库进行引用，并附上仓库链接。

示例引用方式（可根据论文格式调整）：

> 本文实验所使用的中文情感分析数据集部分来源于开源项目 ChineseNlpCorpus（GitHub 仓库：SophonPlus/ChineseNlpCorpus）。

### 1.2 本文使用的子数据集与处理方式

在众多子数据集中，本文选取了适用于影评情感分析任务的中文评论数据（如影评类或电商评论类子数据集），并进行了如下预处理步骤：

1. **数据筛选与清洗**
   - 保留包含明确情感标签（正向 / 负向）的样本；
   - 过滤明显无效或空文本记录；
   - 对极端短文本和异常长文本进行合理截断或舍弃。

2. **数据划分**
   - 将原始语料划分为训练集、验证集与测试集（比例 7:1:2）：
     - 训练集：`dataset/processed/train.csv`（约 104 万条）
     - 验证集：`dataset/processed/val.csv`（约 14.9 万条）
     - 测试集：`dataset/processed/test.csv`（约 29.7 万条）
   - 所有文件均包含 `text` 与 `label` 两列，其中：
     - `text`：中文影评文本；
     - `label`：情感标签（0 表示负向评价，1 表示正向评价）。

3. **词表与长度设定**
   - 在自构建的 Bi‑LSTM+Attention 模型中，对训练语料进行分词与统计，构建大小为 **30,000** 的词表；
   - 最大序列长度（`max_len`）统一设定为 **128**，超长文本在预处理阶段进行截断，短文本在模型输入阶段进行补齐（padding）。

### 1.3 与基线模型的一致性

为了确保与传统基线模型和 BERT 强基线对比时的公平性，所有模型均在相同的数据划分上进行训练与评估：

- 传统基线（TF‑IDF + Naive Bayes）：
  使用 `dataset/processed/train.csv` / `test.csv`，对文本进行 `jieba` 分词后构建 TF‑IDF 特征，特征维度上限同样设置为 30,000。
- 主模型（Bi‑LSTM + Attention）：
  使用同一份训练集、验证集与测试集，通过自定义词表和嵌入层进行序列建模。
- 上限基线（BERT‑Base Chinese）：
  使用相同的 `train.csv` / `test.csv`，通过 `bert-base-chinese` 进行微调，并在测试集上评估性能。

### 1.4 使用建议

若其他研究者希望复现实验或在本数据基础上开展进一步工作，建议：

1. 直接从 [SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus) 获取原始数据，放置于 `dataset/raw/` 目录；
2. 运行 `poetry run python main.py init_data` 自动完成数据清洗与划分；
3. 明确在论文或项目文档中注明数据来源，尊重原仓库及数据提供者的工作成果。

---

## 2. 环境依赖

本项目主要基于 Python 和 PyTorch 实现，建议使用 **Python 3.10+ 或 3.11**。

### 2.1 本地环境（Bi‑LSTM / Naive Bayes）

本项目使用 Poetry 管理依赖，主要用于运行以下子命令：

- `python main.py init_data` - 数据初始化
- `python main.py preprocess` - 词表构建
- `python main.py train` - Bi‑LSTM 训练
- `python main.py predict` - 模型评估与可视化
- `python main.py baseline_nb` - Naive Bayes 基线
- `python main.py plot_curve` - 训练曲线绘制

**环境要求**：

- Python：3.10 / 3.11
- 深度学习与传统 NLP：
  - `torch>=2.0.0`
  - `scikit-learn>=1.3.0`
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`
  - `jieba>=0.42.1`
  - `opencc>=1.1.6`（繁简转换）
- 可视化：
  - `matplotlib>=3.7.0`
  - `seaborn>=0.12.0`
- 其他：
  - `tqdm>=4.65.0`
  - `pandarallel>=1.6.5`（并行加速）

**安装方式**（推荐使用 Poetry）：

```bash
cd code/
poetry install
```

或参考 `requires` 文件使用 pip 安装（仅供参考）。

> 说明：由于本地 CUDA 版本较新，`transformers` + `torch` 在本地环境下存在兼容性问题，因此 **BERT 基线实验在 Docker 容器中完成**，详见下文 2.2。

### 2.2 Docker 环境（仅用于 BERT 基线）

为了避免本地 CUDA / PyTorch / transformers 版本不兼容导致的错误，BERT‑Base 基线在单独的 Docker 容器中运行。

#### 为什么需要 Docker？

| 环境     | 问题描述                                                       |
|----------|----------------------------------------------------------------|
| 本地     | CUDA 版本过高（如 12.x），与 transformers / torch 组合存在兼容性问题 |
| Docker   | 使用 PyTorch 官方镜像，预装兼容的 CUDA 12.1 + PyTorch 2.3.0，环境隔离 |

#### 容器内运行 vs 本地运行

本项目通过 `main.py` 子命令统一调度各模块，以下是各子命令的运行环境：

| 运行环境 | 子命令                        | 说明                         |
|----------|-------------------------------|------------------------------|
| Docker   | `python main.py baseline_bert` | BERT 微调训练与测试集评估    |
| Docker   | `python main.py plot_bert_cm`  | BERT 测试集混淆矩阵绘制      |
| 本地     | `python main.py init_data`     | 数据初始化                   |
| 本地     | `python main.py preprocess`    | 文本预处理与词表构建         |
| 本地     | `python main.py train`         | Bi-LSTM + Attention 训练     |
| 本地     | `python main.py predict`       | Bi-LSTM 预测与可视化         |
| 本地     | `python main.py baseline_nb`   | 朴素贝叶斯基线               |
| 本地     | `python main.py plot_curve`    | 训练曲线绘制                 |

#### 前置条件

1. **Docker 已安装**：确保系统已安装 Docker Engine（建议 20.10+）。
2. **NVIDIA Container Toolkit 已配置**：用于在容器中启用 GPU 支持。
   ```bash
   # Arch Linux 安装示例
   sudo pacman -S nvidia-container-toolkit
   sudo systemctl restart docker
   ```
3. **GPU 驱动正常**：宿主机需安装兼容的 NVIDIA 驱动（支持 CUDA 12.1+）。

#### 构建镜像

在 `code/` 目录下执行：

```bash
docker build -t sentiment-bert .
```

构建过程会：
1. 拉取 `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` 基础镜像（约 5GB）；
2. 安装 `docker_requirements.txt` 中列出的 Python 依赖（transformers、pandas 等）。

首次构建耗时较长，后续重复构建会利用缓存加速。

#### 运行容器

**基本命令格式**：

```bash
docker run --gpus all --ipc=host -it --rm -v "$(pwd)":/app sentiment-bert [命令]
```

**参数说明**：

| 参数           | 说明                                                         |
|----------------|--------------------------------------------------------------|
| `--gpus all`   | 启用所有可用 GPU（需要 NVIDIA Container Toolkit）            |
| `--ipc=host`   | 共享宿主机 IPC 命名空间，避免 PyTorch DataLoader 内存不足    |
| `-it`          | 交互模式 + 分配伪终端                                        |
| `--rm`         | 容器退出后自动删除                                           |
| `-v "$(pwd)":/app` | 将当前目录（code/）挂载到容器内 /app，实现数据共享       |

**示例 1：进入容器交互式 Shell**

```bash
cd code/
docker run --gpus all --ipc=host -it --rm -v "$(pwd)":/app sentiment-bert
# 进入容器后执行
python main.py baseline_bert
```

**示例 2：直接运行 BERT 微调训练**

```bash
cd code/
docker run --gpus all --ipc=host -it --rm -v "$(pwd)":/app sentiment-bert \
    python main.py baseline_bert
```

**示例 3：绘制 BERT 混淆矩阵**（需先完成 BERT 微调训练）

```bash
cd code/
docker run --gpus all --ipc=host -it --rm -v "$(pwd)":/app sentiment-bert \
    python main.py plot_bert_cm
```

> **说明**：`main.py` 是项目的统一命令行入口，`baseline_bert` 和 `plot_bert_cm` 是其子命令。
> 容器工作目录为 `/app`，通过 `-v "$(pwd)":/app` 挂载后，容器内可直接访问 `dataset/`、`checkpoints/` 等目录。

#### 常见问题排查

| 问题                                      | 可能原因                     | 解决方案                              |
|-------------------------------------------|------------------------------|---------------------------------------|
| `docker: Error response from daemon: could not select device driver` | 未安装 nvidia-container-toolkit | 安装 nvidia-container-toolkit 并重启 Docker |
| `CUDA out of memory`                      | GPU 显存不足                 | 减小 batch_size 或关闭其他占用 GPU 的程序 |
| `No such file or directory: 'dataset/...'` | 未正确挂载目录              | 确保在 `code/` 目录下执行 docker run |
| 模型下载缓慢                              | HuggingFace 服务器网络问题  | 设置 HF_ENDPOINT 或使用镜像源        |

---

## 3. 脚本说明与运行顺序

本项目通过 `main.py` 统一调度所有功能模块。以下命令均在 `code/` 目录下执行，本地环境使用 Poetry 管理依赖。

### 3.1 数据准备与预处理（本地运行）

1. **数据初始化**

   ```bash
   poetry run python main.py init_data
   ```

   - 功能：从原始数据源（`dataset/raw/ratings.csv`）构建本项目所需的影评数据，并划分为训练集、验证集和测试集。
   - 输出：
     - `dataset/processed/train.csv`：训练集（约 104 万条）
     - `dataset/processed/val.csv`：验证集
     - `dataset/processed/test.csv`：测试集（约 29.7 万条）
     - `dataset/processed/meta.json`：数据集统计信息（长度分布、P95 等）
     - `dataset/figures/length_distribution.png`：文本长度分布图

2. **文本预处理与词表构建**

   ```bash
   poetry run python main.py preprocess
   ```

   - 功能：对训练集进行分词与词频统计，构建 Bi‑LSTM 模型使用的词表。
   - 输出：
     - `dataset/processed/vocab.pkl`：词表文件（包含 30,000 个 token）

### 3.2 主模型：Bi‑LSTM + Attention（本地运行）

1. **模型训练**

   ```bash
   poetry run python main.py train
   ```

   - 功能：训练 Bi‑LSTM + Attention 模型，使用训练集与验证集进行迭代训练，采用早停策略。
   - 核心输出：
     - `checkpoints/best_model.pth`：验证集 F1 最优的模型参数
     - `dataset/reports/training_log.csv`：训练日志（每个 epoch 的 loss、accuracy、F1）

2. **测试集评估与可视化**

   ```bash
   poetry run python main.py predict
   ```

   - 功能：加载最优模型，在测试集上进行评估，并生成可视化结果。
   - 输出：
     - `dataset/reports/test_report.txt`：测试集分类报告（precision / recall / F1 / accuracy）
     - `dataset/figures/confusion_matrix.png`：测试集混淆矩阵图
     - `dataset/figures/case_study_1.png` 等：典型样例的注意力热力图（用于论文 Case Study）

3. **训练过程曲线**

   ```bash
   poetry run python main.py plot_curve
   ```

   - 功能：根据训练日志绘制 Loss 和 Accuracy 随 Epoch 变化的曲线。
   - 输出：
     - `dataset/figures/training_curve.png`：训练/验证曲线图

### 3.3 下限基线：TF‑IDF + Naive Bayes（本地运行）

```bash
poetry run python main.py baseline_nb
```

- 功能：使用 jieba 分词 + TF‑IDF 特征提取，训练 Multinomial Naive Bayes 模型，作为传统方法下限基线。
- 输出：
  - `dataset/reports/baseline_report.txt`：Naive Bayes 测试集分类报告
  - `dataset/figures/baseline_confusion_matrix.png`：Naive Bayes 混淆矩阵

### 3.4 上限基线：BERT‑Base Chinese（Docker 中运行）

> 由于本地 CUDA 版本过新，与部分 `transformers`/`torch` 版本存在兼容性问题，**BERT‑Base 微调在 Docker 容器中进行**，容器内使用 PyTorch 2.3.0 + CUDA 12.1 的官方镜像。详细的 Docker 环境配置和使用说明请参考 [2.2 节](#22-docker-环境仅用于-bert-基线)。

1. **BERT 微调训练**

   ```bash
   # 确保在 code/ 目录下执行
   docker run --gpus all --ipc=host -it --rm -v "$(pwd)":/app sentiment-bert \
       python main.py baseline_bert
   ```

   - 功能：通过 `main.py baseline_bert` 子命令，基于 `bert-base-chinese` 在 `train.csv` 上进行三轮微调，并在 `test.csv` 上进行评估。
   - 输出：
     - `dataset/reports/bert_report.txt`：BERT 测试集分类报告
     - `checkpoints/bert_finetuned.pth`：微调后的 BERT 参数
     - `dataset/figures/model_comparison.png`：Bi‑LSTM 与 BERT 在准确率和模型体积上的对比图

2. **BERT 混淆矩阵绘制**（需先完成上一步）

   ```bash
   docker run --gpus all --ipc=host -it --rm -v "$(pwd)":/app sentiment-bert \
       python main.py plot_bert_cm
   ```

   - 功能：通过 `main.py plot_bert_cm` 子命令，加载已微调的 BERT 模型，在测试集上进行推理并绘制混淆矩阵。
   - 输出：
     - `dataset/figures/bert_confusion_matrix.png`：BERT 测试集混淆矩阵热力图

---

## 4. 结果汇总与对比

在完整运行上述脚本后，可得到如下三种模型在测试集（约 29.7 万条样本）上的表现：

| 模型                     | Accuracy | weighted F1 | 训练时间      | 模型参数量    | 模型大小 |
|--------------------------|----------|-------------|---------------|---------------|----------|
| Naive Bayes (TF‑IDF)     | 88.23%   | 87.41%      | 0.1 秒        | -             | -        |
| Bi‑LSTM + Attention      | 92.10%   | 92.15%      | ~51 分钟      | 6,208,514     | ~24 MB   |
| BERT‑Base Chinese        | 94.08%   | 94.03%      | ~708 分钟     | 102,269,186   | ~390 MB  |

> 以上数据来源于 `code/运行结果/` 目录下的实验日志。

**分析**：所提出的 Bi‑LSTM + Attention 模型在仅使用约 1/16 BERT 参数、1/16 模型体积、1/14 训练时间的情况下，将性能显著提升于传统 Naive Bayes（+3.87% Accuracy），并在准确率和 F1 指标上逼近 BERT‑Base（仅差 1.98%），为资源受限场景提供了一种具有较好性能-效率折中的解决方案。

---

## 5. 项目目录结构

```
code/
├── main.py                     # 统一命令行入口
├── pyproject.toml              # Poetry 依赖配置
├── requires                    # pip 依赖清单（备用）
├── Dockerfile                  # Docker 镜像构建文件
├── docker_requirements.txt     # Docker 容器内依赖
├── README.md                   # 本文件
│
├── src/                        # 源代码目录
│   ├── data/                   # 数据处理模块
│   │   ├── init_data.py        #   数据初始化与划分
│   │   └── preprocess.py       #   文本预处理与词表构建
│   ├── models/                 # 模型定义模块
│   │   ├── bilstm_attention.py #   Bi‑LSTM + Attention 模型
│   │   ├── baseline_naive_bayes.py  # Naive Bayes 基线
│   │   └── baseline_bert.py    #   BERT 微调基线
│   ├── utils/                  # 工具与绘图模块
│   │   ├── dataset.py          #   PyTorch Dataset 封装
│   │   ├── plot_curve.py       #   训练曲线绘制
│   │   └── plot_bert_confusion.py  # BERT 混淆矩阵绘制
│   ├── train.py                # Bi‑LSTM 训练逻辑
│   └── predict.py              # Bi‑LSTM 测试与可视化
│
├── dataset/                    # 数据目录
│   ├── raw/                    #   原始数据（ratings.csv）
│   ├── processed/              #   处理后数据（train/val/test.csv、vocab.pkl）
│   ├── figures/                #   生成的图表
│   └── reports/                #   生成的文本报告
│
├── checkpoints/                # 模型权重
│   ├── best_model.pth          #   Bi‑LSTM 最优模型
│   └── bert_finetuned.pth      #   BERT 微调后模型
│
└── 运行结果/                   # 实验运行日志（只读参考）
    ├── ini_data.txt            #   init_data 运行日志
    ├── proprecess.txt          #   preprocess 运行日志
    ├── train.txt               #   train 运行日志
    ├── predict.txt             #   predict 运行日志
    ├── baseline_nb.txt         #   baseline_nb 运行日志
    ├── baseline_bert.txt       #   baseline_bert 运行日志
    ├── plot_curve.txt          #   plot_curve 运行日志
    ├── plot_bert_cm.txt        #   plot_bert_cm 运行日志
    └── 跑.md                   #   完整运行流程说明
```

---

## 6. 致谢

本项目使用的数据集来源于 [SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)，在此对该项目的维护者和贡献者表示感谢。
如在此基础上开展进一步研究，请在论文或项目文档中注明数据集与本仓库的来源。
BERT 基线实验使用了官方 PyTorch Docker 镜像 `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`，感谢相关开源社区的支持。