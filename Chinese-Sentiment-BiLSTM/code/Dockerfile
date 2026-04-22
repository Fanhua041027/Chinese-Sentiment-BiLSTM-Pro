# =============================================================================
# Dockerfile - BERT 基线实验专用容器
# =============================================================================
#
# 用途：
#   本 Dockerfile 用于构建运行 BERT-Base-Chinese 微调实验的 GPU 容器环境。
#   由于宿主机 CUDA 版本较高，与 transformers / torch 存在兼容性问题，
#   因此 BERT 相关任务需在此容器中通过 main.py 子命令运行。
#
# 容器内通过 main.py 子命令运行的任务：
#   - python main.py baseline_bert      BERT 微调训练与评估
#   - python main.py plot_bert_cm       BERT 测试集混淆矩阵绘制
#
# 本地直接运行的子命令（无需 Docker）：
#   - python main.py init_data          数据初始化
#   - python main.py preprocess         文本预处理与词表构建
#   - python main.py train              Bi-LSTM + Attention 训练
#   - python main.py predict            Bi-LSTM 预测与可视化
#   - python main.py baseline_nb        朴素贝叶斯基线
#   - python main.py plot_curve         训练曲线绘制
#
# =============================================================================

# -----------------------------------------------------------------------------
# 基础镜像
# -----------------------------------------------------------------------------
# 选用 PyTorch 官方镜像，预装 PyTorch 2.3.0 + CUDA 12.1 + cuDNN 8。
# 使用 runtime 版本而非 devel 版本，镜像体积更小，足以满足推理与微调需求。
# 镜像内已包含 Python 3.10 和完整的 CUDA 运行时库，无需额外安装驱动。
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# -----------------------------------------------------------------------------
# 工作目录
# -----------------------------------------------------------------------------
# 设置容器内工作目录为 /app。
# 运行容器时通过 -v 参数将宿主机的 code/ 目录挂载到此处，
# 使容器可以直接访问数据集、脚本和输出目录。
WORKDIR /app

# -----------------------------------------------------------------------------
# Python 依赖安装
# -----------------------------------------------------------------------------
# 复制依赖清单到容器内，并使用 pip 安装。
# --no-cache-dir：不缓存下载的包，减小镜像体积。
# 注意：torch 已在基础镜像中预装，docker_requirements.txt 中无需重复声明。
COPY docker_requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r docker_requirements.txt

# -----------------------------------------------------------------------------
# 环境变量
# -----------------------------------------------------------------------------
# LANG：设置字符编码为 UTF-8，确保中文文本处理正常。
# TZ：设置时区为亚洲/上海，使日志时间戳与本地一致。
ENV LANG=C.UTF-8
ENV TZ=Asia/Shanghai

# -----------------------------------------------------------------------------
# 默认入口命令
# -----------------------------------------------------------------------------
# 以交互式 bash 作为默认入口，方便用户进入容器后手动执行命令。
#
# 典型用法 1：进入容器交互式 Shell
#   docker run --gpus all --ipc=host -it --rm -v "$(pwd)":/app sentiment-bert
#   然后在容器内执行：python main.py baseline_bert
#
# 典型用法 2：直接运行 BERT 微调训练
#   docker run --gpus all --ipc=host -it --rm -v "$(pwd)":/app sentiment-bert \
#       python main.py baseline_bert
#
# 典型用法 3：绘制 BERT 混淆矩阵（需先完成微调训练）
#   docker run --gpus all --ipc=host -it --rm -v "$(pwd)":/app sentiment-bert \
#       python main.py plot_bert_cm
#
# 注意：容器工作目录为 /app，通过 -v "$(pwd)":/app 挂载宿主机的 code/ 目录，
#       使容器可以直接访问 dataset/、checkpoints/ 等目录，输出结果也会写入宿主机。
CMD ["/bin/bash"]
