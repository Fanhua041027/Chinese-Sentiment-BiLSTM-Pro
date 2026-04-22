"""
main.py

中文影评情感分析项目 - 统一命令行入口。

本文件作为项目的主程序入口，通过子命令方式调用各个功能模块。
所有核心逻辑均位于 src/ 目录下，本文件仅负责命令行参数解析与模块调度。

使用方式：
    python main.py init_data       # 数据预处理（清洗、划分 train/val/test）
    python main.py preprocess      # 构建词表（基于训练集）
    python main.py train           # 训练 Bi-LSTM + Attention 模型
    python main.py predict         # 测试 Bi-LSTM 模型并生成可视化
    python main.py baseline_nb     # 运行 Naive Bayes 基线实验
    python main.py baseline_bert   # 运行 BERT 微调基线实验
    python main.py plot_curve      # 绘制训练曲线图
    python main.py plot_bert_cm    # 绘制 BERT 混淆矩阵图

项目结构：
    code/
    ├── main.py                 # 本文件，统一入口
    ├── requires                # 依赖清单
    ├── dataset/                # 数据目录
    └── src/
        ├── data/               # 数据处理模块
        │   ├── init_data.py
        │   └── preprocess.py
        ├── models/             # 模型定义模块
        │   ├── bilstm_attention.py
        │   ├── baseline_naive_bayes.py
        │   └── baseline_bert.py
        ├── utils/              # 工具与绘图模块
        │   ├── dataset.py
        │   ├── plot_curve.py
        │   └── plot_bert_confusion.py
        ├── train.py            # Bi-LSTM 训练逻辑
        └── predict.py          # Bi-LSTM 测试与可视化
"""

import sys


def print_usage() -> None:
    """
    打印帮助信息。

    当用户未提供参数或提供 --help / -h 时调用，
    显示所有可用的子命令及其功能说明。
    """
    usage_text = """
中文影评情感分析项目 - 命令行入口

使用方式：
    python main.py <command>

可用命令：
    init_data       数据预处理：清洗原始数据，划分 train/val/test 集
    preprocess      构建词表：基于训练集进行分词与词频统计
    train           训练模型：训练 Bi-LSTM + Attention 情感分析模型
    predict         测试模型：在测试集上评估并生成可视化结果
    baseline_nb     基线实验：运行 TF-IDF + Naive Bayes 基线
    baseline_bert   基线实验：运行 BERT-Base-Chinese 微调基线
    plot_curve      绘制曲线：根据训练日志绘制 Loss/Accuracy 曲线
    plot_bert_cm    绘制混淆矩阵：绘制 BERT 测试集混淆矩阵
    trend           趋势分析：情感趋势分析与时间序列预测

示例：
    python main.py init_data       # 第一步：数据预处理
    python main.py preprocess      # 第二步：构建词表
    python main.py train           # 第三步：训练模型
    python main.py predict         # 第四步：测试模型
    python main.py trend           # 情感趋势分析与预测

帮助：
    python main.py --help          # 显示本帮助信息
    python main.py -h              # 同上
"""
    print(usage_text)


def run_init_data() -> None:
    """
    执行数据预处理流程。

    调用 src/data/init_data.py 中的 process_data() 函数，
    完成原始数据清洗、文本过滤、情感标签生成、数据划分等操作。
    """
    print("=" * 60)
    print("执行命令：init_data - 数据预处理")
    print("=" * 60)
    # 直接导入模块，避免通过__init__.py导入torch
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.data.init_data import process_data
    process_data()


def run_preprocess() -> None:
    """
    执行词表构建流程。

    调用 src/data/preprocess.py 的脚本入口逻辑，
    基于训练集进行分词与词频统计，生成 vocab.pkl 词表文件。
    """
    print("=" * 60)
    print("执行命令：preprocess - 构建词表")
    print("=" * 60)

    # 直接导入并执行 preprocess.py 的主逻辑
    import os
    import json
    import pandas as pd
    from src.data.preprocess import TextPreprocessor

    # 计算项目根目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 路径配置（位于 dataset/processed/）
    TRAIN_CSV = os.path.join(BASE_DIR, "dataset", "processed", "train.csv")
    META_JSON = os.path.join(BASE_DIR, "dataset", "processed", "meta.json")
    VOCAB_PKL = os.path.join(BASE_DIR, "dataset", "processed", "vocab.pkl")

    # 检查训练集是否存在
    if not os.path.exists(TRAIN_CSV):
        print(f"错误：找不到训练集文件 {TRAIN_CSV}。")
        print("提示：请先运行 'python main.py init_data' 生成数据集。")
        return

    # 读取 meta.json 获取 max_len
    if not os.path.exists(META_JSON):
        print(f"警告：未找到元信息文件 {META_JSON}，将使用默认 max_len=128。")
        max_len = 128
    else:
        print(f"信息：读取元信息文件: {META_JSON}")
        with open(META_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)
        p95 = meta.get("length_stats", {}).get("p95", 128)
        max_len = int(p95)

    print(f"信息：将采用 max_len={max_len} 构建词表。")

    # 读取训练集文本
    print(f"信息：正在读取训练集: {TRAIN_CSV}")
    df_train = pd.read_csv(TRAIN_CSV)

    if "text" not in df_train.columns:
        raise KeyError("训练集中未找到 'text' 列，请确认数据格式。")

    texts = df_train["text"].astype(str).tolist()

    # 构建词表
    processor = TextPreprocessor(max_len=max_len)
    processor.build_vocab(texts, save_path=VOCAB_PKL)

    print("信息：词表构建流程执行完毕。")


def run_train() -> None:
    """
    执行 Bi-LSTM + Attention 模型训练。

    调用 src/train.py 中的 main() 函数，
    完成模型初始化、训练循环、早停策略、模型保存等操作。
    """
    print("=" * 60)
    print("执行命令：train - 训练 Bi-LSTM + Attention 模型")
    print("=" * 60)
    from src.train import main
    main()


def run_predict() -> None:
    """
    执行 Bi-LSTM 模型测试与可视化。

    调用 src/predict.py 的主逻辑，
    在测试集上评估模型并生成分类报告、混淆矩阵、注意力热力图等。
    """
    print("=" * 60)
    print("执行命令：predict - 测试 Bi-LSTM 模型")
    print("=" * 60)
    from src.predict import load_model, evaluate_test_set, predict_case_studies

    # 加载模型与词表
    model, processor, id2token = load_model()

    # 测试集评估与混淆矩阵
    evaluate_test_set(model)

    # Case Study 可视化
    paper_cases = [
        "完全浪费时间，不仅演员演技尴尬，剧本也是一塌糊涂。",
        "这是我今年看过的最好的电影，特效炸裂，强烈推荐！",
        "虽然剧情有点老套，但是特效真的太棒了，完全值回票价。",
    ]
    predict_case_studies(model, processor, id2token, paper_cases)

    print("\n信息：测试集评估与可视化已全部生成，请查看 dataset/ 目录。")


def run_baseline_nb() -> None:
    """
    执行 Naive Bayes 基线实验。

    调用 src/models/baseline_naive_bayes.py 中的 run_baseline() 函数，
    完成 TF-IDF 特征提取、Naive Bayes 训练与测试集评估。
    """
    print("=" * 60)
    print("执行命令：baseline_nb - Naive Bayes 基线实验")
    print("=" * 60)
    from src.models.baseline_naive_bayes import run_baseline
    run_baseline()


def run_baseline_bert() -> None:
    """
    执行 BERT 微调基线实验。

    调用 src/models/baseline_bert.py 中的相关函数，
    完成 BERT-Base-Chinese 微调、测试集评估与模型对比图绘制。
    """
    print("=" * 60)
    print("执行命令：baseline_bert - BERT 微调基线实验")
    print("=" * 60)
    from src.models.baseline_bert import run_bert_experiment, plot_comparison

    # 运行 BERT 微调实验
    bert_acc, train_time_min, model_size_mb = run_bert_experiment()

    # 打印对比摘要
    print("\n" + "=" * 60)
    print("Bi-LSTM 与 BERT 对比摘要")
    print("=" * 60)
    print(f"BERT-Base Chinese Accuracy: {bert_acc:.2f}%")
    print(f"BERT 训练总耗时约：{train_time_min:.2f} 分钟")
    print(f"BERT 模型文件大小约：{model_size_mb:.1f} MB")
    print("=" * 60)

    # 绘制对比图
    plot_comparison(bert_acc, model_size_mb)


def run_plot_curve() -> None:
    """
    绘制训练曲线图。

    调用 src/utils/plot_curve.py 中的 plot_from_csv() 函数，
    基于训练日志绘制 Loss 和 Accuracy 随 Epoch 变化的曲线。
    """
    print("=" * 60)
    print("执行命令：plot_curve - 绘制训练曲线")
    print("=" * 60)
    from src.utils.plot_curve import plot_from_csv
    plot_from_csv()


def run_plot_bert_cm() -> None:
    """
    绘制 BERT 测试集混淆矩阵。

    调用 src/utils/plot_bert_confusion.py 中的 main() 函数，
    加载微调后的 BERT 模型并在测试集上生成混淆矩阵图。
    """
    print("=" * 60)
    print("执行命令：plot_bert_cm - 绘制 BERT 混淆矩阵")
    print("=" * 60)
    from src.utils.plot_bert_confusion import main
    main()


def run_sentiment_trend() -> None:
    """
    执行情感趋势分析与时间序列预测。

    调用 src/sentiment_trend.py 中的情感趋势分析功能：
    1. 从原始数据加载并处理时间序列
    2. 按周/月聚合计算情感指数
    3. 绘制情感趋势图
    4. 训练 LSTM 模型进行趋势预测
    5. 生成预测可视化与趋势报告
    """
    print("=" * 60)
    print("执行命令：trend - 情感趋势分析与预测")
    print("=" * 60)
    from src.sentiment_trend import SentimentTrendAnalyzer, CONFIG

    analyzer = SentimentTrendAnalyzer()

    analyzer.load_and_process_data()

    analyzer.aggregate_time_series(agg_freq=CONFIG["aggregation"])

    analyzer.plot_trend()

    predictions = analyzer.predict_future(days=CONFIG["prediction_days"])

    analyzer.plot_prediction(predictions, future_days=CONFIG["prediction_days"])

    stats = analyzer.generate_trend_report()

    print("\n" + "=" * 60)
    print("情感趋势分析报告")
    print("=" * 60)
    print(f"分析周期数: {stats['total_periods']}")
    print(f"时间范围: {stats['date_range']['start']} ~ {stats['date_range']['end']}")
    print(f"平均情感指数: {stats['sentiment_stats']['mean']:.4f}")
    print(f"情感指数标准差: {stats['sentiment_stats']['std']:.4f}")
    print(f"整体趋势方向: {stats['trend_direction']} ({stats['overall_change']:+.4f})")
    print(f"近期趋势: {stats.get('recent_trend', 'N/A')}")
    print("=" * 60)


# ============================
# 命令映射表
# ============================
# 将命令字符串映射到对应的执行函数
COMMAND_MAP = {
    "init_data": run_init_data,
    "preprocess": run_preprocess,
    "train": run_train,
    "predict": run_predict,
    "baseline_nb": run_baseline_nb,
    "baseline_bert": run_baseline_bert,
    "plot_curve": run_plot_curve,
    "plot_bert_cm": run_plot_bert_cm,
    "trend": run_sentiment_trend,
}


def main() -> None:
    """
    主入口函数。

    解析命令行参数并调用对应的子命令处理函数。
    支持 --help / -h 显示帮助信息。
    """
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("错误：未指定命令。")
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    # 处理帮助请求
    if command in ("--help", "-h", "help"):
        print_usage()
        sys.exit(0)

    # 查找并执行对应的命令
    if command in COMMAND_MAP:
        try:
            COMMAND_MAP[command]()
        except KeyboardInterrupt:
            print("\n\n信息：用户中断执行。")
            sys.exit(130)
        except Exception as e:
            print(f"\n错误：执行命令 '{command}' 时发生异常：{e}")
            raise
    else:
        print(f"错误：未知命令 '{command}'。")
        print(f"提示：使用 'python main.py --help' 查看可用命令。")
        sys.exit(1)


# ============================
# 脚本入口
# ============================
if __name__ == "__main__":
    main()
