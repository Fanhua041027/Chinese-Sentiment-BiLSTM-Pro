"""
src/sentiment_trend.py

情感趋势分析与时间序列预测模块。

本模块实现基于历史情感数据的时间序列分析与趋势预测功能，主要包括：
1. 数据加载与时间聚合（按天/周/月）
2. 情感指数计算与趋势分析
3. 时间序列可视化
4. 趋势预测（LSTM 模型）

设计目标是为情感趋势分析和预测提供一个完整的解决方案，
支持事件影响分析和舆情监控等应用场景。
"""

import os
import json
import pickle
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="white", palette="muted")

CONFIG = {
    "raw_data_path": "dataset/raw/ratings.csv",
    "output_dir": "dataset/trend_figures",
    "model_save_path": "checkpoints/trend_model.pth",
    "prediction_days": 30,
    "aggregation": "W",
    "min_samples_per_period": 10,
}


class TrendLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(TrendLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class SentimentTrendAnalyzer:
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or CONFIG["raw_data_path"]
        self.output_dir = CONFIG["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self.df: Optional[pd.DataFrame] = None
        self.time_series: Optional[pd.Series] = None
        self.scaler = MinMaxScaler()
        self.model: Optional[TrendLSTM] = None

    def load_and_process_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件未找到: {self.data_path}")

        print(f"信息：正在加载数据 from {self.data_path}")
        df = pd.read_csv(self.data_path, on_bad_lines="skip")

        print(f"信息：原始数据共 {len(df):,} 条记录")

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.dropna(subset=["timestamp", "rating"])

        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df[df["rating"].between(1, 5)]

        df["sentiment_label"] = df["rating"].apply(lambda x: 1 if x > 3 else (0 if x < 3 else np.nan))
        df = df.dropna(subset=["sentiment_label"])

        df = df[df["comment"].notna() & (df["comment"].str.len() >= 5)]

        print(f"信息：清洗后有效记录 {len(df):,} 条")
        self.df = df
        return df

    def aggregate_time_series(self, agg_freq: str = "W") -> pd.Series:
        if self.df is None:
            self.load_and_process_data()

        self.df.set_index("timestamp", inplace=True)

        sentiment_series = self.df.groupby(pd.Grouper(freq=agg_freq)).agg(
            sentiment_score=("sentiment_label", "mean"),
            count=("sentiment_label", "count"),
            avg_rating=("rating", "mean"),
        )

        sentiment_series = sentiment_series[sentiment_series["count"] >= CONFIG["min_samples_per_period"]]

        print(f"信息：时间聚合完成，共 {len(sentiment_series)} 个时间窗口")
        self.time_series = sentiment_series["sentiment_score"]
        return self.time_series

    def plot_trend(self, save_path: Optional[str] = None) -> str:
        if self.time_series is None:
            self.aggregate_time_series()

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        axes[0].plot(self.time_series.index, self.time_series.values, marker="o", markersize=3, linewidth=1.5, color="#4c72b0", label="情感指数")
        rolling_mean = self.time_series.rolling(window=4, min_periods=1).mean()
        axes[0].plot(self.time_series.index, rolling_mean.values, color="#c44e52", linewidth=2, linestyle="--", label="4期移动平均")
        axes[0].set_title("Sentiment Trend Over Time (Weekly)", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Date", fontsize=12)
        axes[0].set_ylabel("Sentiment Index (0=Negative, 1=Positive)", fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if len(self.time_series) > 4:
            trend_diff = np.diff(self.time_series.values)
            colors = ["#55a868" if d > 0 else "#c44e52" for d in trend_diff]
            axes[1].bar(self.time_series.index[1:], trend_diff, color=colors, alpha=0.7, width=5)
            axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            axes[1].set_title("Weekly Sentiment Change (Delta)", fontsize=14, fontweight="bold")
            axes[1].set_xlabel("Date", fontsize=12)
            axes[1].set_ylabel("Sentiment Change", fontsize=12)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "sentiment_trend.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"信息：趋势图已保存至: {save_path}")
        plt.close()

        return save_path

    def prepare_sequences(self, data: np.ndarray, seq_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    def train_prediction_model(self, seq_length: int = 12, epochs: int = 100, batch_size: int = 32):
        if self.time_series is None:
            self.aggregate_time_series()

        data = self.time_series.values.reshape(-1, 1)
        data_scaled = self.scaler.fit_transform(data)

        X, y = self.prepare_sequences(data_scaled, seq_length)
        X = torch.FloatTensor(X).unsqueeze(-1)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = torch.FloatTensor(y[:train_size]).unsqueeze(-1), torch.FloatTensor(y[train_size:]).unsqueeze(-1)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model = TrendLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        print(f"信息：开始训练趋势预测模型，设备: {device}")
        for epoch in tqdm(range(epochs), desc="训练进度"):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}")

        os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)
        torch.save(self.model.state_dict(), CONFIG["model_save_path"])
        print(f"信息：模型已保存至: {CONFIG['model_save_path']}")

        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            predictions = self.model(X_test).cpu().numpy()

        predictions = self.scaler.inverse_transform(predictions)
        y_actual = self.scaler.inverse_transform(y_test.numpy())

        mse = np.mean((predictions - y_actual) ** 2)
        mae = np.mean(np.abs(predictions - y_actual))
        print(f"信息：测试集 MSE: {mse:.6f}, MAE: {mae:.6f}")

        return predictions, y_actual

    def predict_future(self, days: int = 30) -> np.ndarray:
        if self.model is None:
            if os.path.exists(CONFIG["model_save_path"]):
                self.model = TrendLSTM(input_size=1, hidden_size=64, num_layers=2)
                self.model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location="cpu"))
                self.model.eval()
                print("信息：已加载预训练模型")
            else:
                print("信息：未找到预训练模型，将先训练模型...")
                self.train_prediction_model()

        if self.time_series is None:
            self.aggregate_time_series()

        data = self.time_series.values.reshape(-1, 1)
        data_scaled = self.scaler.fit_transform(data)

        seq_length = 12
        last_sequence = data_scaled[-seq_length:]

        predictions = []
        current_seq = torch.FloatTensor(last_sequence).unsqueeze(0).unsqueeze(-1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            for _ in range(days):
                pred = self.model(current_seq.to(device)).cpu().numpy()
                predictions.append(pred[0, 0])
                current_seq = torch.cat([current_seq[:, 1:, :], torch.FloatTensor(pred).unsqueeze(0).unsqueeze(-1)], dim=1)

        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        print(f"信息：已生成未来 {days} 期的预测")

        return predictions.flatten()

    def plot_prediction(self, predictions: np.ndarray, future_days: int = 30, save_path: Optional[str] = None) -> str:
        if self.time_series is None:
            self.aggregate_time_series()

        last_date = self.time_series.index[-1]
        if isinstance(last_date, pd.Timestamp):
            future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=future_days, freq="W")
        else:
            future_dates = range(len(self.time_series), len(self.time_series) + future_days)

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(self.time_series.index, self.time_series.values, marker="o", markersize=3, linewidth=1.5, color="#4c72b0", label="历史情感指数")

        rolling_mean = self.time_series.rolling(window=4, min_periods=1).mean()
        ax.plot(self.time_series.index, rolling_mean.values, color="#dd8452", linewidth=2, linestyle="--", label="移动平均趋势")

        ax.plot(future_dates, predictions, marker="s", markersize=4, linewidth=2, color="#c44e52", label="预测情感指数")

        ax.fill_between(future_dates, predictions - 0.05, predictions + 0.05, color="#c44e52", alpha=0.2, label="预测置信区间")

        ax.set_title(f"Sentiment Trend & Prediction (Next {future_days} Periods)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Sentiment Index", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        if save_path is None:
            save_path = os.path.join(self.output_dir, "sentiment_prediction.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"信息：预测图已保存至: {save_path}")
        plt.close()

        return save_path

    def generate_trend_report(self, save_path: Optional[str] = None) -> dict:
        if self.time_series is None:
            self.aggregate_time_series()

        stats = {
            "total_periods": len(self.time_series),
            "date_range": {
                "start": str(self.time_series.index.min()),
                "end": str(self.time_series.index.max()),
            },
            "sentiment_stats": {
                "mean": float(self.time_series.mean()),
                "std": float(self.time_series.std()),
                "min": float(self.time_series.min()),
                "max": float(self.time_series.max()),
            },
            "trend_direction": "increasing" if self.time_series.iloc[-1] > self.time_series.iloc[0] else "decreasing",
            "overall_change": float(self.time_series.iloc[-1] - self.time_series.iloc[0]),
        }

        if len(self.time_series) > 4:
            recent_trend = self.time_series.diff().tail(4).mean()
            stats["recent_trend"] = "increasing" if recent_trend > 0 else "decreasing"
            stats["recent_change_rate"] = float(recent_trend)

        if save_path is None:
            save_path = os.path.join(self.output_dir, "trend_report.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"信息：趋势报告已保存至: {save_path}")

        return stats


def main():
    analyzer = SentimentTrendAnalyzer()

    analyzer.load_and_process_data()

    analyzer.aggregate_time_series(agg_freq="W")

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


if __name__ == "__main__":
    main()
