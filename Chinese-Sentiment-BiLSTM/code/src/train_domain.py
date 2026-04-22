"""
src/train_domain.py

跨领域自适应情感分析模型的训练模块。

本模块提供：
1. DomainConfig: 领域自适应训练配置
2. 领域感知数据加载与批处理
3. 多任务训练（情感分类 + 领域识别）
4. 领域自适应微调功能

训练策略：
- 使用多任务学习联合训练情感分析和领域识别
- 支持领域嵌入的预训练和微调
- 提供领域特定评估指标
"""

import os
import time
import random
import pickle
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from src.models.domain_adaptive_model import (
    DomainAdaptiveModel,
    MultiTaskDomainModel,
    create_domain_adaptive_model
)
from src.models.bilstm_attention import SentimentModel
from src.data.domain_preprocessor import (
    DomainAwareTextPreprocessor,
    DomainIdentifier
)
from src.utils.dataset import collate_fn, calculate_pos_weight


class DomainConfig:
    """跨领域训练配置"""

    TRAIN_CSV: str = "dataset/processed/train.csv"
    VAL_CSV: str = "dataset/processed/val.csv"
    VOCAB_PATH: str = "dataset/processed/vocab.pkl"
    LOG_SAVE_PATH: str = "dataset/reports/training_log.csv"
    MODEL_SAVE_PATH: str = "checkpoints/best_model.pth"
    DOMAIN_MODEL_SAVE_PATH: str = "checkpoints/domain_adaptive_model.pth"

    MAX_LEN: int = 128
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED: int = 42

    NUM_WORKERS: int = 8
    PIN_MEMORY: bool = True

    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 256
    OUTPUT_DIM: int = 1
    N_LAYERS: int = 2
    DROPOUT: float = 0.5

    DOMAIN_EMBED_DIM: int = 32

    BATCH_SIZE: int = 512
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 20
    PATIENCE: int = 5

    DOMAIN_LOSS_WEIGHT: float = 0.2
    USE_MULTI_TASK: bool = True


class DomainSentimentDataset(Dataset):
    """
    支持领域标签的情感分析数据集。
    """

    DOMAIN_MAP = {
        "film": 0,
        "product": 1,
        "news": 2,
        "mixed": 3
    }

    def __init__(
        self,
        csv_path: str,
        vocab_path: str,
        max_len: int = 128,
        auto_detect_domain: bool = True
    ):
        """
        初始化数据集。

        参数：
            csv_path (str): CSV 文件路径。
            vocab_path (str): 词表路径。
            max_len (int): 最大序列长度。
            auto_detect_domain (bool): 是否自动识别领域。
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据集未找到: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.auto_detect_domain = auto_detect_domain

        self.processor = DomainAwareTextPreprocessor(max_len=max_len)
        self.processor.load_vocab(vocab_path)

        self.domain_identifier = DomainIdentifier()

    def __len__(self) -> int:
        return len(self.df)

    def _detect_domain(self, text: str) -> int:
        """检测文本领域"""
        domain = self.domain_identifier.identify(text)
        return self.DOMAIN_MAP.get(domain, self.DOMAIN_MAP["mixed"])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回 (input_ids, label, domain_id)
        """
        row = self.df.iloc[idx]
        text = str(row["text"])
        label = int(row["label"])

        domain = self._detect_domain(text) if self.auto_detect_domain else 3

        seq_list = self.processor.text_to_sequence(text, domain=list(self.DOMAIN_MAP.keys())[domain])
        input_ids = torch.tensor(seq_list, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float)
        domain_tensor = torch.tensor(domain, dtype=torch.long)

        return input_ids, label_tensor, domain_tensor


def set_seed(seed: int) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: nn.Module,
    iterator: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    pos_weight: torch.Tensor = None,
    domain_criterion: nn.Module = None,
    domain_loss_weight: float = 0.2
) -> Tuple[float, float, float]:
    """
    训练单轮。

    返回：(训练损失, 训练准确率, 领域准确率)
    """
    model.train()
    epoch_loss = 0.0
    correct = 0
    domain_correct = 0
    total = 0

    one_tensor = torch.tensor(1.0, device=device)

    if use_amp:
        amp_context = autocast(device_type="cuda")
    else:
        from contextlib import nullcontext
        amp_context = nullcontext()

    for texts, labels, domain_ids in tqdm(iterator, desc="Training", leave=False):
        texts = texts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)

        optimizer.zero_grad()

        with amp_context:
            sentiment_logits, domain_logits, _ = model(texts, domain_ids)

            sentiment_loss = criterion(sentiment_logits, labels)

            if pos_weight is not None:
                weight_mask = torch.where(labels == 1, pos_weight, one_tensor)
                sentiment_loss = sentiment_loss * weight_mask

            sentiment_loss = sentiment_loss.mean()

            if domain_criterion is not None and domain_logits is not None:
                domain_loss = domain_criterion(domain_logits, domain_ids)
                total_loss = sentiment_loss + domain_loss_weight * domain_loss
            else:
                total_loss = sentiment_loss

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_loss += total_loss.item()

        probs = torch.sigmoid(sentiment_logits)
        predicted = (probs > 0.5).float()
        correct += (predicted == labels).sum().item()

        if domain_criterion is not None and domain_logits is not None:
            domain_preds = domain_logits.argmax(dim=-1)
            domain_correct += (domain_preds == domain_ids).sum().item()

        total += labels.size(0)

    avg_loss = epoch_loss / len(iterator)
    accuracy = correct / total if total > 0 else 0.0
    domain_acc = domain_correct / total if total > 0 else 0.0

    return avg_loss, accuracy, domain_acc


def evaluate(
    model: nn.Module,
    iterator: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    domain_criterion: nn.Module = None
) -> Tuple[float, float, float, Dict]:
    """
    评估模型。

    返回：(损失, 准确率, F1, 领域统计)
    """
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []
    all_domain_preds = []
    all_domain_labels = []

    with torch.no_grad():
        for texts, labels, domain_ids in iterator:
            texts = texts.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            domain_ids = domain_ids.to(device, non_blocking=True)

            sentiment_logits, domain_logits, _ = model(texts, domain_ids)

            loss = criterion(sentiment_logits, labels).mean()
            epoch_loss += loss.item()

            probs = torch.sigmoid(sentiment_logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

            if domain_logits is not None:
                domain_preds = domain_logits.argmax(dim=-1)
                all_domain_preds.extend(domain_preds.cpu().numpy())
                all_domain_labels.extend(domain_ids.cpu().numpy())

    avg_loss = epoch_loss / len(iterator)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")

    domain_acc = 0.0
    if len(all_domain_preds) > 0:
        domain_acc = accuracy_score(all_domain_labels, all_domain_preds)

    domain_stats = {
        "sentiment_accuracy": accuracy,
        "sentiment_f1": f1,
        "domain_accuracy": domain_acc
    }

    return avg_loss, accuracy, f1, domain_stats


def train_domain_adaptive_model(
    config: DomainConfig = None,
    base_model_path: str = None
) -> Dict[str, Any]:
    """
    训练领域自适应模型。

    参数：
        config: 训练配置。
        base_model_path: 基础模型路径（用于迁移学习）。

    返回：
        训练历史记录。
    """
    if config is None:
        config = DomainConfig()

    set_seed(config.SEED)
    device = config.DEVICE
    use_amp = device.type == "cuda"

    print(f"信息：当前计算设备: {device}")
    print(f"信息：自动混合精度: {'启用' if use_amp else '未启用'}")

    if not os.path.exists(config.VOCAB_PATH):
        raise FileNotFoundError(f"词表文件未找到: {config.VOCAB_PATH}")

    with open(config.VOCAB_PATH, "rb") as f:
        vocab_obj = pickle.load(f)

    if isinstance(vocab_obj, dict) and "vocab" in vocab_obj:
        vocab = vocab_obj["vocab"]
    else:
        vocab = vocab_obj

    vocab_size = len(vocab)
    print(f"信息：词表大小: {vocab_size}")

    print("信息：正在构建领域感知数据集...")
    train_data = DomainSentimentDataset(
        csv_path=config.TRAIN_CSV,
        vocab_path=config.VOCAB_PATH,
        max_len=config.MAX_LEN,
        auto_detect_domain=True
    )
    val_data = DomainSentimentDataset(
        csv_path=config.VAL_CSV,
        vocab_path=config.VOCAB_PATH,
        max_len=config.MAX_LEN,
        auto_detect_domain=True
    )

    pos_weight = calculate_pos_weight(config.TRAIN_CSV).to(device)
    print(f"信息：正样本权重: {pos_weight.item():.4f}")

    train_loader = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=True
    )

    print("信息：正在初始化领域自适应模型...")
    model = MultiTaskDomainModel(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        n_layers=config.N_LAYERS,
        dropout=config.DROPOUT,
        domain_embed_dim=config.DOMAIN_EMBED_DIM,
        domain_loss_weight=config.DOMAIN_LOSS_WEIGHT
    )

    if base_model_path and os.path.exists(base_model_path):
        print(f"信息：从基础模型加载权重: {base_model_path}")
        try:
            base_model = SentimentModel(
                vocab_size=vocab_size,
                embedding_dim=config.EMBEDDING_DIM,
                hidden_dim=config.HIDDEN_DIM,
                output_dim=config.OUTPUT_DIM,
                n_layers=config.N_LAYERS,
                dropout=config.DROPOUT
            )
            base_state = torch.load(base_model_path, map_location=device)
            base_model.load_state_dict(base_state)
            model = create_domain_adaptive_model(base_model, config.DOMAIN_EMBED_DIM)
            print("信息：基础模型权重迁移成功")
        except Exception as e:
            print(f"警告：基础模型权重加载失败: {e}，使用随机初始化")

    model = model.to(device)

    num_params = count_parameters(model)
    print(f"信息：模型可训练参数总数: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    domain_criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    scaler = GradScaler(enabled=use_amp)

    best_metric = 0.0
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    print(f"信息：开始训练，共 {config.EPOCHS} 个 epoch")

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        train_loss, train_acc, train_domain_acc = train_one_epoch(
            model=model,
            iterator=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            pos_weight=pos_weight,
            domain_criterion=domain_criterion if config.USE_MULTI_TASK else None,
            domain_loss_weight=config.DOMAIN_LOSS_WEIGHT
        )

        val_loss, val_acc, val_f1, val_stats = evaluate(
            model=model,
            iterator=val_loader,
            criterion=criterion,
            device=device,
            domain_criterion=domain_criterion if config.USE_MULTI_TASK else None
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)

        history.append({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Acc": train_acc,
            "Train Domain Acc": train_domain_acc,
            "Val Loss": val_loss,
            "Val Acc": val_acc,
            "Val F1": val_f1,
            "Val Domain Acc": val_stats.get("domain_accuracy", 0),
            "Time": f"{int(mins)}m {int(secs)}s",
            "LR": current_lr
        })

        print(f"Epoch: {epoch + 1:02} | Time: {int(mins)}m {int(secs)}s | LR: {current_lr:.6f}")
        print(f"\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | Train Domain Acc: {train_domain_acc * 100:.2f}%")
        print(f"\t Val. Loss: {val_loss:.4f} | Val. Acc: {val_acc * 100:.2f}% | Val F1: {val_f1 * 100:.2f}%")

        if val_f1 > best_metric:
            best_metric = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), config.DOMAIN_MODEL_SAVE_PATH)
            print(f"\t信息：验证集 F1 提升，已保存模型 (Val F1: {best_metric * 100:.2f}%)")
        else:
            patience_counter += 1
            print(f"\t信息：验证集 F1 未提升 ({patience_counter}/{config.PATIENCE})")

        if patience_counter >= config.PATIENCE:
            print("信息：触发早停机制，停止训练")
            break

    os.makedirs(os.path.dirname(config.LOG_SAVE_PATH), exist_ok=True)
    pd.DataFrame(history).to_csv(config.LOG_SAVE_PATH, index=False)
    print(f"信息：训练日志已保存至: {config.LOG_SAVE_PATH}")

    return history


if __name__ == "__main__":
    history = train_domain_adaptive_model()
    print("\n信息：领域自适应模型训练完成")
