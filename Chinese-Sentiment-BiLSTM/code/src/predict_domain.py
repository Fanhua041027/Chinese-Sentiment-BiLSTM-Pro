"""
src/predict_domain.py

跨领域情感分析的预测模块。

本模块提供：
1. DomainPredictor: 支持领域自动识别的情感分析预测器
2. 跨领域批量预测功能
3. 领域特定分析报告生成
"""

import os
import pickle
from typing import Dict, List, Tuple, Optional

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.models.domain_adaptive_model import DomainAdaptiveModel, MultiTaskDomainModel
from src.data.domain_preprocessor import DomainAwareTextPreprocessor, load_domain_preprocessor
from src.utils.dataset import collate_fn
from torch.utils.data import DataLoader


class DomainPredictor:
    """
    跨领域情感分析预测器。

    支持：
    - 自动领域识别 + 情感分析
    - 指定领域情感分析
    - 批量预测
    - 领域分布统计
    """

    DOMAIN_NAMES = ["film", "product", "news", "mixed"]

    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        max_len: int = 128,
        device: str = None
    ):
        """
        初始化领域预测器。

        参数：
            model_path (str): 模型文件路径。
            vocab_path (str): 词表文件路径。
            max_len (int): 序列最大长度。
            device (str, optional): 计算设备。
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len

        self.preprocessor = load_domain_preprocessor(vocab_path, max_len)

        self.model = self._load_model(model_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: str) -> DomainAdaptiveModel:
        """
        加载模型。

        参数：
            model_path (str): 模型文件路径。

        返回：
            DomainAdaptiveModel: 加载的模型。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)

        model = DomainAdaptiveModel(
            vocab_size=30000,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=1,
            n_layers=2,
            dropout=0.5,
            domain_embed_dim=32
        )

        model.load_state_dict(state_dict, strict=False)

        return model

    def predict(
        self,
        text: str,
        domain: Optional[str] = None,
        auto_detect_domain: bool = True
    ) -> Dict:
        """
        预测单条文本的情感和领域。

        参数：
            text (str): 输入文本。
            domain (str, optional): 指定领域。如果为 None 且 auto_detect_domain=True，则自动识别。
            auto_detect_domain (bool): 是否自动识别领域。

        返回：
            Dict: 包含预测结果的字典。
        """
        if auto_detect_domain and domain is None:
            domain, confidence = self.preprocessor.identify_domain_with_confidence(text)
            detected_domain = domain
        else:
            domain = domain or "mixed"
            confidence = 1.0
            detected_domain = None

        seq = self.preprocessor.text_to_sequence(text, domain=domain)
        input_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)

        domain_id = self.model.get_domain_id(domain)
        domain_tensor = torch.tensor([domain_id], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits, attn_weights = self.model(input_tensor, domain_tensor)
            prob = torch.sigmoid(logits).item()

        sentiment = "positive" if prob >= 0.5 else "negative"
        confidence_score = prob if prob >= 0.5 else 1.0 - prob

        tokens = list(self.preprocessor.vocab.keys())[:len(seq)]
        valid_len = min(len([s for s in seq if s != 0]), self.max_len)
        attention = attn_weights.squeeze(0).cpu().numpy()[:valid_len].tolist()

        result = {
            "text": text,
            "sentiment": sentiment,
            "sentiment_score": round(prob, 4),
            "confidence": round(confidence_score, 4),
            "domain": domain,
            "tokens": tokens[:valid_len],
            "attention": [round(float(a), 4) for a in attention]
        }

        if detected_domain is not None:
            result["detected_domain"] = detected_domain
            result["domain_confidence"] = round(confidence, 4)

        return result

    def predict_batch(
        self,
        texts: List[str],
        domains: Optional[List[str]] = None,
        auto_detect_domain: bool = True,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        批量预测。

        参数：
            texts (List[str]): 输入文本列表。
            domains (List[str], optional): 领域列表。
            auto_detect_domain (bool): 是否自动识别领域。
            batch_size (int): 批次大小。

        返回：
            List[Dict]: 预测结果列表。
        """
        if domains is None:
            domains = [None] * len(texts)

        if auto_detect_domain:
            detected_domains = []
            for i, text in enumerate(texts):
                if domains[i] is None:
                    domain, _ = self.preprocessor.identify_domain_with_confidence(text)
                    domains[i] = domain
                    detected_domains.append((i, domain))
        else:
            domains = [d if d is not None else "mixed" for d in domains]
            detected_domains = []

        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="批量预测"):
            batch_texts = texts[i:i+batch_size]
            batch_domains = domains[i:i+batch_size]

            sequences = [
                self.preprocessor.text_to_sequence(text, domain=domain)
                for text, domain in zip(batch_texts, batch_domains)
            ]
            input_tensors = torch.tensor(sequences, dtype=torch.long).to(self.device)

            domain_ids = [
                self.model.get_domain_id(d) for d in batch_domains
            ]
            domain_tensors = torch.tensor(domain_ids, dtype=torch.long).to(self.device)

            with torch.no_grad():
                logits, _ = self.model(input_tensors, domain_tensors)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

            for j, (text, domain, prob) in enumerate(zip(batch_texts, batch_domains, probs)):
                sentiment = "positive" if prob >= 0.5 else "negative"
                confidence = prob if prob >= 0.5 else 1.0 - prob

                result = {
                    "text": text,
                    "sentiment": sentiment,
                    "sentiment_score": round(float(prob), 4),
                    "confidence": round(float(confidence), 4),
                    "domain": domain
                }
                results.append(result)

        for idx, domain in detected_domains:
            results[idx]["detected_domain"] = domain

        return results

    def analyze_domain_distribution(
        self,
        texts: List[str]
    ) -> Dict:
        """
        分析文本集合的领域分布。

        参数：
            texts (List[str]): 文本列表。

        返回：
            Dict: 包含领域分布统计的字典。
        """
        domain_stats = {domain: 0 for domain in self.DOMAIN_NAMES}

        for text in texts:
            domain, _ = self.preprocessor.identify_domain_with_confidence(text)
            domain_stats[domain] += 1

        total = len(texts)
        distribution = {
            domain: {
                "count": count,
                "percentage": round(count / total * 100, 2) if total > 0 else 0
            }
            for domain, count in domain_stats.items()
        }

        return {
            "total": total,
            "distribution": distribution
        }

    def get_domain_sentiment_stats(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Dict:
        """
        获取各领域的情感统计。

        参数：
            texts (List[str]): 文本列表。
            labels (List[int]): 真实标签列表。

        返回：
            Dict: 各领域情感统计。
        """
        domain_results = {domain: {"positive": 0, "negative": 0, "total": 0}
                         for domain in self.DOMAIN_NAMES}

        for text, label in zip(texts, labels):
            domain, _ = self.preprocessor.identify_domain_with_confidence(text)
            sentiment = "positive" if label == 1 else "negative"
            domain_results[domain][sentiment] += 1
            domain_results[domain]["total"] += 1

        for domain in self.DOMAIN_NAMES:
            total = domain_results[domain]["total"]
            if total > 0:
                domain_results[domain]["positive_ratio"] = round(
                    domain_results[domain]["positive"] / total * 100, 2
                )
                domain_results[domain]["negative_ratio"] = round(
                    domain_results[domain]["negative"] / total * 100, 2
                )

        return domain_results


def load_predictor(
    model_path: str = "checkpoints/best_model.pth",
    vocab_path: str = "dataset/processed/vocab.pkl",
    max_len: int = 128,
    device: str = None
) -> DomainPredictor:
    """
    创建领域预测器实例。

    参数：
        model_path (str): 模型路径。
        vocab_path (str): 词表路径。
        max_len (int): 最大序列长度。
        device (str): 设备。

    返回：
        DomainPredictor: 预测器实例。
    """
    return DomainPredictor(
        model_path=model_path,
        vocab_path=vocab_path,
        max_len=max_len,
        device=device
    )


if __name__ == "__main__":
    predictor = DomainPredictor(
        model_path="checkpoints/best_model.pth",
        vocab_path="dataset/processed/vocab.pkl"
    )

    test_texts = [
        "这部电影太精彩了，特效炸裂，强烈推荐大家去看！",
        "收到货了，质量很好，性价比很高，会回购的",
        "国务院召开新闻发布会，发布最新经济政策",
        "今天天气真好，心情也很不错"
    ]

    print("=" * 60)
    print("跨领域情感分析测试")
    print("=" * 60)

    for text in test_texts:
        result = predictor.predict(text)
        print(f"\n文本: {text[:40]}...")
        print(f"识别领域: {result['domain']} (置信度: {result.get('domain_confidence', 1.0):.2f})")
        print(f"情感: {result['sentiment']} (概率: {result['sentiment_score']:.4f})")
        print(f"置信度: {result['confidence']:.4f}")
