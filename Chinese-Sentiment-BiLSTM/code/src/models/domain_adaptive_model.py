"""
src/models/domain_adaptive_model.py

跨领域自适应情感分析模型。

本模块在原有 Bi-LSTM + Attention 模型基础上扩展，支持：
1. 领域嵌入：学习不同领域的向量表示
2. 领域自适应层：通过领域特定适配器调整特征表示
3. 领域感知预测：根据领域信息调整输出层行为
4. 多任务学习：联合学习领域识别和情感分析

模型架构：
    输入序列（token ID）
        → 词嵌入层（Embedding）
        → 双向 LSTM（Bi-LSTM）
        → 领域自适应层（Domain Adaptation）
            ├── 领域嵌入（Domain Embedding）
            ├── 领域注意力（Domain Attention）
            └── 领域适配器（Domain Adapter）
        → 注意力加权池化（Attention Pooling）
        → 全连接层（Linear）
        → 输出 Logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from src.models.bilstm_attention import SentimentModel


class DomainEmbedding(nn.Module):
    """
    领域嵌入层。

    将领域标签映射为密集向量表示，供后续领域自适应模块使用。
    """

    def __init__(self, num_domains: int, embedding_dim: int):
        """
        初始化领域嵌入层。

        参数：
            num_domains (int): 领域数量。
            embedding_dim (int): 领域嵌入维度。
        """
        super(DomainEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_domains, embedding_dim)

    def forward(self, domain_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            domain_ids (Tensor): 领域 ID，形状为 [batch_size]。

        返回：
            Tensor: 领域嵌入，形状为 [batch_size, embedding_dim]。
        """
        return self.embedding(domain_ids)


class DomainAdapter(nn.Module):
    """
    领域适配器模块。

    通过领域嵌入来调整输入特征的表示，实现跨领域适应。
    采用门控机制，允许模型学习如何根据领域信息调整特征。
    """

    def __init__(self, feature_dim: int, domain_embed_dim: int):
        """
        初始化领域适配器。

        参数：
            feature_dim (int): 输入特征维度。
            domain_embed_dim (int): 领域嵌入维度。
        """
        super(DomainAdapter, self).__init__()

        self.feature_dim = feature_dim
        self.domain_embed_dim = domain_embed_dim

        self.domain_transform = nn.Linear(domain_embed_dim, feature_dim)

        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim + domain_embed_dim, feature_dim),
            nn.Sigmoid()
        )

        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(
        self,
        features: torch.Tensor,
        domain_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播。

        参数：
            features (Tensor): 输入特征，形状为 [batch_size, feature_dim]。
            domain_embed (Tensor): 领域嵌入，形状为 [batch_size, domain_embed_dim]。

        返回：
            Tensor: 领域适配后的特征，形状为 [batch_size, feature_dim]。
        """
        domain_transformed = self.domain_transform(domain_embed)

        combined = torch.cat([features, domain_embed], dim=-1)
        gate = self.gate_network(combined)

        adapted = features * gate + domain_transformed * (1 - gate)
        adapted = self.feature_transform(adapted)

        return adapted


class DomainAttention(nn.Module):
    """
    领域注意力模块。

    在序列处理过程中融入领域信息，使模型能够关注对特定领域重要的词汇。
    """

    def __init__(self, hidden_dim: int, domain_embed_dim: int):
        """
        初始化领域注意力模块。

        参数：
            hidden_dim (int): LSTM 隐藏层维度（双向后为 2*hidden_dim）。
            domain_embed_dim (int): 领域嵌入维度。
        """
        super(DomainAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.domain_embed_dim = domain_embed_dim

        self.domain_attention_transform = nn.Linear(domain_embed_dim, hidden_dim * 2)

        self.attention_weights = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        lstm_output: torch.Tensor,
        domain_embed: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        参数：
            lstm_output (Tensor): LSTM 输出，形状为 [batch_size, seq_len, 2*hidden_dim]。
            domain_embed (Tensor): 领域嵌入，形状为 [batch_size, domain_embed_dim]。
            mask (Tensor): padding 掩蔽，形状为 [batch_size, seq_len]。

        返回：
            Tuple[Tensor, Tensor]: (上下文向量, 注意力权重)。
        """
        domain_attn = self.domain_attention_transform(domain_embed)

        domain_attn = domain_attn.unsqueeze(1)

        enhanced_output = lstm_output + domain_attn

        attn_scores = self.attention_weights(enhanced_output).squeeze(-1)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e4)

        soft_attn_weights = F.softmax(attn_scores, dim=1)

        context = torch.sum(
            lstm_output * soft_attn_weights.unsqueeze(-1),
            dim=1
        )

        return context, soft_attn_weights


class DomainAdaptiveModel(nn.Module):
    """
    领域自适应情感分析模型。

    在 Bi-LSTM + Attention 基础上增加领域自适应能力：
    1. 领域嵌入层学习领域表示
    2. 领域注意力在序列处理中融入领域信息
    3. 领域适配器在特征层面进行调整

    支持两种模式：
    - 领域感知模式：提供领域 ID，使用领域自适应
    - 通用模式：不提供领域 ID，使用平均领域嵌入
    """

    DOMAIN_MAP = {
        "film": 0,
        "product": 1,
        "news": 2,
        "mixed": 3
    }
    DOMAIN_COUNT = 4

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        domain_embed_dim: int = 32
    ):
        """
        初始化领域自适应模型。

        参数：
            vocab_size (int): 词表大小。
            embedding_dim (int): 词嵌入维度。
            hidden_dim (int): LSTM 隐藏层维度。
            output_dim (int): 输出维度。
            n_layers (int): LSTM 层数。
            dropout (float): Dropout 比例。
            domain_embed_dim (int): 领域嵌入维度。
        """
        super(DomainAdaptiveModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.domain_embed_dim = domain_embed_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        self.domain_embedding = DomainEmbedding(
            num_domains=self.DOMAIN_COUNT,
            embedding_dim=domain_embed_dim
        )

        self.domain_attention = DomainAttention(
            hidden_dim=hidden_dim,
            domain_embed_dim=domain_embed_dim
        )

        self.domain_adapter = DomainAdapter(
            feature_dim=hidden_dim * 2,
            domain_embed_dim=domain_embed_dim
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout_layer = nn.Dropout(dropout)

        self._init_fallback_domain_embed()

    def _init_fallback_domain_embed(self):
        """
        初始化fallback领域嵌入，用于通用模式。
        """
        self.register_buffer(
            "fallback_domain_embed",
            torch.zeros(1, self.domain_embed_dim)
        )

    def get_domain_id(self, domain: str) -> int:
        """
        获取领域对应的 ID。

        参数：
            domain (str): 领域名称。

        返回：
            int: 领域 ID。
        """
        return self.DOMAIN_MAP.get(domain, self.DOMAIN_MAP["mixed"])

    def forward(
        self,
        text: torch.Tensor,
        domain: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        参数：
            text (Tensor): 输入 token ID 序列，形状为 [batch_size, seq_len]。
            domain (Tensor, optional): 领域 ID，形状为 [batch_size]。如果为 None，则使用 fallback 嵌入。

        返回：
            Tuple[Tensor, Tensor]: (logits, attention_weights)。
        """
        batch_size = text.size(0)
        mask = (text != 0)

        embedded = self.dropout_layer(self.embedding(text))

        lstm_output, _ = self.lstm(embedded)

        if domain is not None:
            domain_embed = self.domain_embedding(domain)
        else:
            domain_embed = self.fallback_domain_embed.expand(batch_size, -1)

        attn_output, attn_weights = self.domain_attention(
            lstm_output,
            domain_embed,
            mask
        )

        adapted_output = self.domain_adapter(attn_output, domain_embed)

        adapted_output = self.dropout_layer(adapted_output)
        logits = self.fc(adapted_output)

        return logits, attn_weights

    def forward_with_domain_name(
        self,
        text: torch.Tensor,
        domain_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用领域名称进行前向传播。

        参数：
            text (Tensor): 输入 token ID 序列，形状为 [batch_size, seq_len]。
            domain_name (str): 领域名称 ("film", "product", "news", "mixed")。

        返回：
            Tuple[Tensor, Tensor]: (logits, attention_weights)。
        """
        domain_id = self.get_domain_id(domain_name)
        domain_tensor = torch.full((text.size(0),), domain_id, dtype=torch.long, device=text.device)
        return self.forward(text, domain_tensor)

    def extract_domain_features(
        self,
        text: torch.Tensor,
        domain: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        提取领域相关特征。

        参数：
            text (Tensor): 输入 token ID 序列。
            domain (Tensor, optional): 领域 ID。

        返回：
            Dict[str, Tensor]: 包含各类特征的字典。
        """
        batch_size = text.size(0)
        mask = (text != 0)

        embedded = self.dropout_layer(self.embedding(text))
        lstm_output, _ = self.lstm(embedded)

        if domain is not None:
            domain_embed = self.domain_embedding(domain)
        else:
            domain_embed = self.fallback_domain_embed.expand(batch_size, -1)

        attn_output, attn_weights = self.domain_attention(
            lstm_output,
            domain_embed,
            mask
        )

        adapted_output = self.domain_adapter(attn_output, domain_embed)

        return {
            "lstm_output": lstm_output,
            "attn_output": attn_output,
            "domain_embed": domain_embed,
            "adapted_output": adapted_output,
            "attention_weights": attn_weights
        }


class MultiTaskDomainModel(nn.Module):
    """
    多任务领域自适应模型。

    同时学习：
    1. 情感分类（主任务）
    2. 领域识别（辅助任务）

    这种多任务学习方式能够提升模型在跨领域场景下的泛化能力。
    """

    DOMAIN_MAP = DomainAdaptiveModel.DOMAIN_MAP
    DOMAIN_COUNT = DomainAdaptiveModel.DOMAIN_COUNT

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        domain_embed_dim: int = 32,
        domain_loss_weight: float = 0.2
    ):
        """
        初始化多任务模型。

        参数：
            vocab_size (int): 词表大小。
            embedding_dim (int): 词嵌入维度。
            hidden_dim (int): LSTM 隐藏层维度。
            output_dim (int): 输出维度。
            n_layers (int): LSTM 层数。
            dropout (float): Dropout 比例。
            domain_embed_dim (int): 领域嵌入维度。
            domain_loss_weight (float): 领域损失权重。
        """
        super(MultiTaskDomainModel, self).__init__()

        self.domain_loss_weight = domain_loss_weight

        self.sentiment_model = DomainAdaptiveModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout,
            domain_embed_dim=domain_embed_dim
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.DOMAIN_COUNT)
        )

    def forward(
        self,
        text: torch.Tensor,
        domain: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播。

        参数：
            text (Tensor): 输入 token ID 序列。
            domain (Tensor, optional): 领域 ID（用于辅助任务）。

        返回：
            Tuple[Tensor, Tensor, Tensor]: (sentiment_logits, domain_logits, attention_weights)。
        """
        features = self.sentiment_model.extract_domain_features(text, domain)

        sentiment_logits, _ = self.sentiment_model(text, domain)

        domain_logits = self.domain_classifier(features["adapted_output"])

        return sentiment_logits, domain_logits, features["attention_weights"]

    def forward_sentiment_only(
        self,
        text: torch.Tensor,
        domain: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅进行情感预测（用于推理）。

        参数：
            text (Tensor): 输入 token ID 序列。
            domain (Tensor, optional): 领域 ID。

        返回：
            Tuple[Tensor, Tensor]: (sentiment_logits, attention_weights)。
        """
        return self.sentiment_model(text, domain)


def create_domain_adaptive_model(
    base_model: SentimentModel,
    domain_embed_dim: int = 32
) -> DomainAdaptiveModel:
    """
    从基础 SentimentModel 创建领域自适应模型。

    将预训练的 Bi-LSTM + Attention 模型权重迁移到新的领域自适应架构中。

    参数：
        base_model (SentimentModel): 基础模型。
        domain_embed_dim (int): 领域嵌入维度。

    返回：
        DomainAdaptiveModel: 领域自适应模型（已加载基础模型权重）。
    """
    model = DomainAdaptiveModel(
        vocab_size=base_model.embedding.num_embeddings,
        embedding_dim=base_model.embedding.embedding_dim,
        hidden_dim=base_model.lstm.hidden_size,
        output_dim=base_model.fc.out_features,
        n_layers=base_model.lstm.num_layers,
        dropout=base_model.dropout.p,
        domain_embed_dim=domain_embed_dim
    )

    model.embedding.weight.data[:base_model.embedding.num_embeddings] = \
        base_model.embedding.weight.data.clone()
    model.lstm.weight_ih_l0.data[:] = base_model.lstm.weight_ih_l0.data[:]
    model.lstm.weight_hh_l0.data[:] = base_model.lstm.weight_hh_l0.data[:]
    model.lstm.weight_ih_l0_reverse.data[:] = base_model.lstm.weight_ih_l0_reverse.data[:]
    model.lstm.weight_hh_l0_reverse.data[:] = base_model.lstm.weight_hh_l0_reverse.data[:]
    model.lstm.bias_ih_l0.data[:] = base_model.lstm.bias_ih_l0.data[:]
    model.lstm.bias_hh_l0.data[:] = base_model.lstm.bias_hh_l0.data[:]
    model.lstm.bias_ih_l0_reverse.data[:] = base_model.lstm.bias_ih_l0_reverse.data[:]
    model.lstm.bias_hh_l0_reverse.data[:] = base_model.lstm.bias_hh_l0_reverse.data[:]
    model.attention_weights.weight.data[:] = base_model.attention_weights.weight.data[:]
    model.attention_weights.bias.data[:] = base_model.attention_weights.bias.data[:]
    model.fc.weight.data[:] = base_model.fc.weight.data[:]
    model.fc.bias.data[:] = base_model.fc.bias.data[:]

    return model


if __name__ == "__main__":
    vocab_size = 30000
    batch_size = 4
    seq_len = 128

    model = DomainAdaptiveModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=1,
        n_layers=2,
        dropout=0.5,
        domain_embed_dim=32
    )

    text = torch.randint(1, vocab_size, (batch_size, seq_len))

    print("测试领域感知模式：")
    for domain_name in ["film", "product", "news", "mixed"]:
        logits, attn = model.forward_with_domain_name(text, domain_name)
        print(f"领域: {domain_name}, Logits shape: {logits.shape}, Attention shape: {attn.shape}")

    print("\n测试通用模式（无领域信息）：")
    logits, attn = model(text, domain=None)
    print(f"Logits shape: {logits.shape}, Attention shape: {attn.shape}")

    print("\n测试多任务模型：")
    multi_task_model = MultiTaskDomainModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=1,
        n_layers=2,
        dropout=0.5
    )
    sent_logits, domain_logits, attn = multi_task_model(text)
    print(f"Sentiment logits shape: {sent_logits.shape}")
    print(f"Domain logits shape: {domain_logits.shape}")
