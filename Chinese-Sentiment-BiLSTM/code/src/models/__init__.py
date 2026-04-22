# -*- coding: utf-8 -*-
"""
src.models package

Model definition modules.

Modules:
    - bilstm_attention: Bi-LSTM + Attention sentiment analysis model (core)
    - baseline_naive_bayes: TF-IDF + Naive Bayes baseline model
    - baseline_bert: BERT-Base-Chinese fine-tuning baseline model
"""

from .bilstm_attention import SentimentModel

__all__ = ["SentimentModel"]
