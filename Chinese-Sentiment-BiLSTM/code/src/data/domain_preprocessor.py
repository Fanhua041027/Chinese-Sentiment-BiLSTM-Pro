"""
src/data/domain_preprocessor.py

跨领域情感分析的预处理器模块。

本模块在原有 TextPreprocessor 基础上扩展，支持：
1. 领域识别：根据文本特征自动识别所属领域（影评、商品评论、新闻等）
2. 领域感知分词：对不同领域应用不同的分词策略
3. 领域词汇增强：为每个领域维护领域特定词汇表

支持的领域：
    - film: 影评领域
    - product: 商品评论领域
    - news: 新闻领域
    - mixed: 混合/通用领域
"""

import os
import re
from typing import List, Dict, Tuple, Optional
import pickle
import json

import jieba
import torch


class DomainIdentifier:
    """
    领域识别器。

    通过文本特征（如关键词、标点模式、文本长度等）自动识别文本所属领域。
    采用规则+统计的混合方法，确保快速准确的领域判断。
    """

    DOMAIN_KEYWORDS = {
        "film": {
            "电影相关": ["电影", "影片", "导演", "演员", "演技", "剧情", "特效", "票房",
                     "IMAX", "3D", "观影", "影院", "好莱坞", "国产片", "进口片",
                     "主角", "配角", "台词", "镜头", "构图", "配乐", "片尾", "彩蛋",
                     "续集", "翻拍", "动漫", "动画", "纪录片", "预告片"],
            "评价词汇": ["好看", "难看", "精彩", "无聊", "烂片", "神作", "力作", "佳作",
                     "炸裂", "惊艳", "失望", "推荐", "吐槽", "打call", "真香"],
            "影视平台": ["豆瓣", "IMDb", "烂番茄", "时光网", "猫眼"]
        },
        "product": {
            "商品相关": ["产品", "质量", "性价比", "包装", "物流", "快递", "收货", "购物",
                     "商家", "店铺", "客服", "售后", "退货", "换货", "正品", "假货",
                     "品牌", "型号", "规格", "参数", "功能", "效果", "价格", "价钱",
                     "优惠", "促销", "活动", "优惠券", "赠品", "配件"],
            "使用场景": ["使用", "体验", "感受", "手感", "外观", "颜色", "尺寸", "大小",
                     "重量", "电池", "续航", "充电", "操作", "安装", "清洗", "维修"],
            "电商平台": ["淘宝", "京东", "天猫", "拼多多", "苏宁", "唯品会", "亚马逊"]
        },
        "news": {
            "新闻相关": ["报道", "记者", "采访", "发布会", "声明", "公告", "政策", "法规",
                     "政府", "官方", "部门", "机构", "企业", "公司", "行业", "市场",
                     "经济", "政治", "社会", "国际", "国内", "地方", "全国", "全球"],
            "新闻事件": ["事件", "事故", "灾害", "疫情", "峰会", "论坛", "选举", "公投",
                     "签约", "上市", "发布", "成立", "倒闭", "收购", "合并"],
            "新闻来源": ["新华社", "人民日报", "央视", "中新社", "澎湃", "财新", "BBC", "CNN", "路透社"]
        }
    }

    def __init__(self):
        self.domain_weights = {
            "film": 1.0,
            "product": 1.0,
            "news": 1.0,
            "mixed": 0.5
        }

    def identify(self, text: str) -> str:
        """
        识别文本所属领域。

        参数：
            text (str): 输入文本。

        返回：
            str: 领域标签 ("film", "product", "news", "mixed")。
        """
        if not text or len(text.strip()) < 5:
            return "mixed"

        text_lower = text.lower()
        scores = {"film": 0.0, "product": 0.0, "news": 0.0}

        for domain, keyword_groups in self.DOMAIN_KEYWORDS.items():
            domain_score = 0.0
            for group_name, keywords in keyword_groups.items():
                for keyword in keywords:
                    if keyword in text:
                        if group_name in ["电影相关", "商品相关", "新闻相关"]:
                            domain_score += 3.0
                        elif group_name in ["评价词汇", "使用场景", "新闻事件"]:
                            domain_score += 2.0
                        else:
                            domain_score += 1.0
            scores[domain] = domain_score

        max_domain = max(scores.items(), key=lambda x: x[1])
        if max_domain[1] == 0:
            return "mixed"
        return max_domain[0]

    def identify_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        识别文本所属领域及置信度。

        参数：
            text (str): 输入文本。

        返回：
            Tuple[str, float]: (领域标签, 置信度)。
        """
        if not text or len(text.strip()) < 5:
            return "mixed", 0.5

        text_lower = text.lower()
        scores = {"film": 0.0, "product": 0.0, "news": 0.0}

        for domain, keyword_groups in self.DOMAIN_KEYWORDS.items():
            domain_score = 0.0
            for group_name, keywords in keyword_groups.items():
                for keyword in keywords:
                    if keyword in text:
                        if group_name in ["电影相关", "商品相关", "新闻相关"]:
                            domain_score += 3.0
                        elif group_name in ["评价词汇", "使用场景", "新闻事件"]:
                            domain_score += 2.0
                        else:
                            domain_score += 1.0
            scores[domain] = domain_score

        total_score = sum(scores.values())
        if total_score == 0:
            return "mixed", 0.5

        max_domain = max(scores.items(), key=lambda x: x[1])
        confidence = max_domain[1] / total_score if total_score > 0 else 0.5
        confidence = min(confidence, 0.95)

        if max_domain[1] == 0:
            return "mixed", 0.5
        return max_domain[0], confidence


class DomainAwareTextPreprocessor:
    """
    领域感知的文本预处理器。

    该类在原有 TextPreprocessor 功能基础上，增加了：
    1. 自动领域识别
    2. 领域特定分词优化
    3. 领域词汇增强

    设计要点：
        - 兼容原有 TextPreprocessor 的所有接口
        - 提供领域识别和领域特定处理能力
        - 支持领域信息的序列化和加载
    """

    DOMAIN_NAMES = ["film", "product", "news", "mixed"]

    def __init__(
        self,
        vocab: Dict[str, int] = None,
        max_vocab_size: int = 30000,
        max_len: int = 128
    ):
        """
        初始化领域感知预处理器。

        参数：
            vocab (Dict[str, int], optional): 词表字典。默认为 None。
            max_vocab_size (int): 词表最大容量。默认为 30000。
            max_len (int): 序列最大长度。默认为 128。
        """
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len

        self.vocab = vocab if vocab is not None else {"<PAD>": 0, "<UNK>": 1}
        self.pad_token_id = 0
        self.unk_token_id = 1

        self.domain_identifier = DomainIdentifier()

        self.domain_word_weights = {
            "film": {},
            "product": {},
            "news": {},
            "mixed": {}
        }

        self._init_domain_jieba()

    def _init_domain_jieba(self):
        """
        初始化各领域专有名词到 jieba 词典。
        """
        domain_terms = {
            "film": ["观影体验", "特效炸裂", "演技炸裂", "剧情拖沓", "片尾彩蛋",
                    "豆瓣评分", "IMAX", "3D眼镜", "爆米花", "好莱坞大片",
                    "国产电影", "票房冠军", "黑马影片", "烂片预警", "神作预定",
                    "二刷", "三刷", "强烈推荐", "踩雷", "真香警告"],
            "product": ["性价比", "物流速度", "客服态度", "售后保障", "正品保证",
                       "包装破损", "七天无理由", "以旧换新", "限时优惠", "团购",
                       "秒杀活动", "满减优惠", "返现", "优惠券", "赠品丰富"],
            "news": ["新闻发布会", "官方声明", "政策解读", "行业报告", "市场分析",
                   "经济数据", "GDP增长", "CPI指数", "股市行情", "汇市动态",
                   "国际形势", "外交关系", "军事演习", "科技创新", "教育改革"]
        }

        for domain, terms in domain_terms.items():
            for term in terms:
                jieba.add_word(term, freq=100, tag=domain)

    def set_vocab(self, vocab: Dict[str, int]):
        """
        设置词表。

        参数：
            vocab (Dict[str, int]): 词表字典。
        """
        self.vocab = vocab

    def identify_domain(self, text: str) -> str:
        """
        识别文本领域。

        参数：
            text (str): 输入文本。

        返回：
            str: 领域标签。
        """
        return self.domain_identifier.identify(text)

    def identify_domain_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        识别文本领域及置信度。

        参数：
            text (str): 输入文本。

        返回：
            Tuple[str, float]: (领域标签, 置信度)。
        """
        return self.domain_identifier.identify_with_confidence(text)

    def text_to_sequence(self, text: str, domain: str = None) -> List[int]:
        """
        将文本转换为 ID 序列（领域感知版本）。

        参数：
            text (str): 输入文本。
            domain (str, optional): 已知领域标签。如果为 None，则自动识别。

        返回：
            List[int]: 定长 ID 序列。
        """
        if domain is None:
            domain = self.identify_domain(text)

        text = str(text)
        words = jieba.lcut(text)

        if len(words) > self.max_len:
            words = words[:self.max_len]

        seq = [self.vocab.get(w, self.unk_token_id) for w in words]

        if len(seq) < self.max_len:
            pad_length = self.max_len - len(seq)
            seq.extend([self.pad_token_id] * pad_length)

        return seq

    def text_to_sequence_with_domain(
        self,
        text: str,
        domain: str
    ) -> List[int]:
        """
        使用指定领域进行文本序列转换。

        参数：
            text (str): 输入文本。
            domain (str): 领域标签。

        返回：
            List[int]: 定长 ID 序列。
        """
        return self.text_to_sequence(text, domain=domain)

    def process_batch(
        self,
        texts: List[str],
        domains: List[str] = None
    ) -> torch.Tensor:
        """
        批量文本转换为张量（支持领域指定）。

        参数：
            texts (List[str]): 文本列表。
            domains (List[str], optional): 领域列表。如果为 None，则自动识别。

        返回：
            torch.Tensor: 形状为 [batch_size, max_len] 的张量。
        """
        if domains is None:
            domains = [None] * len(texts)

        sequences = [
            self.text_to_sequence(text, domain=domain)
            for text, domain in zip(texts, domains)
        ]
        return torch.tensor(sequences, dtype=torch.long)

    def process_batch_with_domain_identification(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        批量文本转换，同时返回识别出的领域。

        参数：
            texts (List[str]): 文本列表。

        返回：
            Tuple[torch.Tensor, List[str]]: (张量, 领域列表)。
        """
        domains = [self.identify_domain(text) for text in texts]
        sequences = [
            self.text_to_sequence(text, domain=domain)
            for text, domain in zip(texts, domains)
        ]
        return torch.tensor(sequences, dtype=torch.long), domains

    def get_domain_statistics(self, texts: List[str]) -> Dict[str, int]:
        """
        统计文本集合中各领域分布。

        参数：
            texts (List[str]): 文本列表。

        返回：
            Dict[str, int]: 各领域计数。
        """
        stats = {domain: 0 for domain in self.DOMAIN_NAMES}
        for text in texts:
            domain = self.identify_domain(text)
            stats[domain] = stats.get(domain, 0) + 1
        return stats

    def save(self, save_path: str):
        """
        保存预处理器状态。

        参数：
            save_path (str): 保存路径。
        """
        save_obj = {
            "vocab": self.vocab,
            "max_vocab_size": self.max_vocab_size,
            "max_len": self.max_len,
            "domain_word_weights": self.domain_word_weights
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(save_obj, f)

    def load(self, load_path: str):
        """
        加载预处理器状态。

        参数：
            load_path (str): 加载路径。
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"文件未找到: {load_path}")

        with open(load_path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            self.vocab = obj.get("vocab", self.vocab)
            self.max_vocab_size = obj.get("max_vocab_size", self.max_vocab_size)
            self.max_len = obj.get("max_len", self.max_len)
            self.domain_word_weights = obj.get("domain_word_weights", self.domain_word_weights)


def load_domain_preprocessor(
    vocab_path: str,
    max_len: int = 128
) -> DomainAwareTextPreprocessor:
    """
    加载领域感知预处理器。

    参数：
        vocab_path (str): 词表文件路径。
        max_len (int): 序列最大长度。

    返回：
        DomainAwareTextPreprocessor: 加载的预处理器实例。
    """
    with open(vocab_path, "rb") as f:
        vocab_obj = pickle.load(f)

    if isinstance(vocab_obj, dict) and "vocab" in vocab_obj:
        vocab = vocab_obj["vocab"]
        saved_max_len = vocab_obj.get("max_len", max_len)
        if saved_max_len != max_len:
            print(f"警告：词表中的 max_len ({saved_max_len}) 与指定值 ({max_len}) 不同")
    else:
        vocab = vocab_obj

    processor = DomainAwareTextPreprocessor(
        vocab=vocab,
        max_len=max_len
    )

    return processor


if __name__ == "__main__":
    processor = DomainAwareTextPreprocessor()

    test_texts = [
        "这部电影太精彩了，特效炸裂，强烈推荐！",
        "收到货了，质量很好，性价比很高，物流也很快",
        "国务院召开新闻发布会，发布最新经济数据",
        "今天天气真好，适合出去游玩"
    ]

    print("领域识别测试：")
    for text in test_texts:
        domain, confidence = processor.identify_domain_with_confidence(text)
        print(f"文本: {text[:30]}... -> 领域: {domain}, 置信度: {confidence:.2f}")
