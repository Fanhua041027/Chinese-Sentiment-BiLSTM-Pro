"""
test_domain_adaptation.py

跨领域适应功能测试脚本。

测试内容：
1. 领域识别功能
2. 领域感知预处理
3. 领域自适应模型
4. 批量预测
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_domain_identifier():
    """测试领域识别器"""
    print("\n" + "=" * 60)
    print("测试 1: 领域识别器")
    print("=" * 60)

    from src.data.domain_preprocessor import DomainIdentifier

    identifier = DomainIdentifier()

    test_cases = [
        ("这部电影太精彩了，特效炸裂，强烈推荐大家去看！", "film"),
        ("收到货了，质量很好，性价比很高，会回购的", "product"),
        ("国务院召开新闻发布会，发布最新经济政策", "news"),
        ("今天天气真好，心情也很不错", "mixed"),
        ("演员演技很棒，剧情紧凑，很值得一看", "film"),
        ("物流很快，包装完好，产品质量不错", "product"),
        ("央行宣布最新货币政策，影响金融市场", "news"),
        ("这是一个普通的日子，没什么特别的", "mixed"),
    ]

    correct = 0
    for text, expected in test_cases:
        domain, confidence = identifier.identify_with_confidence(text)
        status = "✓" if domain == expected else "✗"
        if domain == expected:
            correct += 1
        print(f"{status} 文本: {text[:30]}...")
        print(f"  预期领域: {expected}, 识别领域: {domain}, 置信度: {confidence:.2f}")
        print()

    accuracy = correct / len(test_cases) * 100
    print(f"领域识别准确率: {accuracy:.1f}% ({correct}/{len(test_cases)})")
    return accuracy >= 75.0


def test_domain_preprocessor():
    """测试领域预处理器"""
    print("\n" + "=" * 60)
    print("测试 2: 领域预处理器")
    print("=" * 60)

    from src.data.domain_preprocessor import DomainAwareTextPreprocessor

    preprocessor = DomainAwareTextPreprocessor()

    test_texts = [
        "这部电影太精彩了",
        "收到货了质量很好",
        "国务院召开新闻发布会",
        "今天天气真好"
    ]

    print("测试文本序列化：")
    for text in test_texts:
        domain, confidence = preprocessor.identify_domain_with_confidence(text)
        seq = preprocessor.text_to_sequence(text, domain=domain)
        print(f"  文本: {text}")
        print(f"  领域: {domain}, 序列长度: {len(seq)}")
        print(f"  前5个token ID: {seq[:5]}")
        print()

    texts_batch = ["电影很精彩", "产品质量不错", "新闻发布会召开", "天气很好"]
    print("测试批量处理：")
    sequences, domains = preprocessor.process_batch_with_domain_identification(texts_batch)
    print(f"  批量张量形状: {sequences.shape}")
    print(f"  识别领域: {domains}")
    print()

    print("测试领域统计：")
    stats = preprocessor.get_domain_statistics(texts_batch)
    print(f"  领域分布: {stats}")
    print()

    return True


def test_domain_model():
    """测试领域自适应模型"""
    print("\n" + "=" * 60)
    print("测试 3: 领域自适应模型")
    print("=" * 60)

    import torch
    from src.models.domain_adaptive_model import (
        DomainAdaptiveModel,
        MultiTaskDomainModel,
        create_domain_adaptive_model
    )
    from src.models.bilstm_attention import SentimentModel

    vocab_size = 30000
    batch_size = 4
    seq_len = 128

    print("测试 DomainAdaptiveModel：")
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

    for domain_name in ["film", "product", "news", "mixed"]:
        logits, attn = model.forward_with_domain_name(text, domain_name)
        print(f"  领域 {domain_name}: logits shape {logits.shape}, attention shape {attn.shape}")

    print("\n测试无领域信息（通用模式）：")
    logits, attn = model(text, domain=None)
    print(f"  logits shape: {logits.shape}, attention shape: {attn.shape}")

    print("\n测试特征提取：")
    features = model.extract_domain_features(text)
    print(f"  提取特征键: {list(features.keys())}")
    print()

    print("测试 MultiTaskDomainModel：")
    multi_task_model = MultiTaskDomainModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=1,
        n_layers=2,
        dropout=0.5
    )

    sent_logits, domain_logits, attn = multi_task_model(text)
    print(f"  情感 logits shape: {sent_logits.shape}")
    print(f"  领域 logits shape: {domain_logits.shape}")
    print(f"  领域预测: {domain_logits.argmax(dim=-1).tolist()}")
    print()

    print("测试从基础模型创建领域自适应模型：")
    try:
        base_model = SentimentModel(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=1,
            n_layers=2,
            dropout=0.5
        )
        adapted_model = create_domain_adaptive_model(base_model)
        print(f"  成功创建领域自适应模型")
        print(f"  模型参数数量: {sum(p.numel() for p in adapted_model.parameters()):,}")
    except Exception as e:
        print(f"  创建失败: {e}")
    print()

    return True


def test_domain_predictor():
    """测试领域预测器"""
    print("\n" + "=" * 60)
    print("测试 4: 领域预测器（需要模型文件）")
    print("=" * 60)

    from src.predict_domain import DomainPredictor

    model_path = "checkpoints/best_model.pth"
    vocab_path = "dataset/processed/vocab.pkl"

    if not os.path.exists(model_path):
        print(f"  跳过: 模型文件不存在 ({model_path})")
        print("  提示: 请先训练模型以启用完整预测功能")
        return True

    try:
        predictor = DomainPredictor(
            model_path=model_path,
            vocab_path=vocab_path
        )

        test_texts = [
            "这部电影太精彩了，特效炸裂，强烈推荐！",
            "收到货了，质量很好，性价比很高",
            "国务院召开新闻发布会，最新政策发布",
            "今天天气真好，适合出去游玩"
        ]

        print("测试单条预测：")
        for text in test_texts:
            result = predictor.predict(text)
            print(f"  文本: {text[:30]}...")
            print(f"  领域: {result['domain']}, 情感: {result['sentiment']}")
            print(f"  置信度: {result['confidence']:.4f}")
            print()

        print("测试批量预测：")
        results = predictor.predict_batch(test_texts)
        print(f"  批量预测完成: {len(results)} 条")
        print()

        print("测试领域分布统计：")
        stats = predictor.analyze_domain_distribution(test_texts)
        print(f"  总数: {stats['total']}")
        print(f"  分布: {stats['distribution']}")
        print()

    except Exception as e:
        print(f"  预测器初始化失败: {e}")
        return False

    return True


def test_api():
    """测试API功能"""
    print("\n" + "=" * 60)
    print("测试 5: Flask API（需要启动服务）")
    print("=" * 60)

    print("  API 端点：")
    print("    - GET  /api/domain/health          : 健康检查")
    print("    - POST /api/domain/analyze          : 自动领域识别情感分析")
    print("    - POST /api/domain/analyze_with_domain : 指定领域情感分析")
    print("    - POST /api/domain/batch_analyze    : 批量情感分析")
    print("    - POST /api/domain/detect_domain    : 领域识别")
    print("    - GET  /api/domain/stats            : 领域统计信息")
    print()
    print("  启动命令: python domain_api.py")
    print()

    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("跨领域适应功能测试")
    print("=" * 60)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    results = {}

    try:
        results["领域识别器"] = test_domain_identifier()
    except Exception as e:
        print(f"  错误: {e}")
        results["领域识别器"] = False

    try:
        results["领域预处理器"] = test_domain_preprocessor()
    except Exception as e:
        print(f"  错误: {e}")
        results["领域预处理器"] = False

    try:
        results["领域自适应模型"] = test_domain_model()
    except Exception as e:
        print(f"  错误: {e}")
        results["领域自适应模型"] = False

    try:
        results["领域预测器"] = test_domain_predictor()
    except Exception as e:
        print(f"  错误: {e}")
        results["领域预测器"] = False

    try:
        results["Flask API"] = test_api()
    except Exception as e:
        print(f"  错误: {e}")
        results["Flask API"] = False

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！跨领域适应功能已就绪。")
    else:
        print("部分测试失败，请检查上述输出。")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
