"""
domain_api.py

跨领域情感分析 Flask API 服务。

提供以下接口：
- POST /api/domain/analyze: 情感分析（自动领域识别）
- POST /api/domain/analyze_with_domain: 指定领域情感分析
- POST /api/domain/batch_analyze: 批量情感分析
- GET /api/domain/stats: 获取领域分布统计
- GET /api/domain/health: 健康检查
"""

from flask import Flask, request, jsonify
import os
import sys

app = Flask(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.predict_domain import DomainPredictor, load_predictor
    from src.data.domain_preprocessor import DomainAwareTextPreprocessor

    predictor = None
    preprocessor = None
    api_loaded = False
    load_error = None

    try:
        predictor = load_predictor(
            model_path="checkpoints/best_model.pth",
            vocab_path="dataset/processed/vocab.pkl",
            max_len=128
        )
        preprocessor = DomainAwareTextPreprocessor()
        preprocessor.load_vocab("dataset/processed/vocab.pkl")
        api_loaded = True
        print("信息：领域感知模型加载成功")
    except Exception as e:
        load_error = str(e)
        print(f"警告：领域感知模型加载失败: {e}")
        print("API 将使用降级模式（基础情感分析）")

except ImportError as e:
    load_error = f"导入错误: {str(e)}"
    print(f"错误：无法导入必要模块: {e}")


@app.route('/api/domain/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok' if api_loaded else 'degraded',
        'model_loaded': api_loaded,
        'domain_support': True,
        'available_domains': ['film', 'product', 'news', 'mixed'],
        'error': load_error
    })


@app.route('/api/domain/analyze', methods=['POST'])
def analyze():
    """
    情感分析接口（自动领域识别）

    请求体:
    {
        "text": "要分析的文本"
    }

    返回:
    {
        "text": "原始文本",
        "sentiment": "positive/negative",
        "sentiment_score": 0.95,
        "confidence": 0.90,
        "domain": "film",
        "detected_domain": "film",
        "domain_confidence": 0.85
    }
    """
    if not api_loaded or predictor is None:
        return jsonify({'error': '模型未加载', 'detail': load_error}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '缺少text字段'}), 400

    text = data['text']
    if not text or len(text.strip()) < 5:
        return jsonify({'error': '文本长度至少为5个字符'}), 400

    try:
        result = predictor.predict(text, auto_detect_domain=True)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500


@app.route('/api/domain/analyze_with_domain', methods=['POST'])
def analyze_with_domain():
    """
    指定领域情感分析接口

    请求体:
    {
        "text": "要分析的文本",
        "domain": "film|product|news|mixed"
    }

    返回:
    {
        "text": "原始文本",
        "sentiment": "positive/negative",
        "sentiment_score": 0.95,
        "confidence": 0.90,
        "domain": "film"
    }
    """
    if not api_loaded or predictor is None:
        return jsonify({'error': '模型未加载', 'detail': load_error}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '缺少text字段'}), 400

    text = data['text']
    domain = data.get('domain', 'mixed')

    if domain not in ['film', 'product', 'news', 'mixed']:
        return jsonify({'error': f'无效的领域: {domain}，可选值: film, product, news, mixed'}), 400

    if not text or len(text.strip()) < 5:
        return jsonify({'error': '文本长度至少为5个字符'}), 400

    try:
        result = predictor.predict(text, domain=domain, auto_detect_domain=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500


@app.route('/api/domain/batch_analyze', methods=['POST'])
def batch_analyze():
    """
    批量情感分析接口

    请求体:
    {
        "texts": ["文本1", "文本2", ...],
        "domains": ["film", "product", ...],  // 可选
        "auto_detect_domain": true  // 可选，默认true
    }

    返回:
    {
        "results": [
            {...},
            {...}
        ],
        "summary": {
            "total": 10,
            "positive": 7,
            "negative": 3,
            "domain_distribution": {
                "film": 5,
                "product": 3,
                "news": 1,
                "mixed": 1
            }
        }
    }
    """
    if not api_loaded or predictor is None:
        return jsonify({'error': '模型未加载', 'detail': load_error}), 503

    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': '缺少texts字段'}), 400

    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'texts必须是非空列表'}), 400

    if len(texts) > 100:
        return jsonify({'error': '批量大小不能超过100'}), 400

    domains = data.get('domains', None)
    auto_detect = data.get('auto_detect_domain', True)

    if domains is not None and len(domains) != len(texts):
        return jsonify({'error': 'texts和domains长度必须一致'}), 400

    try:
        results = predictor.predict_batch(
            texts=texts,
            domains=domains,
            auto_detect_domain=auto_detect
        )

        summary = {
            'total': len(results),
            'positive': sum(1 for r in results if r['sentiment'] == 'positive'),
            'negative': sum(1 for r in results if r['sentiment'] == 'negative'),
            'domain_distribution': {}
        }

        for r in results:
            domain = r.get('detected_domain', r.get('domain', 'mixed'))
            summary['domain_distribution'][domain] = summary['domain_distribution'].get(domain, 0) + 1

        return jsonify({
            'results': results,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': f'批量预测失败: {str(e)}'}), 500


@app.route('/api/domain/detect_domain', methods=['POST'])
def detect_domain():
    """
    领域识别接口

    请求体:
    {
        "text": "要识别的文本"
    }

    返回:
    {
        "text": "原始文本",
        "domain": "film",
        "confidence": 0.85,
        "all_domains": {
            "film": 0.85,
            "product": 0.10,
            "news": 0.03,
            "mixed": 0.02
        }
    }
    """
    if preprocessor is None:
        return jsonify({'error': '预处理器未加载'}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '缺少text字段'}), 400

    text = data['text']
    if not text:
        return jsonify({'error': '文本不能为空'}), 400

    try:
        domain, confidence = preprocessor.identify_domain_with_confidence(text)

        all_domains = {}
        for d in ['film', 'product', 'news', 'mixed']:
            if d == domain:
                all_domains[d] = confidence
            else:
                all_domains[d] = round((1 - confidence) / 3, 4)

        return jsonify({
            'text': text,
            'domain': domain,
            'confidence': round(confidence, 4),
            'all_domains': all_domains
        })
    except Exception as e:
        return jsonify({'error': f'领域识别失败: {str(e)}'}), 500


@app.route('/api/domain/stats', methods=['GET'])
def get_domain_stats():
    """
    获取支持的领域统计信息

    返回:
    {
        "supported_domains": [
            {"name": "film", "description": "影评领域"},
            {"name": "product", "description": "商品评论领域"},
            {"name": "news", "description": "新闻领域"},
            {"name": "mixed", "description": "混合/通用领域"}
        ],
        "model_info": {
            "domain_adaptive": True,
            "auto_detection": True
        }
    }
    """
    return jsonify({
        'supported_domains': [
            {'name': 'film', 'description': '影评领域 - 电影、电视剧、综艺等评论'},
            {'name': 'product', 'description': '商品评论领域 - 购物、使用体验等评论'},
            {'name': 'news', 'description': '新闻领域 - 时事、政策、财经等新闻'},
            {'name': 'mixed', 'description': '混合/通用领域 - 未分类或跨领域文本'}
        ],
        'model_info': {
            'domain_adaptive': api_loaded,
            'auto_detection': True,
            'batch_support': True,
            'max_batch_size': 100
        }
    })


@app.route('/', methods=['GET'])
def index():
    """API首页"""
    return jsonify({
        'name': '跨领域情感分析 API',
        'version': '1.0.0',
        'endpoints': [
            '/api/domain/health',
            '/api/domain/analyze',
            '/api/domain/analyze_with_domain',
            '/api/domain/batch_analyze',
            '/api/domain/detect_domain',
            '/api/domain/stats'
        ]
    })


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=False, host='0.0.0.0', port=5002)
