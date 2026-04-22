from flask import Flask, render_template, request, jsonify
import os
import sys
import torch
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.bilstm_attention import SentimentModel
from src.data.preprocess import TextPreprocessor
from src.data.domain_preprocessor import DomainAwareTextPreprocessor, DomainIdentifier

app = Flask(__name__)

model_path = 'checkpoints/best_model.pth'
vocab_path = 'dataset/processed/vocab.pkl'

model = None
preprocessor = None
domain_identifier = None
device = None
model_loaded = False

def load_vocab(vocab_path):
    import pickle
    with open(vocab_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'vocab' in obj:
        return obj['vocab']
    return obj

def init_model():
    global model, preprocessor, domain_identifier, device, model_loaded

    try:
        vocab = load_vocab(vocab_path)

        vocab_size = len(vocab)
        embedding_dim = 128
        hidden_dim = 256
        output_dim = 1
        n_layers = 2
        dropout = 0.5
        max_len = 128

        model = SentimentModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            model_loaded = True
            print("信息：模型加载成功")
        else:
            print(f"警告：模型文件未找到：{model_path}")

        domain_identifier = DomainIdentifier()
        preprocessor = TextPreprocessor(max_len=max_len)
        preprocessor.vocab = vocab

        print("信息：领域识别器初始化成功")

    except Exception as e:
        print(f"初始化错误：{str(e)}")

init_model()

def preprocess_text(text):
    if preprocessor is None:
        tokens = list(jieba.cut(text))
        return tokens, [1] * len(tokens)
    seq = preprocessor.text_to_sequence(text)
    tokens = list(jieba.cut(text))[:128]
    return tokens, seq

def predict(text):
    global model, device, model_loaded

    if not model_loaded or model is None:
        raise Exception("模型未加载")

    tokens, input_ids = preprocess_text(text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits, _ = model(input_tensor)
        prob = torch.sigmoid(logits).item()

    return 'positive' if prob >= 0.5 else 'negative', round(prob, 4), tokens

@app.route('/')
def index():
    return render_template('domain_analysis.html')

@app.route('/api/domain/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'domain_support': True,
        'available_domains': ['film', 'product', 'news', 'mixed']
    })

@app.route('/api/domain/analyze', methods=['POST'])
def analyze():
    if domain_identifier is None:
        return jsonify({'error': '领域识别器未初始化'}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '缺少text字段'}), 400

    text = data['text']
    if not text or len(text.strip()) < 5:
        return jsonify({'error': '文本长度至少为5个字符'}), 400

    try:
        domain, confidence = domain_identifier.identify_with_confidence(text)
        sentiment, prob, tokens = predict(text)
        conf = prob if prob >= 0.5 else 1 - prob

        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'sentiment_score': prob,
            'confidence': conf,
            'domain': domain,
            'detected_domain': domain,
            'domain_confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/domain/analyze_with_domain', methods=['POST'])
def analyze_with_domain():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '缺少text字段'}), 400

    text = data['text']
    domain = data.get('domain', 'mixed')

    if domain not in ['film', 'product', 'news', 'mixed']:
        return jsonify({'error': f'无效的领域: {domain}'}), 400

    try:
        sentiment, prob, tokens = predict(text)
        conf = prob if prob >= 0.5 else 1 - prob

        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'sentiment_score': prob,
            'confidence': conf,
            'domain': domain
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/domain/batch_analyze', methods=['POST'])
def batch_analyze():
    if domain_identifier is None:
        return jsonify({'error': '领域识别器未初始化'}), 503

    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': '缺少texts字段'}), 400

    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'texts必须是非空列表'}), 400

    if len(texts) > 100:
        return jsonify({'error': '批量大小不能超过100'}), 400

    auto_detect = data.get('auto_detect_domain', True)

    try:
        results = []
        domain_stats = {'film': 0, 'product': 0, 'news': 0, 'mixed': 0}
        positive_count = 0
        negative_count = 0

        for text in texts:
            if auto_detect:
                domain, _ = domain_identifier.identify_with_confidence(text)
            else:
                domain = 'mixed'

            sentiment, prob, tokens = predict(text)
            conf = prob if prob >= 0.5 else 1 - prob

            results.append({
                'text': text,
                'sentiment': sentiment,
                'sentiment_score': prob,
                'confidence': conf,
                'domain': domain
            })

            domain_stats[domain] = domain_stats.get(domain, 0) + 1
            if sentiment == 'positive':
                positive_count += 1
            else:
                negative_count += 1

        return jsonify({
            'results': results,
            'summary': {
                'total': len(results),
                'positive': positive_count,
                'negative': negative_count,
                'domain_distribution': domain_stats
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/domain/detect_domain', methods=['POST'])
def detect_domain():
    if domain_identifier is None:
        return jsonify({'error': '领域识别器未初始化'}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '缺少text字段'}), 400

    text = data['text']
    if not text:
        return jsonify({'error': '文本不能为空'}), 400

    try:
        domain, confidence = domain_identifier.identify_with_confidence(text)

        all_domains = {}
        for d in ['film', 'product', 'news', 'mixed']:
            if d == domain:
                all_domains[d] = confidence
            else:
                all_domains[d] = round((1 - confidence) / 3, 4)

        return jsonify({
            'text': text,
            'domain': domain,
            'confidence': confidence,
            'all_domains': all_domains
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/domain/stats', methods=['GET'])
def stats():
    return jsonify({
        'supported_domains': [
            {'name': 'film', 'description': '影评领域 - 电影、电视剧、综艺等评论'},
            {'name': 'product', 'description': '商品评论领域 - 购物、使用体验等评论'},
            {'name': 'news', 'description': '新闻领域 - 时事、政策、财经等新闻'},
            {'name': 'mixed', 'description': '混合/通用领域 - 未分类或跨领域文本'}
        ],
        'model_info': {
            'domain_adaptive': model_loaded,
            'auto_detection': True,
            'batch_support': True,
            'max_batch_size': 100
        }
    })

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=False, host='0.0.0.0', port=5003)
